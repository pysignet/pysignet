"""Convenience API for logic compilation."""

from typing import Callable, Dict, Optional

import sympy as sp
import torch

from pysignet.predicate import Predicate
from pysignet.compilation import TNormCompiler
from pysignet.compilation.compiled_expression import CompiledExpression
from pysignet.loss import LogicLoss
from pysignet.tnorms import TNorm, RProductTNorm


def compile_logic(
    expr: sp.Basic,
    predicates: Dict[str, Predicate | Callable[..., torch.Tensor]],
    mode: str = "tnorm",
    tnorm: Optional[TNorm] = None,
) -> CompiledExpression:
    """Compile logic expression into a CompiledExpression.

    This is the main entry point for most users. It compiles a SymPy logic
    expression into a CompiledExpression that can evaluate satisfaction
    degrees per-batch. Wrap the result in LogicLoss for loss computation
    and batch quantification.

    Args:
        expr: SymPy logic expression (e.g., sp.And(P(X), Q(X)))
        predicates: Dict mapping predicate names to Predicate objects or
            callables that produce torch Tensors
        mode: Compilation mode - 'tnorm' (default), or 'semantic' (future)
        tnorm: T-norm for mode='tnorm' (default: RProductTNorm)

    Returns:
        CompiledExpression instance for evaluating satisfaction degrees

    Raises:
        ValueError: If unknown mode specified

    Example:
        Direct satisfaction evaluation::

            >>> P, Q = Symbol("P Q")
            >>> X = Variable("X")
            >>> expr = sp.And(P(X), Q(X))
            >>> compiled = compile_logic(expr, {"P": model_p, "Q": model_q})
            >>> satisfaction = compiled(X=x)  # shape: (batch_size,)

        Wrap in LogicLoss for training::

            >>> compiled = compile_logic(expr, predicates)
            >>> logic_loss = LogicLoss(compiled)
            >>> loss = logic_loss.loss(X=x)

        With custom t-norm::

            >>> from pysignet.tnorms import LukasiewiczTNorm
            >>> compiled = compile_logic(
            ...     expr, predicates,
            ...     tnorm=LukasiewiczTNorm()
            ... )
    """
    # Auto-wrap raw callables in Predicate objects
    wrapped_predicates: Dict[
        str, Predicate | Callable[..., torch.Tensor]
    ] = {}
    for key, value in predicates.items():
        if isinstance(value, Predicate):
            # Already a Predicate, use as-is
            wrapped_predicates[key] = value
        elif callable(value):
            # Raw callable (function, lambda, nn.Module) - auto-wrap
            wrapped_predicates[key] = Predicate(value)
        else:
            # Not callable - raise helpful error
            raise TypeError(
                f"Predicate '{key}' must be callable (function, lambda, "
                f"nn.Module) or a Predicate instance, "
                f"got {type(value).__name__}"
            )

    if mode == "tnorm":
        # Create t-norm compiler
        tnorm_instance = tnorm or RProductTNorm()
        compiler = TNormCompiler(tnorm=tnorm_instance)
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Expected 'tnorm'. "
            + "(Future: 'semantic', 'kenn')"
        )

    # Compile the expression with wrapped predicates
    # Returns CompiledExpression (with compiler reference)
    return compiler.compile(expr, wrapped_predicates)


def logic_to_loss(
    expr: sp.Basic,
    predicates: Dict[str, Predicate | Callable[..., torch.Tensor]],
    mode: str = "tnorm",
    tnorm: Optional[TNorm] = None,
    post_processing: (
        str | Callable[[torch.Tensor], torch.Tensor] | None
    ) = None,
) -> LogicLoss:
    """Compile logic expression and wrap in LogicLoss.

    Convenience function that compiles a logic expression and wraps it
    in a LogicLoss for training. Equivalent to::

        compiled = compile_logic(expr, predicates, mode=mode, tnorm=tnorm)
        LogicLoss(compiled, post_processing=post_processing)

    Args:
        expr: SymPy logic expression (e.g., sp.And(P(X), Q(X)))
        predicates: Dict mapping predicate names to Predicate objects or
            callables that produce torch Tensors
        mode: Compilation mode - 'tnorm' (default)
        tnorm: T-norm for mode='tnorm' (default: RProductTNorm)
        post_processing: Post-processing mode - 'log', 'linear', callable,
            or None to use t-norm's recommendation (default)

    Returns:
        LogicLoss instance ready for computing satisfaction and loss

    Example:
        >>> P, Q = Symbol("P Q")
        >>> X = Variable("X")
        >>> expr = sp.Implies(P(X), Q(X))
        >>> logic_loss = logic_to_loss(expr, {"P": model_p, "Q": model_q})
        >>> loss = logic_loss.loss(X=x)
    """
    compiled = compile_logic(
        expr, predicates, mode=mode, tnorm=tnorm
    )
    return LogicLoss(compiled, post_processing=post_processing)
