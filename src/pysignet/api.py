"""Convenience API for logic compilation and evaluation."""

from typing import Callable, Dict, Optional, Union

import sympy as sp
import torch

from pysignet.predicate import Predicate
from pysignet.compilation import TNormCompiler, LinearThresholdUnitCompiler
from pysignet.compilation.compiled_expression import CompiledExpression
from pysignet.eval.report import ConsistencyReport
from pysignet.loss import LogicLoss
from pysignet.tnorms import TNorm, MixedTNorm


def compile_logic(
    expr: sp.Basic,
    predicates: Dict[str, Predicate | Callable[..., torch.Tensor]],
    mode: str = "tnorm",
    tnorm: Optional[TNorm] = None,
    alpha: float = 1.0,
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
        mode: Compilation mode - 'tnorm' (default) or 'ltu'
        tnorm: T-norm for mode='tnorm' (default: MixedTNorm). Ignored
            for other modes.
        alpha: Sigmoid sharpness for mode='ltu' (default: 1.0). Larger
            values make AND/OR thresholds sharper.

    Returns:
        CompiledExpression instance for evaluating satisfaction degrees

    Raises:
        ValueError: If unknown mode specified, or tnorm= given with
            mode='ltu'

    Examples:
        Default (MixedTNorm):

        ```python
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))
        compiled = compile_logic(expr, {"P": model_p, "Q": model_q})
        satisfaction = compiled(X=x)  # shape: (batch_size,)
        ```

        With a custom t-norm:

        ```python
        from pysignet.tnorms import LukasiewiczTNorm
        compiled = compile_logic(expr, predicates, tnorm=LukasiewiczTNorm())
        ```

        With the LTU compiler:

        ```python
        compiled = compile_logic(expr, predicates, mode='ltu', alpha=2.0)
        ```
    """
    # Auto-wrap raw callables in Predicate objects
    wrapped_predicates: Dict[str, Predicate | Callable[..., torch.Tensor]] = {}
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
        tnorm_instance = tnorm or MixedTNorm()
        compiler: TNormCompiler | LinearThresholdUnitCompiler = (
            TNormCompiler(tnorm=tnorm_instance)
        )
    elif mode == "ltu":
        if tnorm is not None:
            raise ValueError(
                "tnorm= is only valid with mode='tnorm'. "
                "Use alpha= to configure the LTU compiler."
            )
        compiler = LinearThresholdUnitCompiler(mode="soft", alpha=alpha)
    else:
        raise NotImplementedError(
            f"Mode '{mode}' is not yet implemented. "
            f"Supported modes: 'tnorm', 'ltu'."
        )

    # Compile the expression with wrapped predicates
    # Returns CompiledExpression (with compiler reference)
    return compiler.compile(expr, wrapped_predicates)


def logic_to_loss(
    expr: sp.Basic,
    predicates: Dict[str, Predicate | Callable[..., torch.Tensor]],
    mode: str = "tnorm",
    tnorm: Optional[TNorm] = None,
    alpha: float = 1.0,
    post_processing: str | Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> LogicLoss:
    """Compile logic expression and wrap in LogicLoss.

    Convenience function that compiles a logic expression and wraps it
    in a LogicLoss for training. Equivalent to:

        compiled = compile_logic(expr, predicates, mode=mode, tnorm=tnorm,
                                 alpha=alpha)
        LogicLoss(compiled, post_processing=post_processing)

    Args:
        expr: SymPy logic expression (e.g., sp.And(P(X), Q(X)))
        predicates: Dict mapping predicate names to Predicate objects or
            callables that produce torch Tensors
        mode: Compilation mode - 'tnorm' (default) or 'ltu'
        tnorm: T-norm for mode='tnorm' (default: MixedTNorm). Ignored
            for other modes.
        alpha: Sigmoid sharpness for mode='ltu' (default: 1.0).
        post_processing: Post-processing mode - 'log', 'linear', callable,
            or None to use the compiler's recommendation (default)

    Returns:
        LogicLoss instance ready for computing satisfaction and loss

    Examples:
        ```python
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Implies(P(X), Q(X))
        logic_loss = logic_to_loss(expr, {"P": model_p, "Q": model_q})
        loss = logic_loss.loss(X=x)
        ```

        With LTU compiler:

        ```python
        logic_loss = logic_to_loss(expr, predicates, mode='ltu', alpha=2.0)
        ```
    """
    compiled = compile_logic(expr, predicates, mode=mode, tnorm=tnorm,
                             alpha=alpha)
    return LogicLoss(compiled, post_processing=post_processing)


def consistency_report(
    expression: Union[
        sp.Basic,
        Dict[str, sp.Basic],
    ],
    predicates: Dict[str, Predicate | Callable[..., torch.Tensor]],
) -> ConsistencyReport:
    """Create a ConsistencyReport for measuring formula consistency.

    Convenience function that auto-wraps raw callables in Predicate
    objects and creates a ConsistencyReport. Equivalent to:

        ConsistencyReport(expression, predicates)

    Accepts a single SymPy expression or a dict mapping constraint
    names to expressions for multi-constraint reporting.

    The antecedent for conditional violation is auto-detected:
    Implies(A, B) uses A; any other formula uses sp.true.

    Args:
        expression: SymPy logic expression or dict of named
            expressions (e.g., {"sym": expr1, "trans": expr2}).
        predicates: Dict mapping predicate names to Predicate objects or
            callables that produce torch Tensors

    Returns:
        ConsistencyReport instance for accumulating and querying metrics

    Example:
        ```python
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Implies(P(X), Q(X))
        report = consistency_report(expr, {"P": model_p, "Q": model_q})
        for x_batch in dataloader:
            report.eval(X=x_batch)
        print(report.global_violation())
        ```
    """
    wrapped_predicates: Dict[str, Predicate] = {}
    for key, value in predicates.items():
        if isinstance(value, Predicate):
            wrapped_predicates[key] = value
        elif callable(value):
            wrapped_predicates[key] = Predicate(value)
        else:
            raise TypeError(
                f"Predicate '{key}' must be callable (function, lambda, "
                f"nn.Module) or a Predicate instance, "
                f"got {type(value).__name__}"
            )
    return ConsistencyReport(expression, wrapped_predicates)
