"""Convenience API for logic compilation."""

from typing import Dict, Optional, Union, Callable

import sympy as sp
import torch

from .predicate import Predicate
from .compilation import TNormCompiler
from .loss import LogicLoss
from .tnorms import TNorm, RProductTNorm


def compile_logic(
    expr: sp.Basic,
    predicates: Dict[str, Predicate],
    mode: str = 'tnorm',
    tnorm: Optional[TNorm] = None,
    post_processing: Optional[
        Union[str, Callable[[torch.Tensor], torch.Tensor]]
    ] = None
) -> LogicLoss:
    """Compile logic expression into a LogicLoss (one-liner convenience API).

    This is the main entry point for most users. It compiles a SymPy logic
    expression into a LogicLoss object that can compute satisfaction degrees
    and losses.

    Args:
        expr: SymPy logic expression (e.g., sp.And(P, sp.Or(Q, sp.Not(R))))
        predicates: Dict mapping predicate names to Predicate objects
        mode: Compilation mode - 'tnorm' (default), or 'semantic' (future)
        tnorm: T-norm for mode='tnorm' (default: RProductTNorm)
        post_processing: Post-processing mode - 'log', 'linear', callable,
                        or None to use t-norm's recommendation (default)

    Returns:
        LogicLoss instance ready for computing satisfaction and loss

    Raises:
        ValueError: If unknown mode specified

    Example:
        Basic usage:
            >>> P, Q, R = sp.symbols('P Q R')
            >>> expr = sp.And(P, sp.Or(Q, sp.Not(R)))
            >>> predicates = {
            ...     'P': Predicate('P', model_p),
            ...     'Q': Predicate('Q', lambda x: (x > 0).float()),
            ...     'R': Predicate('R', lambda x: torch.sigmoid(x.sum(-1)))
            ... }
            >>> logic_loss = compile_logic(expr, predicates)
            >>>
            >>> # Compute satisfaction
            >>> satisfaction = logic_loss(x)  # Returns [0, 1]
            >>>
            >>> # Compute loss
            >>> loss = logic_loss.loss(x)  # Returns scalar loss
            >>>
            >>> # Get parameters for optimization
            >>> params = logic_loss.get_trainable_parameters()
            >>> optimizer = torch.optim.Adam(params, lr=0.001)

        With custom t-norm:
            >>> from logic_as_loss.tnorms import LukasiewiczTNorm
            >>> logic_loss = compile_logic(
            ...     expr, predicates,
            ...     tnorm=LukasiewiczTNorm(),
            ...     post_processing='linear'
            ... )

        With log post-processing:
            >>> logic_loss = compile_logic(
            ...     expr, predicates,
            ...     post_processing='log'  # Use -log(satisfaction)
            ... )
    """
    # Auto-wrap raw callables in Predicate objects
    wrapped_predicates: Dict[str, Predicate] = {}
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
                f"nn.Module) or a Predicate instance, got {type(value).__name__}"
            )

    if mode == 'tnorm':
        # Create t-norm compiler
        tnorm_instance = tnorm or RProductTNorm()
        compiler = TNormCompiler(tnorm=tnorm_instance)

        # Use t-norm's recommended post-processing if not specified
        if post_processing is None:
            post_processing = tnorm_instance.recommended_postprocessing
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Expected 'tnorm'. "
            f"(Future: 'semantic', 'kenn')"
        )

    # Compile the expression with wrapped predicates
    # Returns CompiledExpression
    compiled = compiler.compile(expr, wrapped_predicates)

    # Wrap CompiledExpression in LogicLoss with t-norm for batch quantification
    return LogicLoss(compiled, post_processing, tnorm=tnorm_instance)
