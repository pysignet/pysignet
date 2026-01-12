"""Arity validation for logic compilation.

This module provides clean, single-purpose arity validation that checks
callable signatures match predicate usage in expressions.

Key principles:
1. Nullary predicates (bare symbols) are disallowed
2. PredicateApplication arity must match callable signature
3. All predicates validated recursively through expression tree
4. Clear error messages with usage examples
"""

import inspect
from typing import Dict

import sympy as sp
import torch.nn as nn

from ..predicate import Predicate
from ..multiclass import PredicateApplication


def validate_predicate_arity(
    expr: sp.Basic,
    predicates: Dict[str, Predicate]
) -> None:
    """Validate that callable signatures match predicate usage.

    Args:
        expr: SymPy expression to validate
        predicates: Dict mapping predicate names to Predicate objects

    Raises:
        ValueError: If any predicate has incorrect arity

    Example:
        >>> X, Y = Variable("X Y")
        >>> P = Symbol("P")
        >>> expr = P(X, Y)
        >>> predicates = {"P": Predicate(lambda x: ...)}  # Wrong!
        >>> validate_predicate_arity(expr, predicates)
        ValueError: Predicate 'P' arity mismatch...
    """
    _validate_arity_recursive(expr, predicates)


def _validate_arity_recursive(
    node: sp.Basic,
    predicates: Dict[str, Predicate]
) -> None:
    """Recursively validate arity for all predicates in expression tree.

    Args:
        node: Current node in expression tree
        predicates: Dict of predicates to validate

    Raises:
        ValueError: If arity mismatch found
    """
    from ..logic.variable import VariableSymbol

    if isinstance(node, PredicateApplication):
        # Validate PredicateApplication: P(X, Y, 0, ...)
        _validate_application_arity(node, predicates)

    elif isinstance(node, sp.Symbol):
        # Check for disallowed nullary usage (bare symbol)
        if (not isinstance(node, VariableSymbol) and
            node not in (sp.true, sp.false) and
            str(node) in predicates):
            # This is a predicate used without arguments - disallowed
            pred_name = str(node)
            raise ValueError(
                f"Predicate '{pred_name}' used without arguments. "
                f"Nullary predicates are not allowed. "
                f"Use '{pred_name}(X)' with an explicit variable instead."
            )

    # Recurse into subexpressions
    for arg in getattr(node, 'args', []):
        _validate_arity_recursive(arg, predicates)


def _validate_application_arity(
    app: PredicateApplication,
    predicates: Dict[str, Predicate]
) -> None:
    """Validate arity for a single PredicateApplication.

    Args:
        app: PredicateApplication to validate
        predicates: Dict of predicates

    Raises:
        ValueError: If arity mismatch found

    Note:
        nn.Module instances are skipped here because they're validated
        separately in _wrap_and_validate_predicates using infer_module_arity().
        They're also handled specially in _evaluate_predicate_application
        to preserve caching behavior for multi-output modules.
    """
    pred_name = app.predicate_name

    # Skip if predicate not in dict (will be caught by symbol extraction)
    if pred_name not in predicates:
        return

    predicate = predicates[pred_name]
    func = predicate.func

    # Skip validation for nn.Module instances - they're validated separately
    # and handled specially in _evaluate_predicate_application for caching
    if isinstance(func, nn.Module):
        return

    # Expected arity: total number of arguments in application
    expected_arity = len(app.application_args)

    # Get actual arity from callable signature
    actual_arity = _get_callable_arity(func)

    if actual_arity != expected_arity:
        raise ValueError(
            f"Predicate '{pred_name}' arity mismatch: "
            f"application {app} has {expected_arity} argument(s) "
            f"but callable accepts {actual_arity} argument(s). "
            f"Callable signature must match total number of arguments."
        )


def _get_callable_arity(func) -> int:
    """Get the arity (number of parameters) of a callable.

    Args:
        func: Callable to inspect

    Returns:
        Number of positional parameters

    Note:
        For bound methods, Python's inspect.signature() automatically
        excludes 'self', so no special handling is needed.

    Raises:
        TypeError: If cannot inspect signature
    """
    try:
        sig = inspect.signature(func)

        # Count positional parameters
        # Note: bound methods automatically have 'self' excluded
        params = [
            p for p in sig.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
        ]

        return len(params)

    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Cannot inspect signature of {func}. "
            f"Ensure callable has inspectable signature."
        ) from e
