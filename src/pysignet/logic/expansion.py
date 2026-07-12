"""Quantifier expansion over domains.

This module provides functionality to expand ForAll and Exists quantifiers
over their specified domains into conjunctions and disjunctions.
"""

from typing import Any

import sympy as sp

from pysignet.logic.quantifier import Exists, ForAll, Quantifier
from pysignet.logic.variable import VariableSymbol
from pysignet.symbols import PredicateApplication


def expand_quantifier(quantifier: Quantifier) -> sp.Basic:
    """Expand a quantifier over its domain.

    ForAll quantifiers expand to conjunctions (AND) over domain values.
    Exists quantifiers expand to disjunctions (OR) over domain values.

    For each value in the domain, the variable is substituted with that
    value in the body expression. Nested quantifiers are recursively expanded.

    Args:
        quantifier: ForAll or Exists quantifier to expand.

    Returns:
        SymPy expression with quantifier expanded:
        - ForAll(Y, [0,1,2], P(Y)) → P(0) ∧ P(1) ∧ P(2)
        - Exists(Y, [0,1,2], P(Y)) → P(0) ∨ P(1) ∨ P(2)
        - ForAll(Y, [], P(Y)) → True (vacuously true)
        - Exists(Y, [], P(Y)) → False (no value satisfies)

    Examples:
        >>> Y = Variable("Y")
        >>> P = Symbol("P")
        >>> forall = ForAll(Y, [0, 1, 2], P(Y))
        >>> expand_quantifier(forall)
        And(P(0), P(1), P(2))

        >>> exists = Exists(Y, [0, 1], P(Y))
        >>> expand_quantifier(exists)
        Or(P(0), P(1))

    Raises:
        TypeError: If argument is not a Quantifier instance.
    """
    if not isinstance(quantifier, Quantifier):
        raise TypeError(
            f"expand_quantifier expects a Quantifier, got {type(quantifier)}"
        )

    variable = quantifier.variable
    domain = quantifier.domain
    body = quantifier.body

    # Convert domain to list for iteration
    domain_values = list(domain)

    # Handle empty domain
    if not domain_values:
        if isinstance(quantifier, ForAll):
            return sp.true  # Vacuously true
        else:  # Exists
            return sp.false  # No value satisfies

    # Expand body for each domain value
    expanded_bodies = []
    for value in domain_values:
        # Substitute the variable(s) with the domain value(s)
        # Handle multi-variable quantifiers: ForAll([I, J], [(0,1), (0,2)], ...)
        if isinstance(variable, list):
            # Multi-variable: value is a tuple matching variable list
            substituted_body = body
            for var, val in zip(variable, value, strict=True):
                substituted_body = _substitute_variable(
                    substituted_body, var, val
                )
        else:
            # Single variable
            substituted_body = _substitute_variable(body, variable, value)

        # Recursively expand any nested quantifiers
        substituted_body = _expand_nested_quantifiers(substituted_body)

        expanded_bodies.append(substituted_body)

    # Single element: return as-is (no And/Or wrapper)
    if len(expanded_bodies) == 1:
        return expanded_bodies[0]

    # Multiple elements: combine with And (ForAll) or Or (Exists)
    if isinstance(quantifier, ForAll):
        return sp.And(*expanded_bodies)
    else:  # Exists
        return sp.Or(*expanded_bodies)


def _substitute_in_predicate_application(
    expr: PredicateApplication, variable: VariableSymbol, value: Any
) -> PredicateApplication:
    """Substitute variable in a PredicateApplication."""
    new_args = []
    for arg in expr.application_args:
        if arg == variable:
            new_args.append(value)
        elif isinstance(arg, sp.Basic) and variable in arg.free_symbols:
            # Substitute within SymPy arithmetic expressions (e.g. S - I).
            # Convert concrete SymPy integers to Python ints so that
            # predicate callables receive indexable integers.
            substituted = arg.subs(variable, value)
            if isinstance(substituted, sp.Integer):
                substituted = int(substituted)
            new_args.append(substituted)
        else:
            new_args.append(arg)
    return PredicateApplication(expr.predicate_name, tuple(new_args))


def _substitute_in_quantifier(
    expr: Quantifier, variable: VariableSymbol, value: Any
) -> Quantifier:
    """Substitute variable in a Quantifier expression.

    Does not substitute if the quantifier binds the same variable.
    """
    # If this quantifier binds the variable, don't substitute inside
    # Handle both single variable and list of variables
    bound_vars: list[VariableSymbol]
    if isinstance(expr.variable, list):
        bound_vars = expr.variable
    else:
        bound_vars = [expr.variable]
    if variable in bound_vars:
        return expr

    # Substitute in the body
    new_body = _substitute_variable(expr.body, variable, value)
    if isinstance(expr, ForAll):
        return ForAll(expr.variable, expr.domain, new_body)
    return Exists(expr.variable, expr.domain, new_body)


def _substitute_variable(
    expr: sp.Basic, variable: VariableSymbol, value: Any
) -> sp.Basic:
    """Substitute a variable with a value in an expression.

    Handles both standard SymPy expressions and PredicateApplications.

    Args:
        expr: Expression to perform substitution in.
        variable: Variable to substitute.
        value: Value to substitute with.

    Returns:
        Expression with variable replaced by value.
    """
    # Handle PredicateApplication specially
    if isinstance(expr, PredicateApplication):
        return _substitute_in_predicate_application(expr, variable, value)

    # Handle quantifiers (don't substitute the bound variable)
    if isinstance(expr, Quantifier):
        return _substitute_in_quantifier(expr, variable, value)

    # For other SymPy expressions, recursively substitute in args
    if hasattr(expr, "args") and expr.args:
        substituted_args: list[sp.Basic] = [
            _substitute_variable(arg, variable, value) for arg in expr.args
        ]
        return expr.func(*substituted_args)

    # Leaf node: if it's the variable, replace it; otherwise return as-is
    if expr == variable:
        return value
    return expr


def _expand_nested_quantifiers(expr: sp.Basic) -> sp.Basic:
    """Recursively expand any nested quantifiers in expression.

    Args:
        expr: Expression that may contain nested quantifiers.

    Returns:
        Expression with all quantifiers expanded.
    """
    # Base case: if expr is a quantifier, expand it
    if isinstance(expr, Quantifier):
        return expand_quantifier(expr)

    # Handle PredicateApplication (no expansion needed, just recurse on args)
    if isinstance(expr, PredicateApplication):
        return expr

    # Recursive case: if expr has args, recursively expand them
    if hasattr(expr, "args") and expr.args:
        expanded_args = [_expand_nested_quantifiers(arg) for arg in expr.args]
        # Reconstruct the expression with expanded arguments
        return expr.func(*expanded_args)

    # Leaf node: return as-is
    return expr
