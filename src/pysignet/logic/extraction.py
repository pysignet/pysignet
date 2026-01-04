"""Variable extraction from logical expressions.

This module provides utilities to extract free variables from logical
expressions containing PredicateApplications with variables.
"""

from typing import Set
import sympy as sp

from .variable import VariableSymbol
from ..multiclass import PredicateApplication


def extract_variables(expr: sp.Basic) -> Set[VariableSymbol]:
    """Extract all free variables from a logical expression.

    Recursively traverses the expression tree and collects all VariableSymbol
    instances found in PredicateApplication arguments.

    Args:
        expr: A SymPy logical expression (can contain PredicateApplications,
              logical operators like And/Or/Not/Implies/Equivalent, and
              boolean constants).

    Returns:
        Set of unique VariableSymbol instances found in the expression.
        Returns empty set if no variables are found.

    Examples:
        >>> from pysignet import Symbol
        >>> from pysignet.logic import Variable, extract_variables
        >>> import sympy as sp
        >>>
        >>> # Single variable
        >>> Digit = Symbol("Digit")
        >>> X = Variable("X")
        >>> expr = Digit(X)
        >>> variables = extract_variables(expr)
        >>> # variables = {X}
        >>>
        >>> # Multiple variables
        >>> P, Q = Symbol("P Q")
        >>> X, Y = Variable("X Y")
        >>> expr = sp.And(P(X), Q(Y))
        >>> variables = extract_variables(expr)
        >>> # variables = {X, Y}
        >>>
        >>> # Variable appears multiple times (counted once)
        >>> expr = sp.And(P(X), Q(X))
        >>> variables = extract_variables(expr)
        >>> # variables = {X}
        >>>
        >>> # No variables
        >>> expr = sp.And(P, Q)
        >>> variables = extract_variables(expr)
        >>> # variables = set()
    """
    variables: Set[VariableSymbol] = set()

    def _traverse(node: sp.Basic) -> None:
        """Recursively traverse expression tree and collect variables."""
        # Base case: PredicateApplication
        if isinstance(node, PredicateApplication):
            # Extract variables from application arguments
            for arg in node.application_args:
                if isinstance(arg, VariableSymbol):
                    variables.add(arg)
            return

        # Base case: leaf nodes (PredicateSymbol, constants, etc.)
        if not hasattr(node, 'args') or len(node.args) == 0:
            return

        # Recursive case: traverse all children
        for child in node.args:
            _traverse(child)

    _traverse(expr)
    return variables


def extract_variables_from_application(
    app: PredicateApplication
) -> Set[VariableSymbol]:
    """Extract free variables from a single PredicateApplication.

    Args:
        app: A PredicateApplication instance.

    Returns:
        Set of unique VariableSymbol instances in the application's arguments.

    Examples:
        >>> from pysignet import Symbol
        >>> from pysignet.logic import Variable
        >>>
        >>> P = Symbol("P")
        >>> X, Y = Variable("X Y")
        >>>
        >>> # Two variables
        >>> app = P(X, Y, 0)
        >>> vars = extract_variables_from_application(app)
        >>> # vars = {X, Y}
        >>>
        >>> # One variable, multiple occurrences
        >>> app = P(X, 0, X)
        >>> vars = extract_variables_from_application(app)
        >>> # vars = {X}
        >>>
        >>> # No variables
        >>> app = P(0, 1, 2)
        >>> vars = extract_variables_from_application(app)
        >>> # vars = set()
    """
    variables: Set[VariableSymbol] = set()
    for arg in app.application_args:
        if isinstance(arg, VariableSymbol):
            variables.add(arg)
    return variables
