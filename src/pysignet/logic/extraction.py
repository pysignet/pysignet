"""Variable and constant extraction from logical expressions.

This module provides utilities to extract free variables and constants from
logical expressions containing PredicateApplications with variables and
constant arguments.
"""

from typing import Any, Set
import sympy as sp

from pysignet.logic.variable import VariableSymbol
from pysignet.symbols import PredicateApplication


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
        if not hasattr(node, "args") or len(node.args) == 0:
            return

        # Recursive case: traverse all children
        for child in node.args:
            _traverse(child)

    _traverse(expr)
    return variables


def extract_variables_from_application(
    app: PredicateApplication,
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


def is_variable(arg: Any) -> bool:
    """Check if an argument is a variable.

    Args:
        arg: An argument from a PredicateApplication.

    Returns:
        True if arg is a VariableSymbol, False otherwise.

    Examples:
        >>> from pysignet.logic import Variable
        >>> X = Variable("X")
        >>> is_variable(X)
        True
        >>> is_variable(5)
        False
        >>> is_variable("red")
        False
    """
    return isinstance(arg, VariableSymbol)


def is_constant(arg: Any) -> bool:
    """Check if an argument is a constant.

    Constants are any values that are not VariableSymbols, including:
    - Integers, floats, strings
    - None
    - Tuples, lists, etc.

    Args:
        arg: An argument from a PredicateApplication.

    Returns:
        True if arg is a constant (not a VariableSymbol), False otherwise.

    Examples:
        >>> from pysignet.logic import Variable
        >>> is_constant(5)
        True
        >>> is_constant("red")
        True
        >>> is_constant(None)
        True
        >>> X = Variable("X")
        >>> is_constant(X)
        False
    """
    return not isinstance(arg, VariableSymbol)


def extract_constants(expr: sp.Basic) -> Set[Any]:
    """Extract all constants from a logical expression.

    Recursively traverses the expression tree and collects all non-Variable
    arguments found in PredicateApplication arguments. Constants include
    integers, strings, floats, None, and other Python literals.

    Args:
        expr: A SymPy logical expression (can contain PredicateApplications,
              logical operators like And/Or/Not/Implies/Equivalent, and
              boolean constants).

    Returns:
        Set of unique constant values found in the expression.
        Returns empty set if no constants are found.

    Examples:
        >>> from pysignet import Symbol
        >>> from pysignet.logic import Variable
        >>> import sympy as sp
        >>>
        >>> # Single constant
        >>> Digit = Symbol("Digit")
        >>> expr = Digit(5)
        >>> constants = extract_constants(expr)
        >>> # constants = {5}
        >>>
        >>> # Multiple constants
        >>> P, Q = Symbol("P Q")
        >>> expr = sp.And(P(0), Q(1))
        >>> constants = extract_constants(expr)
        >>> # constants = {0, 1}
        >>>
        >>> # Mixed variables and constants
        >>> X = Variable("X")
        >>> expr = sp.And(P(X, 5), Q(X, "red"))
        >>> constants = extract_constants(expr)
        >>> # constants = {5, "red"}
        >>>
        >>> # No constants
        >>> expr = sp.And(P(X), Q(X))
        >>> constants = extract_constants(expr)
        >>> # constants = set()
    """
    constants: Set[Any] = set()

    def _traverse(node: sp.Basic) -> None:
        """Recursively traverse expression tree and collect constants."""
        # Base case: PredicateApplication
        if isinstance(node, PredicateApplication):
            # Extract constants from application arguments
            for arg in node.application_args:
                if is_constant(arg):
                    constants.add(arg)
            return

        # Base case: leaf nodes (PredicateSymbol, constants, etc.)
        if not hasattr(node, "args") or len(node.args) == 0:
            return

        # Recursive case: traverse all children
        for child in node.args:
            _traverse(child)

    _traverse(expr)
    return constants


def extract_constants_from_application(app: PredicateApplication) -> Set[Any]:
    """Extract constants from a single PredicateApplication.

    Args:
        app: A PredicateApplication instance.

    Returns:
        Set of unique constant values in the application's arguments.

    Examples:
        >>> from pysignet import Symbol
        >>> from pysignet.logic import Variable
        >>>
        >>> P = Symbol("P")
        >>> X, Y = Variable("X Y")
        >>>
        >>> # Two constants
        >>> app = P(0, 1, 2)
        >>> consts = extract_constants_from_application(app)
        >>> # consts = {0, 1, 2}
        >>>
        >>> # Mixed variables and constants
        >>> app = P(X, 5, Y, "red")
        >>> consts = extract_constants_from_application(app)
        >>> # consts = {5, "red"}
        >>>
        >>> # No constants
        >>> app = P(X, Y)
        >>> consts = extract_constants_from_application(app)
        >>> # consts = set()
    """
    constants: Set[Any] = set()
    for arg in app.application_args:
        if is_constant(arg):
            constants.add(arg)
    return constants
