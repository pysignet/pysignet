"""Symbol API for predicates in first-order logic expressions.

This module provides the Symbol() function which creates predicate symbols
for use with Variables in FOL expressions. All predicates must be called
with at least one Variable argument.

Usage:
    >>> from pysignet import Symbol, Variable
    >>> import sympy as sp
    >>>
    >>> # Create variables and predicates
    >>> X = Variable("X")
    >>> P, Q, Digit = Symbol("P Q Digit")
    >>>
    >>> # Basic predicates with variable
    >>> expr = sp.And(P(X), Q(X))
    >>>
    >>> # Multi-class predicates with variable and constant
    >>> expr = sp.Or(Digit(X, 0), Digit(X, 1))
"""

from __future__ import annotations

from typing import Any, Tuple, TYPE_CHECKING

import sympy as sp
from sympy.logic.boolalg import Boolean

if TYPE_CHECKING:
    from .logic.variable import VariableSymbol


def Symbol(names: str) -> "PredicateSymbol" | Tuple["PredicateSymbol", ...]:
    """Create predicate symbols for use in FOL expressions.

    All predicates must be called with at least one Variable argument.

    Args:
        names: Space-separated predicate names (e.g., "P Q Digit")

    Returns:
        Single PredicateSymbol if one name, or tuple of PredicateSymbols
        if multiple names.

    Example:
        Basic predicates (with variable):
            >>> X = Variable("X")
            >>> P, Q = Symbol("P Q")
            >>> expr = sp.And(P(X), Q(X))  # Applied to variable X

        Multi-class predicates (with variable and constant):
            >>> X = Variable("X")
            >>> Digit = Symbol("Digit")
            >>> # X is input, 0/1 are class indices
            >>> expr = sp.Or(Digit(X, 0), Digit(X, 1))

        Multi-variable predicates:
            >>> X, Y = Variable("X Y")
            >>> Similar = Symbol("Similar")
            >>> expr = Similar(X, Y)  # Applied to two variables

    Note:
        The compiler validates that each predicate has at least one Variable
        and is used with consistent arity throughout the expression.
    """
    name_list = names.split()
    predicates = [PredicateSymbol(name) for name in name_list]
    return tuple(predicates) if len(predicates) > 1 else predicates[0]


class PredicateSymbol(sp.Symbol):  # type: ignore[misc]
    """Symbol that represents a predicate in FOL expressions.

    PredicateSymbol inherits from sp.Symbol, so it works seamlessly with
    SymPy's logical operators (And, Or, Not, etc.). All predicates MUST
    be called with at least one Variable argument:

    1. **Unary predicate**: One variable argument
       >>> X = Variable("X")
       >>> P = Symbol("P")
       >>> expr = P(X)  # P applied to variable X

    2. **With constants**: Variable plus constant arguments
       >>> X = Variable("X")
       >>> Digit = Symbol("Digit")
       >>> expr = Digit(X, 0)  # Digit applied to X, selecting class 0

    3. **Multi-variable**: Multiple variable arguments
       >>> X, Y = Variable("X Y")
       >>> Similar = Symbol("Similar")
       >>> expr = Similar(X, Y)  # Similar applied to X and Y

    The compiler validates that each predicate has at least one variable
    and is used with consistent arity throughout an expression.

    Example:
        >>> from pysignet import Symbol, Variable, logic_to_loss
        >>> import sympy as sp
        >>>
        >>> # Create variables and predicates
        >>> X = Variable("X")
        >>> P, Q, Digit = Symbol("P Q Digit")
        >>>
        >>> # Build FOL expression
        >>> expr = sp.And(P(X), sp.Or(Q(X), Digit(X, 0)))
        >>>
        >>> # Map to networks
        >>> predicates = {
        ...     "P": binary_model_p,
        ...     "Q": binary_model_q,
        ...     "Digit": digit_classifier  # Multi-output network
        ... }
        >>>
        >>> logic_loss = logic_to_loss(expr, predicates)
        >>> loss = logic_loss.loss(X=x_tensor)  # Keyword arg for variable

    Note:
        This is SymPy's Symbol with added __call__ support. We don't override
        __new__ or __init__ because sp.Symbol is immutable and handles all
        initialization. We only add the __call__ method for creating
        PredicateApplication instances.
    """

    def __call__(self, *args: int | "VariableSymbol") -> "PredicateApplication":
        """Call with arguments to create a predicate application (n-ary).

        Create a PredicateApplication AST node representing this predicate
        applied to arguments (concrete values or variables).

        Args:
            *args: Arguments can be:
                  - Concrete values (int): e.g., Digit(0), Digit(1)
                  - Logic variables: e.g., Digit(X) where X = Variable("X")
                  - Mixed: e.g., P(X, 5, Y)
                  The number of arguments determines the arity.

        Returns:
            PredicateApplication AST node that can be used in logical
            expressions with SymPy operators.

        Example:
            >>> # Concrete arguments
            >>> Digit = Symbol("Digit")
            >>> app = Digit(0)  # Unary application with constant
            >>>
            >>> # Variable arguments
            >>> from pysignet.logic import Variable
            >>> X = Variable("X")
            >>> app = Digit(X)  # Unary application with variable
            >>>
            >>> # Mixed arguments
            >>> app = Digit(X, 5)  # Binary with variable and constant
            >>>
            >>> # N-ary predicates
            >>> Rel = Symbol("Rel")
            >>> X, Y = Variable("X Y")
            >>> expr = Rel(X, Y)  # Binary application (arity 2)

        Note:
            The compiler validates that each predicate is used with consistent
            arity throughout the expression.
        """
        return PredicateApplication(str(self), args)


class PredicateApplication(Boolean):  # type: ignore[misc]
    """AST node representing application of predicate to arguments.

    PredicateApplication represents the application of a PredicateSymbol
    to arguments (concrete values or logic variables). For example, Digit(0)
    creates a PredicateApplication with predicate_name="Digit" and args=(0,),
    while Digit(X) creates one with args=(X,) where X is a Variable.

    The arity is determined by the number of arguments provided.

    This class inherits from SymPy's Boolean to integrate seamlessly with
    SymPy's logical operators (And, Or, Not, Implies, Equivalent).

    Example:
        >>> # Concrete arguments
        >>> Digit = Symbol("Digit")
        >>> app0 = Digit(0)  # Unary (arity 1) with constant
        >>> app1 = Digit(1)  # Unary (arity 1) with constant
        >>>
        >>> # Logic variables
        >>> from pysignet.logic import Variable
        >>> X = Variable("X")
        >>> app_var = Digit(X)  # Unary (arity 1) with variable
        >>>
        >>> # Mixed arguments
        >>> app_mixed = Digit(X, 5)  # Binary (arity 2) mixed
        >>>
        >>> # Use with SymPy operators
        >>> expr = sp.And(app0, app_var)
        >>> expr = sp.Or(app0, sp.Not(app1))
        >>> expr = sp.Implies(app_var, app1)
        >>>
        >>> # N-ary predicates
        >>> Rel = Symbol("Rel")
        >>> X, Y = Variable("X Y")
        >>> app = Rel(X, Y)  # Binary (arity 2)

    Args:
        predicate_name: Name of the predicate being applied.
        args: Tuple of argument values (constants or variables).

    Attributes:
        predicate_name: Name of the predicate.
        application_args: Tuple of argument values (arity determined by length).
    """

    def __init__(
        self,
        predicate_name: str,
        args: Tuple[int | "VariableSymbol", ...],
    ) -> None:
        """Initialize a new PredicateApplication instance.

        Args:
            predicate_name: Name of the predicate being applied.
            args: Tuple of concrete argument values.
        """
        self.predicate_name = predicate_name
        self.application_args = args

    @property
    def args(self) -> Tuple[()]:
        """Return args for SymPy compatibility.

        SymPy expects Basic objects to have an args property.
        We return an empty tuple since our arguments are in
        application_args (not SymPy sub-expressions).
        """
        return ()

    def __eq__(self, other: Any) -> bool:
        """Check equality of predicate applications.

        Two PredicateApplications are equal if they have the same
        predicate name and arguments.
        """
        if not isinstance(other, PredicateApplication):
            return False
        return (
            self.predicate_name == other.predicate_name
            and self.application_args == other.application_args
        )

    def __hash__(self) -> int:
        """Make PredicateApplication hashable for use in sets/dicts."""
        return hash((self.predicate_name, self.application_args))

    def __repr__(self) -> str:
        """Return string representation of predicate application.

        Returns:
            String in format "PredicateName(args)".

        Example:
            >>> Digit = Symbol("Digit")
            >>> app = Digit(0)
            >>> repr(app)
            'Digit(0)'
        """
        args_str = ", ".join(str(arg) for arg in self.application_args)
        return f"{self.predicate_name}({args_str})"

    def __str__(self) -> str:
        """Return string representation."""
        return self.__repr__()

    def _sympystr(self, printer: Any) -> str:
        """Return SymPy string representation for printing.

        This method is called by SymPy's printing system when converting
        expressions to strings.

        Args:
            printer: SymPy printer instance (unused).

        Returns:
            String in format "PredicateName(args)".
        """
        del printer  # unused
        args_str = ", ".join(str(arg) for arg in self.application_args)
        return f"{self.predicate_name}({args_str})"

    def _sympyrepr(self, printer: Any) -> str:
        """Return SymPy repr representation.

        Args:
            printer: SymPy printer instance (unused).

        Returns:
            String representation for SymPy repr printing.
        """
        return self._sympystr(printer)
