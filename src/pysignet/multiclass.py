"""Symbol API for predicates.

This module provides the Symbol() function which creates predicates that
can be used as nullary (binary predicates) or unary/n-ary (multi-class
predicates) based on usage.

Usage:
    >>> P, Q, Digit = Symbol("P Q Digit")
    >>> expr = sp.And(P, Digit(0))  # P is nullary, Digit is unary

    >>> # With logic variables
    >>> from pysignet.logic import Variable
    >>> X = Variable("X")
    >>> expr = Digit(X)  # Predicate application with variable

This is the foundation for full first-order logic support.
"""

from typing import Tuple, Any, Union, TYPE_CHECKING

import sympy as sp
from sympy.logic.boolalg import Boolean

if TYPE_CHECKING:
    from .logic.variable import VariableSymbol


def Symbol(names: str) -> Union["PredicateSymbol", Tuple["PredicateSymbol", ...]]:
    """Create predicate symbols that work as nullary or n-ary based on usage.

    Predicates of any arity are declared using SymPy-style syntax.

    Args:
        names: Space-separated predicate names (e.g., "P Q Digit")

    Returns:
        Single PredicateSymbol if one name, or tuple of PredicateSymbols
        if multiple names.

    Example:
        Binary predicates (nullary - arity 0):
            >>> P, Q = Symbol("P Q")
            >>> expr = sp.And(P, Q)  # Used without arguments

        Multi-class predicates (unary - arity 1):
            >>> Digit = Symbol("Digit")
            >>> expr = sp.Or(Digit(0), Digit(1), Digit(2))  # Used with index

        Mixed usage:
            >>> P, Q, Digit = Symbol("P Q Digit")
            >>> expr = sp.And(P, Digit(0))  # P has arity 0, Digit has arity 1

        Invalid (enforced at compile time):
            >>> P, Q, Digit = Symbol("P Q Digit")
            >>> expr = sp.And(P, Digit(0), P(1))  # ERROR: P used with different arities!

    Note:
        The compiler validates that each predicate is used with consistent
        arity throughout the expression.
    """
    name_list = names.split()
    predicates = [PredicateSymbol(name) for name in name_list]
    return tuple(predicates) if len(predicates) > 1 else predicates[0]


class PredicateSymbol(sp.Symbol):  # type: ignore[misc]
    """Symbol that can be used as nullary or called with arguments for n-ary.

    PredicateSymbol inherits from sp.Symbol, so it works seamlessly with
    SymPy's logical operators (And, Or, Not, etc.). It can be used in
    multiple ways depending on arity:

    1. **Nullary (arity 0)**: Used directly as a symbol
       >>> P = Symbol("P")
       >>> expr = sp.And(P, Q)  # P is nullary (arity 0)

    2. **Unary (arity 1)**: Called with one argument
       >>> Digit = Symbol("Digit")
       >>> expr = sp.Or(Digit(0), Digit(1))  # Digit is unary (arity 1)

    3. **N-ary (arity n)**: Called with n arguments (future support)
       >>> Rel = Symbol("Rel")
       >>> expr = Rel(0, 1)  # Rel is binary (arity 2)

    The compiler validates that each predicate is used with consistent arity
    throughout an expression.

    Example:
        >>> # Create predicates
        >>> P, Q, Digit = Symbol("P Q Digit")
        >>>
        >>> # P and Q used as nullary (arity 0)
        >>> # Digit used as unary (arity 1)
        >>> expr = sp.And(P, sp.Or(Q, Digit(0)))
        >>>
        >>> # Map to networks
        >>> predicates = {
        ...     "P": binary_model_p,
        ...     "Q": binary_model_q,
        ...     "Digit": digit_classifier  # Multi-output network
        ... }
        >>>
        >>> compiled = compile_logic(expr, predicates)
        >>> loss = compiled.loss(x)  # Only ONE forward pass for Digit!

    Note:
        This is SymPy's Symbol with added __call__ support. We don't override
        __new__ or __init__ because sp.Symbol is immutable and handles all
        initialization. We only add the __call__ method for creating
        PredicateApplication instances.
    """

    def __call__(
        self, *args: Union[int, "VariableSymbol"]
    ) -> "PredicateApplication":
        """Call with arguments to create a predicate application (n-ary).

        This creates a PredicateApplication AST node representing the
        application of this predicate to arguments (concrete values or variables).

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
        args: Tuple[Union[int, "VariableSymbol"], ...],
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
        return (self.predicate_name == other.predicate_name and
                self.application_args == other.application_args)

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
