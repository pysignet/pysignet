"""Quantifier classes for first-order logic.

This module provides ForAll and Exists quantifiers that operate over
explicit finite domains. Each quantifier binds a single variable to
values from the domain.

Multiple variables are handled via nesting quantifiers.
"""

from typing import Any, Iterable, Tuple
import sympy as sp

from pysignet.logic.variable import VariableSymbol


class Quantifier(sp.Basic):  # type: ignore[misc]
    """Base class for quantifiers.

    Quantifiers bind a variable to values from a domain and evaluate
    an expression (body) for each binding.

    Args:
        variable: The variable to bind (single VariableSymbol instance).
        domain: Iterable of values the variable can take.
        body: The expression to evaluate (SymPy expression).
    """

    def __init__(
        self, variable: VariableSymbol, domain: Iterable[Any], body: sp.Basic
    ) -> None:
        """Initialize a new quantifier instance.

        Args:
            variable: The variable to bind.
            domain: Domain values for the variable.
            body: Expression to quantify over.
        """
        self._variable = variable
        self._domain = domain
        self._body = body

    @property
    def variable(self) -> VariableSymbol:
        """The variable bound by this quantifier."""
        return self._variable

    @property
    def domain(self) -> Iterable[Any]:
        """The domain of values for the variable."""
        return self._domain

    @property
    def body(self) -> sp.Basic:
        """The expression body."""
        return self._body

    @property
    def args(self) -> Tuple[VariableSymbol, sp.Basic]:
        """Return args tuple for SymPy compatibility.

        Returns:
            Tuple of (variable, body) for SymPy tree traversal.
            Note: domain is not included in args to avoid issues with
            non-hashable iterables.
        """
        return (self._variable, self._body)

    def __eq__(self, other: object) -> bool:
        """Check equality with another quantifier.

        Args:
            other: Object to compare with.

        Returns:
            True if same type, variable, domain, and body.
        """
        if not isinstance(other, self.__class__):
            return False

        # Compare variable and body
        if self.variable != other.variable or self.body != other.body:
            return False

        # Compare domains
        # Handle different iterable types
        try:
            # For lists, tuples, sets - convert to list for comparison
            if isinstance(self.domain, (list, tuple, set)):
                self_domain = list(self.domain)
            elif isinstance(self.domain, range):
                self_domain = list(self.domain)
            else:
                self_domain = list(self.domain)

            if isinstance(other.domain, (list, tuple, set)):
                other_domain = list(other.domain)
            elif isinstance(other.domain, range):
                other_domain = list(other.domain)
            else:
                other_domain = list(other.domain)

            return self_domain == other_domain
        except (TypeError, ValueError):
            # If conversion fails, try direct comparison
            return self.domain == other.domain

    def __hash__(self) -> int:
        """Hash for use in sets/dicts.

        Returns:
            Hash based on variable and body.
            Note: domain not included due to potential non-hashability.
        """
        return hash((self.__class__, self.variable, self.body))


class ForAll(Quantifier):
    """Universal quantifier: ∀variable ∈ domain. body

    Semantics: The body must hold for all values in the domain.
    Expands to conjunction over domain values.

    Example:
        >>> X = Variable("X")
        >>> P = Symbol("P")
        >>> forall = ForAll(X, [0, 1, 2], P(X))
        >>> # Means: P(0) ∧ P(1) ∧ P(2)

    Args:
        variable: Variable to quantify over.
        domain: Values the variable can take.
        body: Expression that must hold for all domain values.
    """

    def __str__(self) -> str:
        """String representation of ForAll."""
        domain_str = str(list(self.domain)[:5])  # Show first 5 elements
        if hasattr(self.domain, "__len__") and len(list(self.domain)) > 5:
            domain_str = domain_str[:-1] + ", ...]"
        return f"ForAll({self.variable}, {domain_str}, {self.body})"

    def __repr__(self) -> str:
        """Repr for ForAll."""
        return self.__str__()


class Exists(Quantifier):
    """Existential quantifier: ∃variable ∈ domain. body

    Semantics: The body must hold for at least one value in the domain.
    Expands to disjunction over domain values.

    Example:
        >>> X = Variable("X")
        >>> P = Symbol("P")
        >>> exists = Exists(X, [0, 1, 2], P(X))
        >>> # Means: P(0) ∨ P(1) ∨ P(2)

    Args:
        variable: Variable to quantify over.
        domain: Values the variable can take.
        body: Expression that must hold for at least one domain value.
    """

    def __str__(self) -> str:
        """String representation of Exists."""
        domain_str = str(list(self.domain)[:5])  # Show first 5 elements
        if hasattr(self.domain, "__len__") and len(list(self.domain)) > 5:
            domain_str = domain_str[:-1] + ", ...]"
        return f"Exists({self.variable}, {domain_str}, {self.body})"

    def __repr__(self) -> str:
        """Repr for Exists."""
        return self.__str__()
