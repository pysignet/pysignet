"""Quantifier classes for first-order logic.

This module provides ForAll and Exists quantifiers that operate over
explicit finite domains. Each quantifier binds a single variable to
values from the domain.

Multiple variables are handled via nesting quantifiers.
"""

from collections.abc import Iterable
from typing import Any

import sympy as sp
from sympy.logic.boolalg import Boolean

from pysignet.logic.variable import VariableSymbol


class Quantifier(Boolean):  # type: ignore[misc]
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
    def args(self) -> tuple[sp.Basic, sp.Basic]:
        """Return args tuple for SymPy compatibility.

        Returns:
            Tuple of (variable, body) for SymPy tree traversal.
            Note: domain is not included in args to avoid issues with
            non-hashable iterables.
        """
        # Wrap list of variables in SymPy Tuple for tree traversal
        if isinstance(self._variable, list):
            var_arg: sp.Basic = sp.Tuple(*self._variable)
        else:
            var_arg = self._variable
        return (var_arg, self._body)

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
            self_domain = list(self.domain)
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
        # Convert list to tuple for hashing
        if isinstance(self.variable, list):
            var = tuple(self.variable)
        else:
            var = self.variable
        return hash((self.__class__, var, self.body))

    def _domain_repr(self) -> str:
        """Return a compact string representation of the domain.

        Returns:
            Domain string, truncated to first 5 elements if longer.
        """
        domain_list: list[Any] = list(self.domain)
        if len(domain_list) > 5:
            return str(domain_list[:5])[:-1] + ", ...]"
        return str(domain_list)

    def _pretty(self, printer: Any) -> Any:
        """Return pretty form for SymPy pretty printer (used by Jupyter).

        Without this method, Jupyter displays nested quantifiers as the
        class name when rendering parent SymPy expressions.

        Args:
            printer: SymPy pretty printer instance (unused).

        Returns:
            prettyForm with the human-readable quantifier string.
        """
        del printer  # unused
        from sympy.printing.pretty.stringpict import (  # pylint: disable=import-outside-toplevel
            prettyForm,
        )
        return prettyForm(str(self))

    def _latex(self, printer: Any) -> str:
        """Return LaTeX representation for Jupyter notebook rendering.

        Args:
            printer: SymPy LaTeX printer instance (unused).

        Returns:
            LaTeX string for this quantifier.
        """
        del printer  # unused
        name = type(self).__name__
        domain_str = self._domain_repr()
        return (
            f"\\text{{{name}}}({self.variable}, "
            f"\\text{{{domain_str}}}, {sp.latex(self.body)})"
        )


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
