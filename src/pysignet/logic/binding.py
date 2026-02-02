"""Variable binding and expression grounding.

This module provides the Binding class for mapping variables to tensor indices,
and the ground() function for substituting variables with concrete values.
"""

from typing import Dict
import sympy as sp

from pysignet.logic.variable import VariableSymbol
from pysignet.multiclass import PredicateApplication


class Binding:
    """Maps variables to concrete tensor indices for batch evaluation.

    A Binding represents a mapping from FOL variables to concrete integer
    indices in a batch. This enables universal quantification over batch
    dimensions by grounding variables to specific batch positions.

    Args:
        bindings: Optional dictionary mapping VariableSymbols to indices.
                  If not provided, creates an empty binding.

    Examples:
        >>> from pysignet.logic import Variable, Binding
        >>> X, Y = Variable("X Y")
        >>>
        >>> # Create binding
        >>> binding = Binding({X: 0, Y: 1})
        >>>
        >>> # Access bindings
        >>> binding[X]  # Returns 0
        >>> binding[Y]  # Returns 1
        >>>
        >>> # Check membership
        >>> X in binding  # Returns True
        >>>
        >>> # Add new binding
        >>> Z = Variable("Z")
        >>> binding[Z] = 2
    """

    def __init__(
        self, bindings: Dict[VariableSymbol, int] | None = None
    ) -> None:
        """Initialize a binding.

        Args:
            bindings: Optional dictionary mapping variables to indices.
        """
        self._bindings: Dict[VariableSymbol, int] = (
            bindings if bindings is not None else {}
        )

    def __getitem__(self, variable: VariableSymbol) -> int:
        """Get the index bound to a variable.

        Args:
            variable: The variable to look up.

        Returns:
            The integer index bound to this variable.

        Raises:
            KeyError: If variable is not in this binding.
        """
        return self._bindings[variable]

    def __setitem__(self, variable: VariableSymbol, index: int) -> None:
        """Bind a variable to an index.

        Args:
            variable: The variable to bind.
            index: The integer index to bind to.
        """
        self._bindings[variable] = index

    def __contains__(self, variable: VariableSymbol) -> bool:
        """Check if a variable is bound.

        Args:
            variable: The variable to check.

        Returns:
            True if variable is bound, False otherwise.
        """
        return variable in self._bindings

    def __len__(self) -> int:
        """Return the number of bindings.

        Returns:
            Number of variables bound.
        """
        return len(self._bindings)

    def __repr__(self) -> str:
        """Return string representation of binding.

        Returns:
            String showing variable-to-index mappings.
        """
        items = [f"{var}: {idx}" for var, idx in self._bindings.items()]
        joined = ", ".join(items)
        return f"Binding({{{joined}}})"

    def __str__(self) -> str:
        """Return string representation of binding.

        Returns:
            String showing variable-to-index mappings.
        """
        return self.__repr__()


def ground(expr: sp.Basic, binding: Binding) -> sp.Basic:
    """Ground an expression by substituting variables with concrete indices.

    Recursively traverses the expression tree and replaces all variables in
    PredicateApplications with their bound indices. Variables not in the
    binding remain unchanged (partial grounding).

    This enables converting FOL expressions with variables into concrete
    expressions that can be evaluated on specific batch indices.

    Args:
        expr: A SymPy expression (potentially containing PredicateApplications
              with variables).
        binding: Binding mapping variables to concrete indices.

    Returns:
        A new expression with variables replaced by their bound indices.
        The structure of the expression is preserved.

    Examples:
        >>> from pysignet import Symbol
        >>> from pysignet.logic import Variable, Binding, ground
        >>> import sympy as sp
        >>>
        >>> # Single predicate application
        >>> Digit = Symbol("Digit")
        >>> X = Variable("X")
        >>> expr = Digit(X)
        >>> binding = Binding({X: 0})
        >>> grounded = ground(expr, binding)
        >>> # grounded is Digit(0)
        >>>
        >>> # Complex expression
        >>> P, Q = Symbol("P Q")
        >>> X, Y = Variable("X Y")
        >>> expr = sp.And(P(X), Q(Y))
        >>> binding = Binding({X: 0, Y: 1})
        >>> grounded = ground(expr, binding)
        >>> # grounded is And(P(0), Q(1))
        >>>
        >>> # Partial grounding
        >>> binding_partial = Binding({X: 0})
        >>> grounded = ground(expr, binding_partial)
        >>> # grounded is And(P(0), Q(Y)) - Y remains unbound
    """

    def _ground_node(node: sp.Basic) -> sp.Basic:
        """Recursively ground a node in the expression tree."""
        # Base case: PredicateApplication
        if isinstance(node, PredicateApplication):
            # Ground each argument
            grounded_args = tuple(
                (
                    binding[arg]
                    if isinstance(arg, VariableSymbol) and arg in binding
                    else arg
                )
                for arg in node.application_args
            )

            # Create new PredicateApplication with grounded args
            return PredicateApplication(node.predicate_name, grounded_args)

        # Base case: leaf nodes (symbols, constants, etc.)
        if not hasattr(node, "args") or len(node.args) == 0:
            return node

        # Recursive case: logical operators (And, Or, Not, Implies, etc.)
        # Ground all children and reconstruct
        grounded_children = [_ground_node(child) for child in node.args]

        # Reconstruct the node with grounded children
        # SymPy's func attribute gives us the class constructor
        return node.func(*grounded_children)

    return _ground_node(expr)
