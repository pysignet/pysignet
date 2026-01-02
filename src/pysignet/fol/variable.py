"""Variable class for first-order logic expressions.

Variables are placeholders in logical expressions that can be bound to
concrete tensors during evaluation.
"""

from typing import Tuple, Union
import sympy as sp


class VariableSymbol(sp.Symbol):
    """A variable in a first-order logic expression.

    Variables are SymPy symbols that represent unknown values to be bound
    during evaluation. They integrate seamlessly with SymPy's logic
    infrastructure.

    This class is an implementation detail. Users should use the Variable()
    function to create variables.
    """

    def __new__(cls, name: str):
        """Create a new VariableSymbol.

        Args:
            name: The name of the variable (e.g., "X", "Y", "Z")

        Returns:
            A new VariableSymbol instance
        """
        return sp.Symbol.__new__(cls, name)


def Variable(
    names: str,
) -> Union[VariableSymbol, Tuple[VariableSymbol, ...]]:
    """Create one or more FOL variables.

    Variables are placeholders in logical expressions that must be bound to
    concrete tensors during evaluation. All free variables are automatically
    universally quantified over the batch dimension.

    Args:
        names: Space-separated variable names (e.g., "X Y Z")

    Returns:
        - Single VariableSymbol if one name provided
        - Tuple of VariableSymbols if multiple names provided

    Examples:
        >>> X = Variable("X")
        >>> X, Y, Z = Variable("X Y Z")
        >>> vars = Variable("A B")  # Returns tuple

    Raises:
        ValueError: If names is empty or whitespace-only
    """
    # Strip and split names
    name_list = names.strip().split()

    if not name_list:
        raise ValueError("Variable names string cannot be empty")

    # Create Variable instances
    variables = [VariableSymbol(name) for name in name_list]

    # Return tuple if multiple, single variable if one
    if len(variables) == 1:
        return variables[0]
    else:
        return tuple(variables)
