"""First-Order Logic (FOL) support for pysignet.

This module provides first-order logic capabilities including variables,
quantifiers, and binding.
"""

from .variable import Variable
from .extraction import (
    extract_variables,
    extract_variables_from_application,
    extract_constants,
    extract_constants_from_application,
    is_variable,
    is_constant,
)
from .binding import Binding, ground
from .quantifier import ForAll, Exists
from .expansion import expand_quantifier

__all__ = [
    "Variable",
    "extract_variables",
    "extract_variables_from_application",
    "extract_constants",
    "extract_constants_from_application",
    "is_variable",
    "is_constant",
    "Binding",
    "ground",
    "ForAll",
    "Exists",
    "expand_quantifier",
]
