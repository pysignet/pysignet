"""First-Order Logic (FOL) support for pysignet.

This module provides first-order logic capabilities including variables,
quantifiers, and binding.
"""

from .binding import Binding, ground
from .expansion import expand_quantifier
from .extraction import (
    extract_constants,
    extract_constants_from_application,
    extract_variables,
    extract_variables_from_application,
    is_constant,
    is_variable,
)
from .quantifier import Exists, ForAll
from .variable import Variable

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
