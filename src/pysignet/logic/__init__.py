"""First-Order Logic (FOL) support for pysignet.

This module provides first-order logic capabilities including variables,
quantifiers, and binding.
"""

from .variable import Variable
from .extraction import extract_variables
from .binding import Binding, ground

__all__ = ["Variable", "extract_variables", "Binding", "ground"]
