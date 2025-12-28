"""T-norm and t-conorm implementations for relaxing logic operators.

This module provides various t-norm implementations for continuous
relaxations of logical operators, enabling differentiable logic-based
neural network training.
"""

from .base import TNorm
from .godel import GodelTNorm
from .lukasiewicz import LukasiewiczTNorm
from .product import RProductTNorm, SProductTNorm

__all__ = [
    "TNorm",
    "RProductTNorm",
    "SProductTNorm",
    "LukasiewiczTNorm",
    "GodelTNorm",
]
