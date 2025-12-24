"""
Logic Loss: A library for converting predicate logic expressions into differentiable loss functions.

Relaxes logical operators using t-norms to create differentiable loss functions compatible with PyTorch.
"""

from .core import LogicLoss, Predicate
from .tnorms import TNorm, ProductTNorm, LukasiewiczTNorm, GodelTNorm

__version__ = "0.1.0"
__all__ = ["LogicLoss", "Predicate", "TNorm", "ProductTNorm", "LukasiewiczTNorm", "GodelTNorm"]
