"""Logic Loss: Converts logic expressions to differentiable losses.

Relaxes logical operators using t-norms to create differentiable
loss functions compatible with PyTorch.
"""

from .core import LogicCompiler, Predicate
from .consistency import ConsistencyChecker
from .tnorms import (
    TNorm,
    RProductTNorm,
    SProductTNorm,
    LukasiewiczTNorm,
    GodelTNorm
)

__version__ = "0.1.0"
__all__ = [
    "LogicCompiler",
    "Predicate",
    "ConsistencyChecker",
    "TNorm",
    "RProductTNorm",
    "SProductTNorm",
    "LukasiewiczTNorm",
    "GodelTNorm"
]
