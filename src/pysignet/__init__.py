"""Logic Loss: Converts logic expressions to differentiable losses.

Relaxes logical operators using t-norms to create differentiable
loss functions compatible with PyTorch.

New API (recommended):
    from logic_as_loss import compile_logic, Predicate
    logic_loss = compile_logic(expr, predicates)
    loss = logic_loss.loss(x)

Old API (backward compatible):
    from logic_as_loss import LogicCompiler, Predicate
    compiler = LogicCompiler(expr, predicates)
    loss = compiler.loss(x)
"""

# New API (recommended)
from .predicate import Predicate
from .compilation import LogicCompiler as BaseLogicCompiler, TNormCompiler
from .loss import LogicLoss
from .api import compile_logic

# Backward compatibility - old LogicCompiler
from .core import LogicCompiler

# Other exports
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
    # New API (recommended)
    "compile_logic",
    "LogicLoss",
    "TNormCompiler",
    "Predicate",
    # Backward compatible
    "LogicCompiler",
    # Other
    "ConsistencyChecker",
    "TNorm",
    "RProductTNorm",
    "SProductTNorm",
    "LukasiewiczTNorm",
    "GodelTNorm"
]
