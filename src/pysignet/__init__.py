"""pysignet: Integrates symbolic logic with neural networks.

Converts SymPy logic expressions to differentiable loss functions using t-norms,
enabling training of neural networks with logical constraints.

Quick Start:
    from pysignet import compile_logic, Predicate
    logic_loss = compile_logic(expr, predicates)
    loss = logic_loss.loss(x)

Advanced Usage:
    from pysignet import TNormCompiler, LogicLoss
    compiler = TNormCompiler(tnorm='rproduct')
    compiled = compiler.compile(expr, predicates)
    logic_loss = LogicLoss(compiled, predicates)
"""

# Core API
from .predicate import Predicate
from .compilation import TNormCompiler
from .loss import LogicLoss
from .api import compile_logic

# Other exports
from .consistency import ConsistencyChecker
from .tnorms import (
    TNorm,
    RProductTNorm,
    SProductTNorm,
    LukasiewiczTNorm,
    GodelTNorm
)

__version__ = "0.2.0"
__all__ = [
    # Core API
    "compile_logic",
    "LogicLoss",
    "TNormCompiler",
    "Predicate",
    # Other
    "ConsistencyChecker",
    "TNorm",
    "RProductTNorm",
    "SProductTNorm",
    "LukasiewiczTNorm",
    "GodelTNorm"
]
