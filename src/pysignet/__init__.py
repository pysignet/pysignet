"""pysignet: Integrates symbolic logic with neural networks.

Converts SymPy logic expressions to differentiable loss functions using t-norms,
enabling training of neural networks with logical constraints.

Quick Start:
    from pysignet import Symbol, compile_logic

    # Create predicates - same syntax for all types
    P, Q, Digit = Symbol("P Q Digit")

    # P, Q are nullary, Digit is unary
    expr = sp.And(P, sp.Or(Q, Digit(0)))

    predicates = {
        "P": binary_model,
        "Q": another_model,
        "Digit": multiclass_classifier
    }

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

# Symbol API for predicates
from .multiclass import Symbol, PredicateSymbol, PredicateApplication
from .context import EvaluationContext

# FOL (First-Order Logic)
from .fol import Variable

# Other exports
from .consistency import ConsistencyChecker
from .tnorms import TNorm, RProductTNorm, SProductTNorm, LukasiewiczTNorm, GodelTNorm

__version__ = "0.2.0"
__all__ = [
    # Core API
    "compile_logic",
    "LogicLoss",
    "TNormCompiler",
    "Predicate",
    # Symbol API
    "Symbol",
    "PredicateSymbol",
    "PredicateApplication",
    "EvaluationContext",
    # FOL
    "Variable",
    # Other
    "ConsistencyChecker",
    "TNorm",
    "RProductTNorm",
    "SProductTNorm",
    "LukasiewiczTNorm",
    "GodelTNorm",
]
