"""pysignet: Integrates symbolic logic with neural networks.

Converts SymPy logic expressions to differentiable loss functions using t-norms,
enabling training of neural networks with logical constraints.

Quick Start:
    import sympy as sp
    from pysignet import Symbol, Variable, logic_to_loss

    # Create variables and predicate symbols
    X = Variable("X")
    P, Q = Symbol("P Q")

    # Build logic expression with FOL syntax
    expr = sp.Implies(P(X), Q(X))  # If P(X) then Q(X)

    predicates = {
        "P": model_p,  # Any callable or nn.Module
        "Q": model_q,
    }

    # Create differentiable logic loss
    logic_loss = logic_to_loss(expr, predicates)
    loss = logic_loss.loss(X=x_tensor)  # Keyword args for variables

Advanced Usage:
    from pysignet import TNormCompiler, LogicLoss

    compiler = TNormCompiler(tnorm='rproduct')
    compiled = compiler.compile(expr, predicates)
    logic_loss = LogicLoss(compiled)
    loss = logic_loss.loss(X=x_tensor)
"""

# Core API
from .predicate import Predicate
from .compilation import TNormCompiler
from .loss import LogicLoss
from .api import compile_logic, logic_to_loss
from .compilation.compiled_expression import CompiledExpression

# Symbol API for predicates
from .symbols import Symbol, PredicateSymbol, PredicateApplication
from .context import EvaluationContext

# Logic (First-Order Logic)
from .logic import Variable, extract_variables, Binding, ground

# Other exports
from .consistency import ConsistencyChecker
from .tnorms import (
    TNorm,
    RProductTNorm,
    SProductTNorm,
    LukasiewiczTNorm,
    GodelTNorm,
)

__version__ = "0.2.0"
__all__ = [
    # Core API
    "compile_logic",
    "logic_to_loss",
    "CompiledExpression",
    "LogicLoss",
    "TNormCompiler",
    "Predicate",
    # Symbol API
    "Symbol",
    "PredicateSymbol",
    "PredicateApplication",
    "EvaluationContext",
    # Logic
    "Variable",
    "extract_variables",
    "Binding",
    "ground",
    # Other
    "ConsistencyChecker",
    "TNorm",
    "RProductTNorm",
    "SProductTNorm",
    "LukasiewiczTNorm",
    "GodelTNorm",
]
