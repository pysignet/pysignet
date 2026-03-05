"""pysignet: Integrates symbolic logic with neural networks.

Converts SymPy logic expressions to differentiable loss functions using t-norms,
enabling training of neural networks with logical constraints.

Quick Start:
    from pysignet import Symbol, Variable, Implies, logic_to_loss

    # Create variables and predicate symbols
    X = Variable("X")
    P, Q = Symbol("P Q")

    # Build logic expression with FOL syntax
    expr = Implies(P(X), Q(X))  # If P(X) then Q(X)

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

Logic Operators (re-exported from SymPy for convenience):
    from pysignet import And, Or, Not, Implies, Equivalent
    from pysignet import ForAll, Exists  # Domain quantifiers

    expr = Implies(And(P(X), Q(X)), R(X))
    expr = ForAll(Y, range(10), Digit(X, Y))

    Power users can import from SymPy directly if needed:
    import sympy as sp
    expr = sp.Implies(P(X), Q(X))  # Equivalent to psn.Implies(P(X), Q(X))
"""

# Core API
from .predicate import Predicate
from .compilation import TNormCompiler
from .loss import LogicLoss
from .api import compile_logic, logic_to_loss, consistency_report
from .compilation.compiled_expression import CompiledExpression

# Symbol API for predicates
from .symbols import Symbol, PredicateSymbol, PredicateApplication
from .context import EvaluationContext

# Logic (First-Order Logic)
from .logic import Variable, extract_variables, Binding, ground
from .logic.quantifier import ForAll, Exists

# Logic operators re-exported from SymPy for convenience.
# These are identical to the SymPy types - no wrapping or copying.
# Power users who need advanced SymPy features can still import sympy directly.
from sympy import And, Or, Not, Implies, Equivalent

# Evaluation
from .eval import ConsistencyChecker, ConsistencyReport
from .tnorms import (
    TNorm,
    RProductTNorm,
    SProductTNorm,
    LukasiewiczTNorm,
    GodelTNorm,
    MixedTNorm,
)

__version__ = "0.2.0"
__all__ = [
    # Core API
    "compile_logic",
    "logic_to_loss",
    "consistency_report",
    "CompiledExpression",
    "LogicLoss",
    "TNormCompiler",
    "Predicate",
    # Symbol API
    "Symbol",
    "PredicateSymbol",
    "PredicateApplication",
    "EvaluationContext",
    # Logic variables and quantifiers
    "Variable",
    "ForAll",
    "Exists",
    "extract_variables",
    "Binding",
    "ground",
    # Logic operators (re-exported from SymPy)
    "And",
    "Or",
    "Not",
    "Implies",
    "Equivalent",
    # Evaluation
    "ConsistencyChecker",
    "ConsistencyReport",
    # T-norms
    "TNorm",
    "RProductTNorm",
    "SProductTNorm",
    "LukasiewiczTNorm",
    "GodelTNorm",
    "MixedTNorm",
]
