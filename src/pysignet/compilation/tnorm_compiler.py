"""T-norm based logic compilation strategy."""

from collections.abc import Callable

import sympy as sp
import torch

from pysignet.compilation.base import LogicCompiler
from pysignet.compilation.compiled_expression import CompiledExpression
from pysignet.context import EvaluationContext
from pysignet.logic import extract_variables
from pysignet.predicate import Predicate
from pysignet.tnorms import MixedTNorm, TNorm
from pysignet.tnorms.product import SProductTNorm


class TNormCompiler(LogicCompiler):
    """Compiles logic expressions using t-norm relaxations.

    TNormCompiler uses continuous t-norm relaxations (Product, Lukasiewicz,
    Godel, etc.) to convert crisp logical operators into differentiable
    operations over [0,1].

    The compile() method returns a CompiledExpression that evaluates the
    expression when called with inputs.

    Args:
        tnorm: T-norm instance for relaxation (default: MixedTNorm)
        jit: If True, wrap the combinator-dispatch step in torch.compile
            for formulas with at least `JIT_SIZE_THRESHOLD` leaf atoms.
            Default False (opt-in; see TODO.md 2.21). Predicate calls
            are never traced -- only the connective combination of
            already-evaluated leaf tensors is compiled.

    Example:
        ```python
        compiler = TNormCompiler()  # uses MixedTNorm by default
        compiled = compiler.compile(expr, predicates)
        satisfaction = compiled(X=x)  # Returns tensor in [0, 1]
        ```
    """

    def __init__(
        self, tnorm: TNorm | None = None, jit: bool = False
    ) -> None:
        """Initialize TNormCompiler with a t-norm.

        Args:
            tnorm: T-norm for logical operator relaxation. If None, uses
                  MixedTNorm as default (matches compile_logic default).
            jit: Opt-in torch.compile path for large formulas. Default
                False. See class docstring.
        """
        self._tnorm = tnorm or MixedTNorm()
        self.jit = jit

    @property
    def recommended_postprocessing(self) -> str:
        """Delegate to t-norm's recommendation."""
        return self._tnorm.recommended_postprocessing

    @property
    def tnorm(self) -> TNorm:
        """The TNorm that is used for this relaxation"""
        return self._tnorm

    def _is_product_conjunction(self) -> bool:
        """Product t-norms support log-space fusion."""
        return isinstance(self._tnorm, SProductTNorm)

    def conjunction(self, values: torch.Tensor) -> torch.Tensor:
        """Delegate to t-norm conjunction."""
        return self._tnorm.conjunction(values)

    def disjunction(self, values: torch.Tensor) -> torch.Tensor:
        """Delegate to t-norm disjunction."""
        return self._tnorm.disjunction(values)

    def negation(self, a: torch.Tensor) -> torch.Tensor:
        """Delegate to t-norm negation."""
        return self._tnorm.negation(a)

    def implication(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Delegate to t-norm implication."""
        return self._tnorm.implication(a, b)

    def equivalence(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Delegate to t-norm equivalence."""
        return self._tnorm.equivalence(a, b)

    def compile(
        self,
        expr: sp.Basic,
        predicates: dict[str, Predicate | Callable[..., torch.Tensor]],
    ) -> CompiledExpression:
        """Compile a logic expression into a differentiable CompiledExpression.

        Args:
            expr: SymPy logic expression (e.g., sp.And(P, sp.Or(Q, sp.Not(R))))
            predicates: Dict mapping predicate names to Predicate objects.
                       Raw callables (functions, nn.Modules) are automatically
                       wrapped in Predicate objects.

        Returns:
            CompiledExpression that can be evaluated with variable bindings,
            supports partial binding, and provides introspection.

        Raises:
            ValueError: If symbols in expr have no corresponding predicates
            TypeError: If predicate values are not callable or Predicate
        """
        # Use base class method for validation and wrapping
        # This also expands quantifiers internally
        wrapped_predicates = self._wrap_and_validate_predicates(
            expr, predicates
        )

        # Re-expand quantifiers to get expanded expression
        # (base class expands internally but doesn't return it)
        expanded_expr = self._expand_quantifiers(expr)

        # Extract free variables for FOL support (batch quantification)
        free_vars = extract_variables(expanded_expr)

        # Create a closure that evaluates the expression. Uses the
        # opt-in jit=True torch.compile path for large formulas, or the
        # ordinary eager evaluator otherwise (default).
        compiled_logic = self._build_evaluator(
            expanded_expr, wrapped_predicates
        )

        # Create a log-space closure for fused log-activation
        def compiled_logic_log(
            inputs: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            """Evaluate compiled logic in log-space.

            Args:
                inputs: Dict mapping variable names to tensors

            Returns:
                Log-satisfaction tensor of shape (batch_size,)
            """
            ctx = EvaluationContext()
            return self._evaluate_expression_log(
                expanded_expr, inputs, wrapped_predicates, ctx
            )

        # Return CompiledExpression with compiler reference
        return CompiledExpression(
            compiled_logic=compiled_logic,
            compiled_logic_log=compiled_logic_log,
            free_variables=set(v.name for v in free_vars),
            predicates=wrapped_predicates,
            compiler=self,
            expr=expr,
        )
