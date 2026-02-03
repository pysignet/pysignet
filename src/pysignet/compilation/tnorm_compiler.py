"""T-norm based logic compilation strategy."""

from typing import Callable, Dict, Optional

import sympy as sp
import torch

from pysignet.compilation.base import LogicCompiler
from pysignet.compilation.compiled_expression import CompiledExpression
from pysignet.predicate import Predicate
from pysignet.tnorms import TNorm, RProductTNorm
from pysignet.context import EvaluationContext
from pysignet.logic import extract_variables


class TNormCompiler(LogicCompiler):
    """Compiles logic expressions using t-norm relaxations.

    TNormCompiler uses continuous t-norm relaxations (Product, Lukasiewicz,
    Godel, etc.) to convert crisp logical operators into differentiable
    operations over [0,1].

    The compile() method returns a CompiledExpression that evaluates the
    expression when called with inputs.

    Args:
        tnorm: T-norm instance for relaxation (default: RProductTNorm)

    Example:
        >>> compiler = TNormCompiler(tnorm=RProductTNorm())
        >>> compiled = compiler.compile(expr, predicates)
        >>> satisfaction = compiled(X=x)  # Returns tensor in [0, 1]
    """

    def __init__(self, tnorm: Optional[TNorm] = None) -> None:
        """Initialize TNormCompiler with a t-norm.

        Args:
            tnorm: T-norm for logical operator relaxation. If None, uses
                  RProductTNorm as default.
        """
        self._tnorm = tnorm or RProductTNorm()

    @property
    def recommended_postprocessing(self) -> str:
        """Delegate to t-norm's recommendation."""
        return self._tnorm.recommended_postprocessing

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
        predicates: Dict[str, Predicate | Callable[..., torch.Tensor]],
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

        # Create a closure that evaluates the expression
        def compiled_logic(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
            """Evaluate compiled logic expression.

            Args:
                inputs: Dict mapping variable names to tensors

            Returns:
                Satisfaction tensor of shape (batch_size,) in [0, 1]
            """
            # Create evaluation context for this evaluation
            # Context manages caching to avoid redundant forward passes
            ctx = EvaluationContext()

            # Evaluate expression
            return self._evaluate_expression(
                expanded_expr, inputs, wrapped_predicates, ctx
            )

        # Return CompiledExpression with compiler reference
        return CompiledExpression(
            compiled_logic=compiled_logic,
            free_variables=set(v.name for v in free_vars),
            predicates=wrapped_predicates,
            compiler=self,
            expr=expr,
        )
