"""T-norm based logic compilation strategy."""

from typing import Callable, Dict, Union, Optional

import sympy as sp
import torch

from pysignet.compilation.base import LogicCompiler
from pysignet.compilation.compiled_expression import CompiledExpression
from pysignet.predicate import Predicate
from pysignet.tnorms import TNorm, RProductTNorm
from pysignet.context import EvaluationContext
from pysignet.multiclass import PredicateApplication
from pysignet.logic import extract_variables


class TNormCompiler(LogicCompiler):
    """Compiles logic expressions using t-norm relaxations.

    TNormCompiler uses continuous t-norm relaxations (Product, Lukasiewicz,
    Godel, etc.) to convert crisp logical operators into differentiable
    operations over [0,1].

    The compile() method returns a closure that evaluates the expression when
    called with inputs.

    Args:
        tnorm: T-norm instance for relaxation (default: RProductTNorm)

    Example:
        >>> compiler = TNormCompiler(tnorm=RProductTNorm())
        >>> compiled = compiler.compile(expr, predicates)
        >>> satisfaction = compiled(x)  # Returns tensor in [0, 1]
    """

    def __init__(self, tnorm: Optional[TNorm] = None) -> None:
        """Initialize TNormCompiler with a t-norm.

        Args:
            tnorm: T-norm for logical operator relaxation. If None, uses
                  RProductTNorm as default.
        """
        self.tnorm = tnorm or RProductTNorm()

    def compile(
        self,
        expr: sp.Basic,
        predicates: Dict[str, Predicate]
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
        wrapped_predicates = self._wrap_and_validate_predicates(expr, predicates)

        # Re-expand quantifiers to get expanded expression
        # (base class expands internally but doesn't return it)
        expanded_expr = self._expand_quantifiers(expr)

        # Extract free variables for FOL support (batch quantification)
        free_vars = extract_variables(expanded_expr)

        # Create a closure that evaluates the expression
        def compiled_logic(
            inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
        ) -> torch.Tensor:
            """Evaluate compiled logic expression.

            Args:
                inputs: Single tensor or dict of tensors

            Returns:
                Satisfaction tensor of shape (batch_size,) in [0, 1]
            """
            # Create evaluation context for this evaluation
            # Context manages caching to avoid redundant forward passes
            ctx = EvaluationContext()

            # Evaluate expression (works for both propositional and FOL)
            return self._evaluate_expression(
                expanded_expr, inputs, wrapped_predicates, ctx
            )

        # Return CompiledExpression (no batch reduction - handled by LogicLoss)
        return CompiledExpression(
            compiled_logic=compiled_logic,
            free_variables=set(v.name for v in free_vars),
            predicates=wrapped_predicates
        )


    def _evaluate_expression(
        self,
        expr: sp.Basic,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        predicates: Dict[str, Predicate],
        ctx: EvaluationContext
    ) -> torch.Tensor:
        """Recursively evaluate SymPy expression using t-norms.

        Args:
            expr: SymPy expression to evaluate
            inputs: Single tensor or dict of tensors
            predicates: Dict of predicates
            ctx: Evaluation context for caching

        Returns:
            Tensor of shape (batch_size,) with values in [0, 1]
        """
        # Base case: PredicateApplication (use base class handler)
        if isinstance(expr, PredicateApplication):
            return self._evaluate_predicate_application(
                expr, inputs, predicates, ctx
            )

        # Base case: predicate symbol (use base class handler)
        if isinstance(expr, sp.Symbol):
            return self._evaluate_symbol(expr, inputs, predicates, ctx)

        # Boolean constants (use base class handler)
        if expr in (sp.true, sp.false):
            return self._evaluate_boolean_constant(expr, inputs)

        # Logical operators
        if isinstance(expr, sp.And):
            # Conjoin all arguments
            result = self._evaluate_expression(expr.args[0], inputs, predicates, ctx)
            for arg in expr.args[1:]:
                result = self.tnorm.conjunction(
                    result,
                    self._evaluate_expression(arg, inputs, predicates, ctx)
                )
            return result

        if isinstance(expr, sp.Or):
            # Disjoin all arguments
            result = self._evaluate_expression(expr.args[0], inputs, predicates, ctx)
            for arg in expr.args[1:]:
                result = self.tnorm.disjunction(
                    result,
                    self._evaluate_expression(arg, inputs, predicates, ctx)
                )
            return result

        if isinstance(expr, sp.Not):
            return self.tnorm.negation(
                self._evaluate_expression(expr.args[0], inputs, predicates, ctx)
            )

        if isinstance(expr, sp.Implies):
            return self.tnorm.implication(
                self._evaluate_expression(expr.args[0], inputs, predicates, ctx),
                self._evaluate_expression(expr.args[1], inputs, predicates, ctx)
            )

        if isinstance(expr, sp.Equivalent):
            return self.tnorm.equivalence(
                self._evaluate_expression(expr.args[0], inputs, predicates, ctx),
                self._evaluate_expression(expr.args[1], inputs, predicates, ctx)
            )

        raise ValueError(f"Unsupported expression type: {type(expr)}")
