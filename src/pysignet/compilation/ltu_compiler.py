"""Linear Threshold Unit (LTU) based logic compilation strategy."""

from typing import Callable, Dict, Union

import sympy as sp
import torch

from .base import LogicCompiler
from ..predicate import Predicate
from ..context import EvaluationContext
from ..multiclass import PredicateApplication


class LinearThresholdUnitCompiler(LogicCompiler):
    """Compiles logic expressions using linear threshold units.

    This compiler represents logical operations as linear threshold units:
    - Conjunction of n literals: sgn(sum(literals) - (n - 0.5))
    - Disjunction of n literals: sgn(sum(literals) - 0.5)
    - Negation: 1 - literal
    - Implication: compiled as (NOT L) OR R
    - Equivalence: compiled as (L => R) AND (R => L)

    Args:
        mode: 'soft' (sigmoid, differentiable) or 'hard' (sign, non-differentiable)
              Default: 'soft'

    Example:
        >>> compiler = LinearThresholdUnitCompiler(mode='soft')
        >>> compiled = compiler.compile(expr, predicates)
        >>> satisfaction = compiled(x)  # Returns tensor in [0, 1]
    """

    def __init__(self, mode: str = 'soft') -> None:
        """Initialize LinearThresholdUnitCompiler.

        Args:
            mode: 'soft' for sigmoid (differentiable) or 'hard' for sign
                  (non-differentiable). Default: 'soft'

        Raises:
            ValueError: If mode is not 'soft' or 'hard'
        """
        if mode not in ('soft', 'hard'):
            raise ValueError(
                f"mode must be 'soft' or 'hard', got '{mode}'"
            )
        self.mode = mode

    def compile(
        self,
        expr: sp.Basic,
        predicates: Dict[str, Predicate]
    ) -> Callable[[Union[torch.Tensor, Dict[str, torch.Tensor]]], torch.Tensor]:
        """Compile a logic expression into a differentiable callable.

        Args:
            expr: SymPy logic expression
            predicates: Dict mapping predicate names to Predicate objects or callables

        Returns:
            Callable that takes inputs and returns satisfaction tensor of
            shape (batch_size,) with values in [0, 1] (soft mode) or {0, 1} (hard mode)

        Raises:
            ValueError: If symbols in expr have no corresponding predicates
            TypeError: If predicate values are not callable
        """
        # Base class handles all validation and preprocessing
        wrapped_predicates = self._wrap_and_validate_predicates(expr, predicates)
        expanded_expr = self._expand_quantifiers(expr)

        # Return a closure that evaluates the expression
        def compiled_logic(
            inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
        ) -> torch.Tensor:
            """Evaluate compiled logic expression.

            Args:
                inputs: Single tensor or dict of tensors

            Returns:
                Satisfaction tensor of shape (batch_size,) in [0, 1]
            """
            ctx = EvaluationContext()
            return self._evaluate_expression(
                expanded_expr, inputs, wrapped_predicates, ctx
            )

        return compiled_logic

    def _evaluate_expression(
        self,
        expr: sp.Basic,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        predicates: Dict[str, Predicate],
        ctx: EvaluationContext
    ) -> torch.Tensor:
        """Recursively evaluate SymPy expression using LTU operations.

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
        if isinstance(expr, sp.Not):
            # Negation: 1 - x
            return 1.0 - self._evaluate_expression(
                expr.args[0], inputs, predicates, ctx
            )

        if isinstance(expr, sp.And):
            # Conjunction: threshold(sum(literals) - (n - 0.5))
            literals = []
            for arg in expr.args:
                literals.append(
                    self._evaluate_expression(arg, inputs, predicates, ctx)
                )

            # Sum all literals
            summed = torch.stack(literals, dim=0).sum(dim=0)

            # Threshold at (n - 0.5)
            n = len(literals)
            threshold = n - 0.5

            if self.mode == 'soft':
                # Soft: sigmoid(k * (sum - threshold))
                # Use k=10 for steepness
                return torch.sigmoid(10.0 * (summed - threshold))
            else:
                # Hard: sign(sum - threshold)
                return ((summed - threshold) >= 0).float()

        if isinstance(expr, sp.Or):
            # Disjunction: threshold(sum(literals) - 0.5)
            literals = []
            for arg in expr.args:
                literals.append(
                    self._evaluate_expression(arg, inputs, predicates, ctx)
                )

            # Sum all literals
            summed = torch.stack(literals, dim=0).sum(dim=0)

            # Threshold at 0.5
            threshold = 0.5

            if self.mode == 'soft':
                # Soft: sigmoid(k * (sum - threshold))
                return torch.sigmoid(10.0 * (summed - threshold))
            else:
                # Hard: sign(sum - threshold)
                return ((summed - threshold) >= 0).float()

        if isinstance(expr, sp.Implies):
            # Implication: (NOT L) OR R
            not_lhs = sp.Not(expr.args[0])
            rhs = expr.args[1]
            return self._evaluate_expression(
                sp.Or(not_lhs, rhs), inputs, predicates, ctx
            )

        if isinstance(expr, sp.Equivalent):
            # Equivalence: (L => R) AND (R => L)
            lhs, rhs = expr.args[0], expr.args[1]
            forward = sp.Implies(lhs, rhs)
            backward = sp.Implies(rhs, lhs)
            return self._evaluate_expression(
                sp.And(forward, backward), inputs, predicates, ctx
            )

        raise ValueError(f"Unsupported expression type: {type(expr)}")
