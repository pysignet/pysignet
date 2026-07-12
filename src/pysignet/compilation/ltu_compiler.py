"""Linear Threshold Unit (LTU) based logic compilation strategy."""

import warnings
from collections.abc import Callable

import sympy as sp
import torch

from pysignet.compilation.base import LogicCompiler
from pysignet.compilation.compiled_expression import CompiledExpression
from pysignet.context import EvaluationContext
from pysignet.logic import extract_variables
from pysignet.predicate import Predicate


class LinearThresholdUnitCompiler(LogicCompiler):
    """Compiles logic expressions using linear threshold units.

    This compiler represents logical operations as linear threshold units:
    - Conjunction of n literals: sgn(sum(literals) - (n - 0.5))
    - Disjunction of n literals: sgn(sum(literals) - 0.5)
    - Negation: 1 - literal
    - Implication: compiled as (NOT L) OR R
    - Equivalence: compiled as (L => R) AND (R => L)

    Args:
        mode: 'soft' (sigmoid, differentiable) or 'hard' (sign,
            non-differentiable). Default: 'soft'
        alpha: Multiplier for sigmoid if mode = 'soft'. Default: 1.0
            When alpha is large, the sigmoids become closer to
            thresholds and have larger gradients around zero.

    Example:
        ```python
        compiler = LinearThresholdUnitCompiler(mode='soft')
        compiled = compiler.compile(expr, predicates)
        satisfaction = compiled(X=x)  # Returns tensor in [0, 1]
        ```
    """

    # Configurable limit for multiplier to the sigmoid
    WARN_ALPHA = 10.0

    def __init__(self, mode: str = "soft", alpha: float = 1.0) -> None:
        """Initialize LinearThresholdUnitCompiler.

        Args:
            mode: 'soft' for sigmoid (differentiable) or 'hard' for
                sign (non-differentiable). Default: 'soft'
            alpha: Multiplier for sigmoid if mode = 'soft'.
                Default: 1.0. When alpha is large, the sigmoids
                become closer to thresholds and have larger
                gradients around zero. Ignored when mode = "hard"

        Raises:
            ValueError: If mode is not 'soft' or 'hard'
        """
        if mode not in ("soft", "hard"):
            raise ValueError(f"mode must be 'soft' or 'hard', got '{mode}'")
        if mode == "soft" and alpha > self.WARN_ALPHA:
            warnings.warn(
                f"Parameter alpha = {alpha} is too large and may "
                f"lead to unreliable gradients. Consider using "
                f"smaller alpha.",
                UserWarning,
                stacklevel=2,
            )
        self.mode = mode
        self.alpha = alpha

    @property
    def recommended_postprocessing(self) -> str:
        """LTU recommends linear post-processing."""
        return "linear"

    def conjunction(self, values: torch.Tensor) -> torch.Tensor:
        """LTU conjunction: threshold(sum(values) - (n - 0.5)).

        Args:
            values: Tensor of shape (n, ...) with values in [0, 1].

        Returns:
            Tensor of shape (...) with conjunction applied.
        """
        n = values.shape[0]
        summed = values.sum(dim=0)
        threshold = n - 0.5

        if self.mode == "soft":
            return torch.sigmoid(self.alpha * (summed - threshold))
        else:
            return ((summed - threshold) >= 0).float()

    def disjunction(self, values: torch.Tensor) -> torch.Tensor:
        """LTU disjunction: threshold(sum(values) - 0.5).

        Args:
            values: Tensor of shape (n, ...) with values in [0, 1].

        Returns:
            Tensor of shape (...) with disjunction applied.
        """
        summed = values.sum(dim=0)
        threshold = 0.5

        if self.mode == "soft":
            return torch.sigmoid(self.alpha * (summed - threshold))
        else:
            return ((summed - threshold) >= 0).float()

    def compile(
        self,
        expr: sp.Basic,
        predicates: dict[str, Predicate | Callable[..., torch.Tensor]],
    ) -> CompiledExpression:
        """Compile a logic expression into a CompiledExpression.

        Args:
            expr: SymPy logic expression
            predicates: Dict mapping predicate names to Predicate
                objects or callables

        Returns:
            CompiledExpression that can be evaluated with variable
            bindings, supports partial binding, and provides
            introspection.

        Raises:
            ValueError: If symbols in expr have no corresponding
                predicates
            TypeError: If predicate values are not callable
        """
        # Base class handles all validation and preprocessing
        wrapped_predicates = self._wrap_and_validate_predicates(
            expr, predicates
        )
        expanded_expr = self._expand_quantifiers(expr)

        # Extract free variables for FOL support
        free_vars = extract_variables(expanded_expr)

        # Create a closure that evaluates the expression
        def compiled_logic(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
            """Evaluate compiled logic expression.

            Args:
                inputs: Dict mapping variable names to tensors

            Returns:
                Satisfaction tensor of shape (batch_size,) in [0, 1]
            """
            ctx = EvaluationContext()
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
