"""Core functionality for converting SymPy logic to PyTorch losses."""

from typing import Callable, Dict, Union, Optional, Set, List, Any

import sympy as sp
import torch

from .tnorms import TNorm, RProductTNorm


class Predicate:
    """Wrapper for a predicate (PyTorch model or callable function).

    Args:
        name: Name of the predicate (must match SymPy symbol)
        func: torch.nn.Module or callable returning [0, 1] values
        is_model: Whether the function is trainable (default: auto-detect)
    """

    def __init__(
        self,
        name: str,
        func: Union[torch.nn.Module, Callable[..., Any]],
        is_model: Optional[bool] = None
    ) -> None:
        self.name = name
        self.func = func

        # Auto-detect if it's a model
        if is_model is None:
            self.is_model = isinstance(func, torch.nn.Module)
        else:
            self.is_model = is_model

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Evaluate the predicate, ensuring output is in [0, 1]."""
        result = self.func(*args, **kwargs)

        # Ensure result is a tensor
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(result, dtype=torch.float32)

        # Clamp to [0, 1] for safety
        return torch.clamp(result, 0.0, 1.0)


class LogicCompiler:
    """Converts SymPy logic expressions to differentiable losses.

    Args:
        expression: SymPy logic expression (And(P, Or(Q, Not(R))))
        predicates: Dict mapping predicate names to Predicate objects
        tnorm: T-norm for relaxation (default: RProductTNorm)

    Example:
        >>> import sympy as sp
        >>> import torch.nn as nn
        >>>
        >>> # Define logic expression
        >>> P, Q, R = sp.symbols('P Q R')
        >>> expr = sp.And(P, sp.Or(Q, sp.Not(R)))
        >>>
        >>> # Define predicates
        >>> model_p = nn.Sequential(nn.Linear(10, 5), nn.Sigmoid())
        >>> predicates = {
        >>>     'P': Predicate('P', model_p),
        >>>     'Q': Predicate('Q', lambda x: (x > 0).float()),
        >>>     'R': Predicate('R', lambda x:
        >>>         torch.sigmoid(x.sum(dim=-1)))
        >>> }
        >>>
        >>> # Create logic compiler
        >>> compiler = LogicCompiler(expr, predicates)
        >>>
        >>> # Compute satisfaction on batch
        >>> x = torch.randn(32, 10)  # batch_size=32, features=10
        >>> satisfaction = compiler(x)  # shape (32,)
        >>> loss = compiler.loss(x)  # Compute loss
    """

    def __init__(
        self,
        expression: sp.Basic,
        predicates: Dict[str, Predicate],
        tnorm: Optional[TNorm] = None
    ) -> None:
        self.expression = expression
        self.predicates = predicates
        self.tnorm = tnorm or RProductTNorm()

        # Verify all symbols have corresponding predicates
        symbols = self._extract_predicate_symbols(expression)
        missing = symbols - set(predicates.keys())
        if missing:
            raise ValueError(
                f"Missing predicates for symbols: {missing}"
            )

    def _extract_predicate_symbols(self, expr: sp.Basic) -> Set[str]:
        """Extract all predicate symbols from a SymPy expression."""
        if isinstance(expr, sp.Symbol):
            return {str(expr)}

        symbols: Set[str] = set()
        for arg in expr.args:
            symbols.update(self._extract_predicate_symbols(arg))
        return symbols

    def _evaluate_expression(
        self,
        expr: sp.Basic,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Recursively evaluate SymPy expression using t-norms.

        Args:
            expr: SymPy expression to evaluate
            inputs: Single tensor or dict of tensors

        Returns:
            Tensor of shape (batch_size,) with values in [0, 1]
        """
        # Base case: predicate symbol
        if isinstance(expr, sp.Symbol):
            pred_name = str(expr)
            predicate = self.predicates[pred_name]

            # Get input for this predicate
            if isinstance(inputs, dict):
                pred_input = inputs.get(pred_name, inputs.get('default'))
            else:
                pred_input = inputs

            return predicate(pred_input)

        # Boolean constant
        if expr == sp.true:
            # Return tensor of ones with appropriate batch size
            if isinstance(inputs, dict):
                sample_input = next(iter(inputs.values()))
            else:
                sample_input = inputs
            batch_size = sample_input.shape[0]
            return torch.ones(
                batch_size,
                device=sample_input.device
            )

        if expr == sp.false:
            if isinstance(inputs, dict):
                sample_input = next(iter(inputs.values()))
            else:
                sample_input = inputs
            batch_size = sample_input.shape[0]
            return torch.zeros(
                batch_size,
                device=sample_input.device
            )

        # Logical operators
        if isinstance(expr, sp.And):
            # Conjoin all arguments
            result = self._evaluate_expression(expr.args[0], inputs)
            for arg in expr.args[1:]:
                result = self.tnorm.conjunction(
                    result,
                    self._evaluate_expression(arg, inputs)
                )
            return result

        if isinstance(expr, sp.Or):
            # Disjoin all arguments
            result = self._evaluate_expression(expr.args[0], inputs)
            for arg in expr.args[1:]:
                result = self.tnorm.disjunction(
                    result,
                    self._evaluate_expression(arg, inputs)
                )
            return result

        if isinstance(expr, sp.Not):
            return self.tnorm.negation(
                self._evaluate_expression(expr.args[0], inputs)
            )

        if isinstance(expr, sp.Implies):
            return self.tnorm.implication(
                self._evaluate_expression(expr.args[0], inputs),
                self._evaluate_expression(expr.args[1], inputs)
            )

        if isinstance(expr, sp.Equivalent):
            return self.tnorm.equivalence(
                self._evaluate_expression(expr.args[0], inputs),
                self._evaluate_expression(expr.args[1], inputs)
            )

        raise ValueError(f"Unsupported expression type: {type(expr)}")

    def __call__(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Evaluate the logic expression on a batch of inputs.

        Args:
            inputs: Single tensor (batch_size, ...) for all predicates,
                   or dict mapping predicate names to specific inputs

        Returns:
            Tensor of shape (batch_size,) with satisfaction in [0, 1].
            Higher values = better satisfaction of logical constraint.
        """
        return self._evaluate_expression(self.expression, inputs)

    def loss(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        reduction: str = 'mean',
        post_processing: Optional[
            Union[str, Callable[[torch.Tensor], torch.Tensor]]
        ] = None
    ) -> torch.Tensor:
        """Compute loss based on logical constraint violation.

        Args:
            inputs: Inputs for predicates
            reduction: 'mean', 'sum', or 'none'
            post_processing: Post-processing mode - 'log', 'linear', callable,
                           or None (uses t-norm's recommendation)

        Returns:
            Loss value (lower = better satisfaction)
        """
        satisfaction = self(inputs)

        # Determine which post-processing to apply
        postprocessing_type = (
            post_processing
            if post_processing is not None
            else self.tnorm.recommended_postprocessing
        )

        # Apply post-processing
        if postprocessing_type == 'log':
            # Negative log with numerical stability
            loss_values = -torch.log(satisfaction + 1e-10)
        elif postprocessing_type == 'linear':
            # Linear: 1 - satisfaction
            loss_values = 1.0 - satisfaction
        elif callable(postprocessing_type):
            # User-provided custom post-processing function
            loss_values = postprocessing_type(satisfaction)
        else:
            raise ValueError(
                f"Unknown post-processing: {postprocessing_type}. "
                f"Expected 'log', 'linear', or a callable."
            )

        # Apply reduction
        if reduction == 'mean':
            result: torch.Tensor = loss_values.mean()
            return result
        if reduction == 'sum':
            result = loss_values.sum()
            return result
        if reduction == 'none':
            return loss_values

        raise ValueError(f"Unknown reduction: {reduction}")

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get all trainable parameters from model-based predicates."""
        params: List[torch.nn.Parameter] = []
        for pred in self.predicates.values():
            if pred.is_model and hasattr(pred.func, 'parameters'):
                params.extend(pred.func.parameters())
        return params
