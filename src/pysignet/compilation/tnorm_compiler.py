"""T-norm based logic compilation strategy."""

from typing import Callable, Dict, Union, Set, Optional

import sympy as sp
import torch

from .base import LogicCompiler
from ..predicate import Predicate
from ..tnorms import TNorm, RProductTNorm


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
    ) -> Callable[[Union[torch.Tensor, Dict[str, torch.Tensor]]], torch.Tensor]:
        """Compile a logic expression into a differentiable callable.

        Args:
            expr: SymPy logic expression (e.g., sp.And(P, sp.Or(Q, sp.Not(R))))
            predicates: Dict mapping predicate names to Predicate objects.
                       Raw callables (functions, nn.Modules) are automatically
                       wrapped in Predicate objects.

        Returns:
            Callable that takes inputs and returns satisfaction tensor of
            shape (batch_size,) with values in [0, 1].

        Raises:
            ValueError: If symbols in expr have no corresponding predicates
            TypeError: If predicate values are not callable or Predicate
        """
        # Auto-wrap raw callables in Predicate objects
        wrapped_predicates: Dict[str, Predicate] = {}
        for key, value in predicates.items():
            if isinstance(value, Predicate):
                # Already a Predicate, use as-is
                wrapped_predicates[key] = value
            elif callable(value):
                # Raw callable (function, lambda, nn.Module) - auto-wrap
                wrapped_predicates[key] = Predicate(value)
            else:
                # Not callable - raise helpful error
                raise TypeError(
                    f"Predicate '{key}' must be callable (function, lambda, "
                    f"nn.Module) or a Predicate instance, got {type(value).__name__}"
                )

        # Assign names to predicates from dict keys
        # Validate that predicates are not reused with different names
        for key, pred in wrapped_predicates.items():
            if pred.name is not None and pred.name != key:
                raise ValueError(
                    f"Predicate already has name '{pred.name}' but is being "
                    f"registered with different key '{key}'. Each Predicate "
                    f"instance can only be used with one name. Create a new "
                    f"Predicate instance if you need the same function with a "
                    f"different name."
                )
            pred.name = key

        # Verify all symbols have corresponding predicates
        symbols = self._extract_predicate_symbols(expr)
        missing = symbols - set(wrapped_predicates.keys())
        if missing:
            raise ValueError(
                f"Missing predicates for symbols: {missing}"
            )

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
            return self._evaluate_expression(expr, inputs, wrapped_predicates)

        return compiled_logic

    def _extract_predicate_symbols(self, expr: sp.Basic) -> Set[str]:
        """Extract all predicate symbols from a SymPy expression.

        Args:
            expr: SymPy expression to analyze

        Returns:
            Set of symbol names (strings)
        """
        if isinstance(expr, sp.Symbol):
            return {str(expr)}

        symbols: Set[str] = set()
        for arg in expr.args:
            symbols.update(self._extract_predicate_symbols(arg))
        return symbols

    def _evaluate_expression(
        self,
        expr: sp.Basic,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        predicates: Dict[str, Predicate]
    ) -> torch.Tensor:
        """Recursively evaluate SymPy expression using t-norms.

        Args:
            expr: SymPy expression to evaluate
            inputs: Single tensor or dict of tensors
            predicates: Dict of predicates

        Returns:
            Tensor of shape (batch_size,) with values in [0, 1]
        """
        # Base case: predicate symbol (named neuron evaluation)
        if isinstance(expr, sp.Symbol):
            pred_name = str(expr)
            predicate = predicates[pred_name]

            # Pass entire inputs to predicate - let predicate choose what
            # to use. This enables flexible input handling:
            # - Single tensor: lambda x: model(x)
            # - Dict with one key: lambda x: model(x["key"])
            # - Dict with multiple keys:
            #   lambda x: model(cat([x["k1"], x["k2"]]))
            # The predicate function decides how to extract/combine inputs
            return predicate(inputs)

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
            result = self._evaluate_expression(expr.args[0], inputs, predicates)
            for arg in expr.args[1:]:
                result = self.tnorm.conjunction(
                    result,
                    self._evaluate_expression(arg, inputs, predicates)
                )
            return result

        if isinstance(expr, sp.Or):
            # Disjoin all arguments
            result = self._evaluate_expression(expr.args[0], inputs, predicates)
            for arg in expr.args[1:]:
                result = self.tnorm.disjunction(
                    result,
                    self._evaluate_expression(arg, inputs, predicates)
                )
            return result

        if isinstance(expr, sp.Not):
            return self.tnorm.negation(
                self._evaluate_expression(expr.args[0], inputs, predicates)
            )

        if isinstance(expr, sp.Implies):
            return self.tnorm.implication(
                self._evaluate_expression(expr.args[0], inputs, predicates),
                self._evaluate_expression(expr.args[1], inputs, predicates)
            )

        if isinstance(expr, sp.Equivalent):
            return self.tnorm.equivalence(
                self._evaluate_expression(expr.args[0], inputs, predicates),
                self._evaluate_expression(expr.args[1], inputs, predicates)
            )

        raise ValueError(f"Unsupported expression type: {type(expr)}")
