"""Consistency checking for neural models using hard boolean logic.

This module provides functionality to measure how often logical constraints
are satisfied by model predictions using discrete (boolean) decisions.
"""

from typing import Dict, Union, Set, Callable, Any

import sympy as sp
import torch

from pysignet.multiclass import PredicateApplication
from pysignet.logic.variable import VariableSymbol


class ConsistencyChecker:
    """Check how often a logical formula is satisfied by model predictions.

    This class evaluates logical formulas using boolean decisions from
    predicates. Each predicate should return a boolean tensor indicating
    whether a condition holds.

    Uses the FOL interface with Variables and PredicateApplications.

    Args:
        expression: SymPy logic expression to check (must use FOL interface)
        predicates: Dict mapping predicate names to callables that return
                   boolean tensors. For example:
                   `{'P': lambda x: model(x).argmax(-1) == 0}`

    Example:
        >>> import sympy as sp
        >>> import torch
        >>> import torch.nn as nn
        >>> from pysignet import ConsistencyChecker, Symbol, Variable
        >>>
        >>> # Define constraint with FOL interface
        >>> P, Q = Symbol("P Q")
        >>> X = Variable("X")
        >>> constraint = sp.Implies(P(X), Q(X))  # If P(X) then Q(X)
        >>>
        >>> # Create models
        >>> model1 = nn.Sequential(nn.Linear(10, 3), nn.Softmax(dim=-1))
        >>> model2 = nn.Sequential(nn.Linear(10, 2), nn.Softmax(dim=-1))
        >>>
        >>> # Predicates return booleans
        >>> predicates = {
        ...     'P': lambda x: model1(x).argmax(dim=-1) == 0,
        ...     'Q': lambda x: model2(x).argmax(dim=-1) == 1,
        ... }
        >>>
        >>> # Create checker
        >>> checker = ConsistencyChecker(constraint, predicates)
        >>>
        >>> # Check on batch
        >>> x = torch.randn(100, 10)
        >>> satisfied = checker(x)  # Boolean tensor, shape (100,)
        >>> consistency_count = satisfied.sum().item()
        >>> print(f"Satisfied: {consistency_count}/100")
    """

    def __init__(
        self,
        expression: sp.Basic,
        predicates: Dict[str, Callable[[Any], torch.Tensor]]
    ) -> None:
        self.expression = expression
        self.predicates = predicates

        # Verify all symbols have corresponding predicates
        symbols = self._extract_predicate_symbols(expression)
        missing = symbols - set(predicates.keys())
        if missing:
            raise ValueError(
                f"Missing predicates for symbols: {missing}"
            )

    def _extract_predicate_symbols(self, expr: sp.Basic) -> Set[str]:
        """Extract all predicate symbols from a SymPy expression.

        Only supports FOL interface (PredicateApplication).
        Variables (VariableSymbol) are skipped.
        """
        # FOL interface: predicate applications P(X), Digit(X, 0)
        if isinstance(expr, PredicateApplication):
            return {expr.predicate_name}

        # Skip variables (they are not predicates)
        if isinstance(expr, VariableSymbol):
            return set()

        # Recurse into compound expressions
        symbols: Set[str] = set()
        for arg in expr.args:
            symbols.update(self._extract_predicate_symbols(arg))
        return symbols

    def _extract_predicate_applications(self, expr: sp.Basic) -> Set[PredicateApplication]:
        """Extract all unique PredicateApplications from expression.

        Args:
            expr: SymPy expression

        Returns:
            Set of all PredicateApplication instances in the expression
        """
        applications: Set[PredicateApplication] = set()

        if isinstance(expr, PredicateApplication):
            applications.add(expr)
            return applications

        # Skip variables
        if isinstance(expr, VariableSymbol):
            return applications

        # Recurse into compound expressions
        for arg in expr.args:
            applications.update(self._extract_predicate_applications(arg))

        return applications

    def _evaluate_predicates(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[Any, torch.Tensor]:
        """Evaluate all predicate applications on inputs to get boolean decisions.

        Args:
            inputs: Single tensor or dict of tensors for predicates

        Returns:
            Dict mapping PredicateApplication instances to boolean tensors
        """
        from pysignet.logic import is_constant

        decisions = {}

        # Extract all unique predicate applications from expression
        applications = self._extract_predicate_applications(self.expression)

        for app in applications:
            pred_name = app.predicate_name
            predicate = self.predicates[pred_name]

            # Extract constants from this application (preserving order)
            constants = [arg for arg in app.application_args if is_constant(arg)]

            # Get input for this predicate
            if isinstance(inputs, dict):
                pred_input = inputs.get(pred_name, inputs.get('default'))
            else:
                pred_input = inputs

            # Evaluate predicate with constants (if any)
            if constants:
                # Pass constants as arguments to predicate
                result = predicate(*constants)
            else:
                # No constants, pass just the input
                result = predicate(pred_input)

            # Convert to boolean if needed
            if result.dtype != torch.bool:
                result = result.bool()

            # Store result keyed by the PredicateApplication instance
            decisions[app] = result

        return decisions

    def _evaluate_boolean(
        self,
        expr: sp.Basic,
        decisions: Dict[Any, torch.Tensor]
    ) -> torch.Tensor:
        """Recursively evaluate SymPy expression with boolean logic.

        Args:
            expr: SymPy expression to evaluate
            decisions: Dict mapping PredicateApplication instances to boolean tensors

        Returns:
            Boolean tensor indicating satisfaction
        """
        # Base case: FOL predicate application P(X), Digit(X, 0), E("P", "H")
        if isinstance(expr, PredicateApplication):
            # Look up by the PredicateApplication instance itself
            return decisions[expr]

        # Boolean constants
        if expr == sp.true:
            sample = next(iter(decisions.values()))
            return torch.ones(
                sample.shape[0],
                dtype=torch.bool,
                device=sample.device
            )

        if expr == sp.false:
            sample = next(iter(decisions.values()))
            return torch.zeros(
                sample.shape[0],
                dtype=torch.bool,
                device=sample.device
            )

        # Logical operators
        if isinstance(expr, sp.And):
            result = self._evaluate_boolean(expr.args[0], decisions)
            for arg in expr.args[1:]:
                result = result & self._evaluate_boolean(arg, decisions)
            return result

        if isinstance(expr, sp.Or):
            result = self._evaluate_boolean(expr.args[0], decisions)
            for arg in expr.args[1:]:
                result = result | self._evaluate_boolean(arg, decisions)
            return result

        if isinstance(expr, sp.Not):
            return ~self._evaluate_boolean(expr.args[0], decisions)

        if isinstance(expr, sp.Implies):
            antecedent = self._evaluate_boolean(expr.args[0], decisions)
            consequent = self._evaluate_boolean(expr.args[1], decisions)
            # A → B ≡ ¬A ∨ B
            return (~antecedent) | consequent

        if isinstance(expr, sp.Equivalent):
            left = self._evaluate_boolean(expr.args[0], decisions)
            right = self._evaluate_boolean(expr.args[1], decisions)
            # A ↔ B ≡ (A ↔ B)
            return left == right

        raise ValueError(f"Unsupported expression type: {type(expr)}")

    def __call__(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Check if formula is satisfied on inputs.

        Args:
            inputs: Single tensor (batch_size, ...) for all predicates,
                   or dict mapping predicate names to specific inputs

        Returns:
            Boolean tensor of shape (batch_size,) indicating whether
            the formula is satisfied for each example

        Example:
            >>> satisfied = checker(x)
            >>> num_satisfied = satisfied.sum().item()
            >>> fraction_satisfied = satisfied.float().mean().item()
        """
        # Evaluate predicates to get boolean decisions
        decisions = self._evaluate_predicates(inputs)

        # Evaluate formula with boolean logic
        return self._evaluate_boolean(self.expression, decisions)
