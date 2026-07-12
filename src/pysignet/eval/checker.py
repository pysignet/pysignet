"""Consistency checking for neural models using hard boolean logic.

This module provides the ConsistencyChecker class, which evaluates
logical formulas using hard (boolean) decisions from model predictions.

ConsistencyChecker accepts Predicate objects and converts soft outputs
to boolean decisions using appropriate rules:
- Binary (sigmoid): threshold at 0.5
- Multiclass (softmax): argmax == class_idx
- Others: threshold at 0.5

ForAll/Exists quantifiers are expanded internally.

Example:
    >>> from pysignet import Symbol, Variable, Predicate
    >>> from pysignet.eval import ConsistencyChecker
    >>>
    >>> P = Symbol("P")
    >>> X = Variable("X")
    >>> model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
    >>>
    >>> checker = ConsistencyChecker(P(X), {"P": Predicate(model)})
    >>> satisfied = checker(X=x_batch)  # Boolean tensor
    >>> accuracy = satisfied.float().mean().item()
"""

from typing import Any

import sympy as sp
import torch

from pysignet.compilation.module_utils import (
    resolve_variable_inputs,
    split_model_and_index_vars,
)
from pysignet.eval.boolean import to_boolean
from pysignet.logic.variable import VariableSymbol
from pysignet.predicate import Predicate
from pysignet.symbols import PredicateApplication


class ConsistencyChecker:
    """Check how often a logical formula is satisfied by predictions.

    Evaluates logical formulas using hard (boolean) decisions from
    Predicate objects. Handles ForAll/Exists quantifiers by expanding
    them into And/Or.

    Boolean conversion rules:
    - Model predicates (is_model=True): pass variable tensors to
      model, use constants for output indexing. Multiclass outputs
      use argmax; binary outputs use threshold at 0.5.
    - Function predicates (is_model=False): pass all args in
      application order, threshold at 0.5.

    Args:
        expression: SymPy logic expression to check.
        predicates: Dict mapping predicate names to Predicate
            objects.

    Raises:
        ValueError: If any predicate symbol in the expression is
            missing from the predicates dict.
    """

    def __init__(
        self,
        expression: sp.Basic,
        predicates: dict[str, Predicate],
    ) -> None:
        # pylint: disable=import-outside-toplevel
        from pysignet.logic.expansion import (
            _expand_nested_quantifiers,
        )

        self._expression = _expand_nested_quantifiers(expression)
        self._predicates = predicates

        # Validate all predicate symbols are provided
        symbols = self._extract_predicate_symbols(
            self._expression
        )
        missing = symbols - set(predicates.keys())
        if missing:
            raise ValueError(
                f"Missing predicates for symbols: {missing}"
            )

        # Configure activation for nn.Module predicates using
        # expression-context arity (mirrors compilation path).
        # Without this, custom modules fall back to clamping
        # which corrupts argmax.
        self._configure_predicate_activations()

    def __call__(
        self, **variable_bindings: torch.Tensor
    ) -> torch.Tensor:
        """Check if formula is satisfied on inputs.

        Args:
            **variable_bindings: Variable bindings as keyword
                arguments (e.g., X=x_tensor, Y=y_tensor).

        Returns:
            Boolean tensor of shape (batch_size,) indicating
            whether the formula is satisfied for each example.
        """
        decisions = self._evaluate_predicates(variable_bindings)
        return self._evaluate_boolean(
            self._expression, decisions
        )

    # ----------------------------------------------------------
    # Predicate evaluation
    # ----------------------------------------------------------

    def _evaluate_predicates(
        self,
        bindings: dict[str, torch.Tensor],
    ) -> dict[PredicateApplication, torch.Tensor]:
        """Evaluate all predicate applications to boolean decisions.

        Args:
            bindings: Dict mapping variable names to tensors.

        Returns:
            Dict mapping PredicateApplication instances to boolean
            tensors.
        """
        decisions: dict[PredicateApplication, torch.Tensor] = {}
        model_cache: dict[
            tuple[int, tuple[int, ...]], torch.Tensor
        ] = {}

        applications = self._extract_predicate_applications(
            self._expression
        )

        for app in applications:
            pred_name = app.predicate_name
            predicate = self._predicates[pred_name]

            # Parse application args
            variables, constants = self._parse_args(app)

            if predicate.is_model:
                result = self._evaluate_model_predicate(
                    predicate, variables, constants,
                    bindings, model_cache,
                )
            else:
                result = self._evaluate_function_predicate(
                    predicate, app, bindings,
                )

            decisions[app] = result

        return decisions

    def _evaluate_model_predicate(
        self,
        predicate: Predicate,
        variables: list[VariableSymbol],
        constants: list[Any],
        bindings: dict[str, torch.Tensor],
        model_cache: dict[
            tuple[int, tuple[int, ...]], torch.Tensor
        ],
    ) -> torch.Tensor:
        """Evaluate nn.Module predicate to boolean.

        Splits variables into model inputs and output indices
        (mirroring the compilation path). Only model input
        variables are passed to the model; extra variables are
        used as per-element output indices via argmax comparison.

        Boolean conversion:
        - Multiclass (batch, C) with C > 1: argmax == class_idx
        - Binary (batch,) or (batch, 1): threshold at 0.5

        Model outputs are cached so the same model with the same
        inputs is only called once.

        Args:
            predicate: Predicate wrapping an nn.Module.
            variables: List of VariableSymbol in the application.
            constants: List of constant args in the application.
            bindings: Variable name -> tensor dict.
            model_cache: Cache for model outputs.

        Returns:
            Boolean tensor of shape (batch_size,).
        """
        assert isinstance(predicate.func, torch.nn.Module)

        # Split variables into model inputs and index variables
        model_vars, index_vars = split_model_and_index_vars(
            predicate.func, variables
        )

        var_tensors = resolve_variable_inputs(
            model_vars, bindings
        )

        # Cache model output (keyed on model inputs only)
        cache_key = (
            id(predicate.func),
            tuple(id(t) for t in var_tensors),
        )
        if cache_key not in model_cache:
            with torch.no_grad():
                model_cache[cache_key] = predicate(
                    *var_tensors
                )
        output = model_cache[cache_key]

        # Handle variable indices: per-element argmax comparison
        if index_vars:
            index_tensors = resolve_variable_inputs(
                index_vars, bindings
            )
            # For multiclass output, compare argmax to the
            # per-element index variable
            return to_boolean(
                output, class_idx=index_tensors[0]
            )

        # Handle constant indices
        int_consts = [
            c for c in constants if isinstance(c, int)
        ]
        class_idx: int | None = (
            int_consts[0] if int_consts else None
        )

        return to_boolean(output, class_idx=class_idx)

    def _evaluate_function_predicate(
        self,
        predicate: Predicate,
        app: PredicateApplication,
        bindings: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate function predicate to boolean.

        Passes all args in application order (variables resolved
        from bindings, constants as-is). Thresholds at 0.5.

        Args:
            predicate: Predicate wrapping a callable.
            app: The PredicateApplication being evaluated.
            bindings: Variable name -> tensor dict.

        Returns:
            Boolean tensor of shape (batch_size,).
        """
        call_args = self._build_call_args(app, bindings)

        with torch.no_grad():
            result = predicate(*call_args)

        return to_boolean(result)

    # ----------------------------------------------------------
    # Boolean formula evaluation
    # ----------------------------------------------------------

    def _evaluate_boolean(
        self,
        expr: sp.Basic,
        decisions: dict[PredicateApplication, torch.Tensor],
    ) -> torch.Tensor:
        """Recursively evaluate SymPy expression with boolean logic.

        Args:
            expr: SymPy expression to evaluate.
            decisions: Dict mapping PredicateApplication instances
                to boolean tensors.

        Returns:
            Boolean tensor indicating satisfaction.

        Raises:
            ValueError: If expression type is unsupported.
        """
        if isinstance(expr, PredicateApplication):
            return decisions[expr]

        if expr == sp.true:
            return self._make_bool_constant(decisions, True)

        if expr == sp.false:
            return self._make_bool_constant(decisions, False)

        if isinstance(expr, sp.And):
            result = self._evaluate_boolean(
                expr.args[0], decisions
            )
            for arg in expr.args[1:]:
                result = result & self._evaluate_boolean(
                    arg, decisions
                )
            return result

        if isinstance(expr, sp.Or):
            result = self._evaluate_boolean(
                expr.args[0], decisions
            )
            for arg in expr.args[1:]:
                result = result | self._evaluate_boolean(
                    arg, decisions
                )
            return result

        if isinstance(expr, sp.Not):
            return ~self._evaluate_boolean(
                expr.args[0], decisions
            )

        if isinstance(expr, sp.Implies):
            antecedent = self._evaluate_boolean(
                expr.args[0], decisions
            )
            consequent = self._evaluate_boolean(
                expr.args[1], decisions
            )
            return (~antecedent) | consequent

        if isinstance(expr, sp.Equivalent):
            left = self._evaluate_boolean(
                expr.args[0], decisions
            )
            right = self._evaluate_boolean(
                expr.args[1], decisions
            )
            return left == right

        raise ValueError(
            f"Unsupported expression type: {type(expr)}"
        )

    # ----------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------

    @staticmethod
    def _parse_args(
        app: PredicateApplication,
    ) -> tuple[list[VariableSymbol], list[Any]]:
        """Split application args into variables and constants.

        Args:
            app: PredicateApplication to parse.

        Returns:
            Tuple of (variables, constants).
        """
        variables: list[VariableSymbol] = []
        constants: list[Any] = []
        for arg in app.application_args:
            if isinstance(arg, VariableSymbol):
                variables.append(arg)
            else:
                constants.append(arg)
        return variables, constants

    @staticmethod
    def _build_call_args(
        app: PredicateApplication,
        bindings: dict[str, torch.Tensor],
    ) -> list[Any]:
        """Build call args for a function predicate.

        Resolves variables from bindings and passes constants
        as-is, preserving application argument order.

        Args:
            app: PredicateApplication to build args for.
            bindings: Variable name -> tensor dict.

        Returns:
            List of args in application order.

        Raises:
            ValueError: If a variable is missing from bindings.
        """
        call_args: list[Any] = []
        for arg in app.application_args:
            if isinstance(arg, VariableSymbol):
                var_name = str(arg)
                if var_name not in bindings:
                    raise ValueError(
                        f"Missing binding for variable "
                        f"'{var_name}'. Available: "
                        f"{sorted(bindings.keys())}"
                    )
                call_args.append(bindings[var_name])
            else:
                call_args.append(arg)
        return call_args

    @staticmethod
    def _make_bool_constant(
        decisions: dict[PredicateApplication, torch.Tensor],
        value: bool,
    ) -> torch.Tensor:
        """Create a boolean constant tensor matching batch size.

        Args:
            decisions: Dict of decisions (to determine shape).
            value: True or False.

        Returns:
            Boolean tensor of shape (batch_size,).
        """
        sample = next(iter(decisions.values()))
        if value:
            return torch.ones(
                sample.shape[0],
                dtype=torch.bool,
                device=sample.device,
            )
        return torch.zeros(
            sample.shape[0],
            dtype=torch.bool,
            device=sample.device,
        )

    def _configure_predicate_activations(self) -> None:
        """Configure activation on predicates from expression arity.

        Walks the expanded expression to find predicate arities,
        then calls configure_activation on each predicate. This
        ensures custom nn.Module predicates (non-Sequential) get
        softmax/sigmoid applied instead of falling back to
        clamping.
        """
        arities: dict[str, int] = {}
        for app in self._extract_predicate_applications(
            self._expression
        ):
            name = app.predicate_name
            arity = len(app.application_args)
            if name not in arities:
                arities[name] = arity

        for name, pred in self._predicates.items():
            if pred.name is None:
                pred.name = name
            if name in arities:
                pred.configure_activation(arities[name])

    def _extract_predicate_symbols(
        self, expr: sp.Basic
    ) -> set[str]:
        """Extract all predicate symbol names from expression.

        Args:
            expr: SymPy expression to scan.

        Returns:
            Set of predicate name strings.
        """
        if isinstance(expr, PredicateApplication):
            return {expr.predicate_name}

        if isinstance(expr, VariableSymbol):
            return set()

        symbols: set[str] = set()
        for arg in expr.args:
            symbols.update(
                self._extract_predicate_symbols(arg)
            )
        return symbols

    def _extract_predicate_applications(
        self, expr: sp.Basic
    ) -> set[PredicateApplication]:
        """Extract all unique PredicateApplications.

        Args:
            expr: SymPy expression to scan.

        Returns:
            Set of PredicateApplication instances.
        """
        applications: set[PredicateApplication] = set()

        if isinstance(expr, PredicateApplication):
            applications.add(expr)
            return applications

        if isinstance(expr, VariableSymbol):
            return applications

        for arg in expr.args:
            applications.update(
                self._extract_predicate_applications(arg)
            )

        return applications
