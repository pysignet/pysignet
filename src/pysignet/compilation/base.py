"""Base class for logic compilation strategies."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Set, Union, cast
import warnings

import sympy as sp
import torch
import torch.nn as nn

from pysignet.context import EvaluationContext
from pysignet.predicate import Predicate
from pysignet.multiclass import PredicateApplication
from pysignet.logic.quantifier import Quantifier
from pysignet.logic.expansion import expand_quantifier
from pysignet.logic.variable import VariableSymbol
from pysignet.compilation.arity import validate_predicate_arity
from pysignet.compilation.module_utils import (
    infer_module_arity,
    wrap_module_as_predicate
)
from pysignet.compilation.compiled_expression import CompiledExpression


class LogicCompiler(ABC):
    """Abstract base class for compiling logic expressions into differentiable
    computations.

    LogicCompiler defines the interface for different compilation strategies
    (t-norms, semantic loss, etc.). Each strategy compiles a SymPy logic
    expression into a PyTorch callable that returns satisfaction degrees.

    The compiled callable can be used directly or wrapped in a LogicLoss for
    loss computation.

    Class Attributes:
        MAX_DOMAIN_SIZE: Maximum allowed domain size for quantifiers (default: 1000)
        WARN_DOMAIN_SIZE: Domain size threshold for warnings (default: 100)
    """

    # Configurable domain size limits for quantifier expansion
    MAX_DOMAIN_SIZE = 1000
    WARN_DOMAIN_SIZE = 100

    @abstractmethod
    def compile(
            self,
            expr: sp.Basic,
            predicates: Dict[str, Union[Predicate, Callable[..., torch.Tensor]]],
    ) -> CompiledExpression:
        """Compile a logic expression into a differentiable CompiledExpression.

        Args:
            expr: SymPy logic expression (e.g., sp.And(P, sp.Or(Q, sp.Not(R))))
            predicates: Dict mapping predicate names to Predicate objects or to
                callables that produce tensors

        Returns:
            CompiledExpression that can be evaluated with variable bindings,
            supports partial binding, and provides introspection.

        Raises:
            ValueError: If symbols in expr have no corresponding predicates
        """
        pass

    def _wrap_and_validate_predicates(
        self,
        expr: sp.Basic,
        predicates: Dict[str, Union[Predicate, Callable[..., torch.Tensor]]]
    ) -> Dict[str, Predicate]:
        """Wrap, validate, and prepare predicates for compilation.

        This method:
        1. Expands quantifiers
        2. Extracts predicate usages and arities from expression
        3. Wraps nn.Modules with smart detection (sigmoid/softmax)
        4. Auto-wraps raw callables in Predicate objects
        5. Assigns names and validates no reuse
        6. Validates predicate usage consistency
        7. Validates predicate arity

        Args:
            expr: SymPy expression to compile
            predicates: Dict mapping predicate names to callables or Predicates

        Returns:
            Dict of wrapped and validated Predicate objects

        Raises:
            ValueError: If validation fails
            TypeError: If predicate values are not callable
        """
        # Expand quantifiers FIRST (needed for arity extraction)
        expanded_expr = self._expand_quantifiers(expr)

        # Extract predicate usages and their arities from expression
        predicate_arities = self._extract_predicate_arities(expanded_expr)

        # Wrap predicates and validate nn.Module arity
        wrapped_predicates: Dict[str, Predicate] = {}
        for key, value in predicates.items():
            # Extract module if wrapped in Predicate
            module_to_check: nn.Module | None = None
            if isinstance(value, nn.Module):
                module_to_check = value
            elif isinstance(value, Predicate) and isinstance(value.func, nn.Module):
                module_to_check = value.func

            if module_to_check is not None and key in predicate_arities:
                # Try to validate nn.Module arity matches usage
                # For unary modules (output dim=1): expect exactly 1 argument
                # For binary modules (output dim>1): expect 2+ arguments (variable + constant(s))
                # For custom modules: skip validation (arity inferred from usage)
                expected_arity = predicate_arities[key]
                module_arity = infer_module_arity(module_to_check)

                # Only validate if arity can be inferred
                if module_arity is not None:
                    # Check compatibility
                    is_valid = False
                    if module_arity == 1:
                        # Unary module: must have exactly 1 argument
                        is_valid = (expected_arity == 1)
                    elif module_arity == 2:
                        # Binary/multi-output module: must have 2+ arguments
                        # Allows P(X, 0), P(X, 0, 1), etc. for channel selection
                        is_valid = (expected_arity >= 2)

                    if not is_valid:
                        arity_names = {1: "unary", 2: "binary"}
                        module_name = arity_names.get(module_arity, f"arity-{module_arity}")
                        raise ValueError(
                            f"Predicate '{key}' arity mismatch: module has {module_name} arity "
                            f"(output dim = {module_arity}) but used with "
                            f"{expected_arity} argument(s). "
                            f"Unary modules need exactly 1 argument, binary modules need 2+ arguments."
                        )

            # Auto-wrap raw callables in Predicate objects
            # Note: nn.Modules are NOT wrapped - they're handled specially in evaluation
            if isinstance(value, Predicate):
                wrapped_predicates[key] = value
            elif callable(value):
                wrapped_predicates[key] = Predicate(value)
            else:
                raise TypeError(
                    f"Predicate '{key}' must be callable (function, lambda, "
                    f"nn.Module) or a Predicate instance, got {type(value).__name__}"
                )

        # Assign names and validate no reuse
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
        symbols = self._extract_predicate_symbols(expanded_expr)
        missing = symbols - set(wrapped_predicates.keys())
        if missing:
            raise ValueError(f"Missing predicates for symbols: {missing}")

        # Validate predicate usage consistency
        self._validate_predicate_usage(expanded_expr)

        # Validate predicate arity using new clean validator
        validate_predicate_arity(expanded_expr, wrapped_predicates)

        return wrapped_predicates

    def _expand_quantifiers(self, expr: sp.Basic) -> sp.Basic:
        """Recursively expand all quantifiers in the expression.

        Expands ForAll and Exists quantifiers into conjunctions and
        disjunctions over their domains. Includes safeguards for large domains.

        Args:
            expr: Expression potentially containing quantifiers

        Returns:
            Expression with all quantifiers expanded

        Raises:
            ValueError: If domain size exceeds MAX_DOMAIN_SIZE
        """
        def _expand_recursive(node: sp.Basic) -> sp.Basic:
            """Recursively expand quantifiers in the expression tree."""
            if isinstance(node, Quantifier):
                # Check domain size
                domain_list = list(node.domain)
                domain_size = len(domain_list)

                if domain_size > self.MAX_DOMAIN_SIZE:
                    raise ValueError(
                        f"Domain too large ({domain_size} elements) in "
                        f"{node.__class__.__name__}. Domain quantification is "
                        f"intended for small symbolic sets (class labels, "
                        f"discrete choices). Maximum allowed: {self.MAX_DOMAIN_SIZE}. "
                        f"Consider restructuring your constraint or sampling "
                        f"from the domain."
                    )
                elif domain_size > self.WARN_DOMAIN_SIZE:
                    warnings.warn(
                        f"Large domain ({domain_size} elements) in "
                        f"{node.__class__.__name__} may impact performance. "
                        f"Consider using smaller domains or restructuring.",
                        UserWarning
                    )

                # Expand this quantifier
                expanded = expand_quantifier(node)
                # Recursively expand any nested quantifiers
                return _expand_recursive(expanded)

            # Recursive case: traverse children
            if hasattr(node, 'args') and node.args:
                expanded_children = [_expand_recursive(child) for child in node.args]
                return node.func(*expanded_children)

            # Leaf node
            return node

        return _expand_recursive(expr)

    def _extract_predicate_arities(self, expr: sp.Basic) -> Dict[str, int]:
        """Extract predicate names and their arities from expression.

        Validates that each predicate is used consistently with the same arity.

        Args:
            expr: SymPy expression to analyze

        Returns:
            Dict mapping predicate names to their arities

        Raises:
            ValueError: If predicate used inconsistently with different arities

        Example:
            expr = sp.And(P(X), Q(X, Y))
            → {"P": 1, "Q": 2}
        """
        arities: Dict[str, int] = {}

        def extract(e: sp.Basic) -> None:
            if isinstance(e, PredicateApplication):
                pred_name = e.predicate_name
                arity = len(e.application_args)

                # Check consistency
                if pred_name in arities:
                    if arities[pred_name] != arity:
                        raise ValueError(
                            f"Predicate '{pred_name}' used inconsistently: "
                            f"found both arity {arities[pred_name]} and "
                            f"arity {arity} in the same expression."
                        )
                else:
                    arities[pred_name] = arity

            # Recurse into subexpressions
            for arg in getattr(e, 'args', []):
                extract(arg)

        extract(expr)
        return arities

    def _extract_predicate_symbols(self, expr: sp.Basic) -> Set[str]:
        """Extract all predicate symbols from a SymPy expression.

        Handles both regular SymPy symbols and PredicateApplication nodes.

        Args:
            expr: SymPy expression to analyze

        Returns:
            Set of symbol names (strings)
        """
        if isinstance(expr, sp.Symbol):
            return {str(expr)}

        if isinstance(expr, PredicateApplication):
            return {expr.predicate_name}

        symbols: Set[str] = set()
        for arg in expr.args:
            symbols.update(self._extract_predicate_symbols(arg))
        return symbols

    def _validate_predicate_usage(self, expr: sp.Basic) -> None:
        """Validate that predicates are used with consistent arity.

        Ensures each predicate appears with the same arity throughout.

        Args:
            expr: SymPy expression to validate

        Raises:
            ValueError: If any predicate is used with inconsistent arity
        """
        predicate_arities: Dict[str, int] = {}

        def collect_usage(e: sp.Basic) -> None:
            """Recursively collect predicate usage."""
            if isinstance(e, sp.Symbol):
                # Nullary (arity 0)
                if not isinstance(e, PredicateApplication):
                    name = str(e)
                    if name in predicate_arities:
                        if predicate_arities[name] != 0:
                            raise ValueError(
                                f"Predicate '{name}' used inconsistently: "
                                f"found both arity 0 (nullary: {name}) and arity "
                                f"{predicate_arities[name]} (n-ary: {name}(...)) "
                                f"in the same expression."
                            )
                    else:
                        predicate_arities[name] = 0

            elif isinstance(e, PredicateApplication):
                # N-ary (arity > 0)
                name = e.predicate_name
                arity = len(e.application_args)

                if name in predicate_arities:
                    if predicate_arities[name] != arity:
                        raise ValueError(
                            f"Predicate '{name}' used inconsistently: "
                            f"found both arity {predicate_arities[name]} and "
                            f"arity {arity} in the same expression."
                        )
                else:
                    predicate_arities[name] = arity

            # Recurse into subexpressions
            for arg in getattr(e, 'args', []):
                collect_usage(arg)

        collect_usage(expr)

    def _parse_predicate_application(
        self,
        app: PredicateApplication
    ) -> tuple[list[VariableSymbol], list[Any]]:
        """Parse PredicateApplication into free variables and constants.

        Extracts and deduplicates free variables, preserving order of first
        appearance. Collects constants separately.

        Args:
            app: PredicateApplication to parse

        Returns:
            Tuple of (free_vars, constants) where:
            - free_vars: List of unique VariableSymbol in order of first appearance
            - constants: List of constant values (integers)

        Example:
            >>> P(X1, X2, 0, X1, 1)  # X1 appears twice
            >>> _parse_predicate_application(app)
            ([X1, X2], [0, 1])
        """
        # Use dict to preserve order while deduplicating
        free_vars_dict: Dict[str, VariableSymbol] = {}
        constants: list[Any] = []

        for arg in app.application_args:
            if isinstance(arg, VariableSymbol):
                var_name = str(arg)
                if var_name not in free_vars_dict:
                    free_vars_dict[var_name] = arg
            else:
                # Constant argument
                constants.append(arg)

        free_vars = list(free_vars_dict.values())
        return free_vars, constants

    def _evaluate_predicate_application(
        self,
        app: PredicateApplication,
        inputs: Dict[str, torch.Tensor],
        predicates: Dict[str, Predicate],
        ctx: EvaluationContext
    ) -> torch.Tensor:
        """Evaluate PredicateApplication with variable and constant arguments.

        All predicates must have at least one free variable. Inputs must be
        provided as a dict mapping variable names to tensors.

        For regular callables: all arguments (variables + constants) are passed
        in the order they appear in the application.

        For nn.Module: variable inputs are passed to the module, and constants
        are used as output indices to select specific output channels.

        Args:
            app: PredicateApplication to evaluate (e.g., P(X), Q(X, 0))
            inputs: Dict mapping variable names to tensors (e.g., {"X": tensor})
            predicates: Dict of predicates
            ctx: EvaluationContext for caching

        Returns:
            Tensor of shape (batch_size,) with values in [0, 1]

        Raises:
            ValueError: If inputs is not a dict
            ValueError: If predicate has no free variables
            ValueError: If required variable is missing from inputs
        """
        # Require dict inputs (keyword argument API)
        if not isinstance(inputs, dict):
            raise ValueError(
                "Inputs must be a dict mapping variable names to tensors. "
                "Use keyword arguments like compiled(X=tensor) instead of "
                "positional arguments like compiled(tensor)."
            )

        pred_name = app.predicate_name
        predicate = predicates[pred_name]
        func = predicate.func

        # Parse application into free vars and constants
        free_vars, constants = self._parse_predicate_application(app)

        # Determine how to handle based on predicate type
        is_module = isinstance(func, torch.nn.Module)

        if is_module:
            # nn.Module: pass variable inputs, use constants as output indices
            return self._evaluate_module_predicate(
                app, inputs, predicate, free_vars, constants, ctx
            )
        else:
            # Regular callable: pass all arguments in order
            return self._evaluate_callable_predicate(
                app, inputs, predicate, ctx
            )

    def _evaluate_callable_predicate(
        self,
        app: PredicateApplication,
        inputs: Dict[str, torch.Tensor],
        predicate: Predicate,
        ctx: EvaluationContext
    ) -> torch.Tensor:
        """Evaluate a regular callable predicate.

        Passes all arguments (variables and constants) in the order they
        appear in the application.

        Args:
            app: PredicateApplication
            inputs: Dict of variable bindings
            predicate: Predicate to call
            ctx: EvaluationContext for caching

        Returns:
            Tensor of shape (batch_size,)
        """
        func = predicate.func

        # Build args in the order they appear in application_args
        call_args: List[Any] = []
        for arg in app.application_args:
            if isinstance(arg, VariableSymbol):
                var_name = str(arg)
                if var_name not in inputs:
                    raise ValueError(
                        f"Missing input for variable '{var_name}'. "
                        f"Expected key in input dict."
                    )
                call_args.append(inputs[var_name])
            else:
                # Constant - pass as-is
                call_args.append(arg)

        # Create cache key
        cache_key = (id(func), tuple(
            id(a) if isinstance(a, torch.Tensor) else a for a in call_args
        ))

        # Call with all arguments
        result = ctx.get_or_compute(cache_key, lambda: predicate(*call_args))
        return cast(torch.Tensor, result)

    def _evaluate_module_predicate(
        self,
        app: PredicateApplication,
        inputs: Dict[str, torch.Tensor],
        predicate: Predicate,
        free_vars: List[VariableSymbol],
        constants: List[Any],
        ctx: EvaluationContext
    ) -> torch.Tensor:
        """Evaluate an nn.Module predicate.

        Passes variable inputs to the module, then uses constants as output
        indices to select specific channels.

        Args:
            app: PredicateApplication
            inputs: Dict of variable bindings
            predicate: Predicate wrapping nn.Module
            free_vars: List of free variables in the application
            constants: List of constant indices
            ctx: EvaluationContext for caching

        Returns:
            Tensor of shape (batch_size,)
        """
        func = predicate.func

        # Extract variable inputs in order
        var_inputs: List[torch.Tensor] = []
        for var in free_vars:
            var_name = str(var)
            if var_name not in inputs:
                raise ValueError(
                    f"Missing input for variable '{var_name}'. "
                    f"Expected key in input dict."
                )
            var_inputs.append(inputs[var_name])

        # Create cache key
        cache_key = (id(func), tuple(id(inp) for inp in var_inputs))

        # Call module with variable inputs
        full_output = ctx.get_or_compute(
            cache_key,
            lambda: predicate(*var_inputs)
        )
        full_output = cast(torch.Tensor, full_output)

        # Handle constants as output indices
        if len(constants) > 0:
            result = full_output
            for const in constants:
                if result.dim() == 1:
                    # Already a batch of scalars, can't index further
                    break
                elif result.dim() == 2:
                    # Shape: (batch, num_outputs) - select column
                    result = result[:, const]
                else:
                    # Higher dimensional - index first non-batch dimension
                    result = result.select(dim=1, index=const)
            return result
        else:
            # No constants - return full output, squeeze if needed
            if full_output.dim() == 2 and full_output.shape[1] == 1:
                return full_output.squeeze(-1)
            return full_output

    def _evaluate_boolean_constant(
        self,
        const: sp.Basic,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Evaluate boolean constant (sp.true or sp.false).

        Args:
            const: sp.true or sp.false
            inputs: Dict mapping variable names to tensors (to determine batch size)

        Returns:
            Tensor of ones (true) or zeros (false) with shape (batch_size,)

        Raises:
            ValueError: If inputs is not a dict or is empty
        """
        if not isinstance(inputs, dict):
            raise ValueError(
                "Inputs must be a dict mapping variable names to tensors. "
                "Use keyword arguments like compiled(X=tensor) instead of "
                "positional arguments like compiled(tensor)."
            )
        if not inputs:
            raise ValueError("Inputs dict cannot be empty.")

        # Determine batch size from first input
        sample_input = next(iter(inputs.values()))
        batch_size = sample_input.shape[0]

        if const == sp.true:
            return torch.ones(batch_size, device=sample_input.device)
        elif const == sp.false:
            return torch.zeros(batch_size, device=sample_input.device)
        else:
            raise ValueError(f"Expected sp.true or sp.false, got {const}")
