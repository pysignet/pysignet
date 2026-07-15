"""Base class for logic compilation strategies."""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

import sympy as sp
import torch
import torch.nn as nn

from pysignet.compilation.arity import validate_predicate_arity
from pysignet.compilation.compiled_expression import CompiledExpression
from pysignet.compilation.module_utils import (
    infer_module_arity,
    resolve_variable_inputs,
    split_model_and_index_vars,
)
from pysignet.context import EvaluationContext
from pysignet.logic.expansion import expand_quantifier
from pysignet.logic.quantifier import Quantifier
from pysignet.logic.variable import VariableSymbol
from pysignet.predicate import Predicate
from pysignet.symbols import PredicateApplication


class LogicCompiler(ABC):
    """Abstract base class for compiling logic expressions into differentiable
    computations.

    LogicCompiler defines the interface for different compilation strategies
    (t-norms, LTU, semantic loss, etc.). Each strategy compiles a SymPy logic
    expression into a PyTorch callable that returns satisfaction degrees.

    Subclasses must implement:
    - compile(): Compile SymPy expression to CompiledExpression
    - conjunction(values): Relaxed AND, reducing along dim=0
    - disjunction(values): Relaxed OR, reducing along dim=0
    - recommended_postprocessing: 'log' or 'linear'

    Default implementations are provided for negation (1 - a),
    implication (NOT a OR b), and equivalence ((a -> b) AND (b -> a)).

    Conjunction and disjunction operate on a tensor reducing along dim=0.
    This supports both n-ary expression evaluation (stacking multiple
    per-batch tensors) and batch reduction (reducing a 1D tensor to scalar).

    Class Attributes:
        MAX_DOMAIN_SIZE: Maximum domain size for quantifiers (default: 1000)
        WARN_DOMAIN_SIZE: Domain size threshold for warnings (default: 100)
    """

    # Configurable domain size limits for quantifier expansion
    MAX_DOMAIN_SIZE = 1000
    WARN_DOMAIN_SIZE = 100

    # Minimum number of distinct leaf atoms before the opt-in jit=True
    # path wraps the combinator-dispatch step in torch.compile. Below
    # this, tracing overhead is not worth it and the eager path is used
    # even when jit=True.
    JIT_SIZE_THRESHOLD = 8

    # Subclasses that support the opt-in JIT path set self.jit in their
    # __init__. This class-level default keeps it False (Phase 1:
    # opt-in) for any subclass that does not set it explicitly.
    jit = False

    @property
    @abstractmethod
    def recommended_postprocessing(self) -> str:
        """Return recommended loss post-processing mode.

        Returns:
            'log' for -log(satisfaction) or 'linear' for 1 - satisfaction
        """

    @abstractmethod
    def conjunction(self, values: torch.Tensor) -> torch.Tensor:
        """Relaxed AND operation, reducing along dim=0.

        Args:
            values: Tensor of shape (n, ...) with values in [0, 1].

        Returns:
            Tensor of shape (...) with conjunction applied.
        """

    @abstractmethod
    def disjunction(self, values: torch.Tensor) -> torch.Tensor:
        """Relaxed OR operation, reducing along dim=0.

        Args:
            values: Tensor of shape (n, ...) with values in [0, 1].

        Returns:
            Tensor of shape (...) with disjunction applied.
        """

    def _is_product_conjunction(self) -> bool:
        """Check if conjunction is product-based (log-fusible).

        Product conjunction satisfies log(AND(a,b)) = log(a) + log(b),
        enabling log-space fusion. Override in subclasses to return
        True for product-based t-norms.

        Returns:
            True if conjunction is product, False otherwise.
        """
        return False

    def negation(self, a: torch.Tensor) -> torch.Tensor:
        """Relaxed NOT operation: 1 - a.

        Args:
            a: Tensor with values in [0, 1].

        Returns:
            Tensor of same shape with negation applied.
        """
        result: torch.Tensor = 1.0 - a
        return result

    def implication(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Relaxed IMPLIES: a -> b = NOT(a) OR b.

        Subclasses may override for custom implication semantics
        (e.g., R-Product residuum).

        Args:
            a: Antecedent tensor (values in [0, 1])
            b: Consequent tensor (values in [0, 1])

        Returns:
            Implication result tensor.
        """
        return self.disjunction(torch.stack([self.negation(a), b]))

    def equivalence(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Relaxed EQUIVALENCE: a <-> b = (a -> b) AND (b -> a).

        Args:
            a: Left tensor (values in [0, 1])
            b: Right tensor (values in [0, 1])

        Returns:
            Equivalence result tensor.
        """
        # pylint: disable=arguments-out-of-order
        return self.conjunction(
            torch.stack([self.implication(a, b), self.implication(b, a)])
        )

    @abstractmethod
    def compile(
        self,
        expr: sp.Basic,
        predicates: dict[str, Predicate | Callable[..., torch.Tensor]],
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

    def _validate_module_arity(
        self,
        key: str,
        module: nn.Module,
        expected_arity: int,
    ) -> None:
        """Validate nn.Module arity matches expected usage.

        Args:
            key: Predicate name
            module: nn.Module to validate
            expected_arity: Expected arity from expression usage

        Raises:
            ValueError: If module arity doesn't match expected usage
        """
        module_arity = infer_module_arity(module)

        # Skip validation if arity cannot be inferred
        if module_arity is None:
            return

        # Check compatibility
        is_valid = False
        if module_arity == 1:
            # Unary module: must have exactly 1 argument
            is_valid = expected_arity == 1
        elif module_arity == 2:
            # Binary/multi-output: 2+ args for channel select
            is_valid = expected_arity >= 2

        if not is_valid:
            arity_names = {1: "unary", 2: "binary"}
            module_name = arity_names.get(module_arity, f"arity-{module_arity}")
            raise ValueError(
                f"Predicate '{key}' arity mismatch: "
                f"module has {module_name} arity "
                f"(output dim = {module_arity}) but used with "
                f"{expected_arity} argument(s). "
                f"Unary needs 1 arg, binary needs 2+."
            )

    def _extract_module_to_check(
        self, value: Predicate | Callable[..., torch.Tensor]
    ) -> nn.Module | None:
        """Extract nn.Module from a predicate value if present.

        Args:
            value: Predicate or callable to check

        Returns:
            nn.Module if found, None otherwise
        """
        if isinstance(value, nn.Module):
            return value
        if isinstance(value, Predicate) and isinstance(value.func, nn.Module):
            return value.func
        return None

    def _wrap_predicate_value(
        self, key: str, value: Predicate | Callable[..., torch.Tensor]
    ) -> Predicate:
        """Wrap a predicate value in a Predicate object if needed.

        Args:
            key: Predicate name
            value: Value to wrap

        Returns:
            Predicate object

        Raises:
            TypeError: If value is not callable
        """
        if isinstance(value, Predicate):
            return value
        if callable(value):
            return Predicate(value)
        raise TypeError(
            f"Predicate '{key}' must be callable or Predicate, "
            f"got {type(value).__name__}"
        )

    def _wrap_and_validate_predicates(
        self,
        expr: sp.Basic,
        predicates: dict[str, Predicate | Callable[..., torch.Tensor]],
    ) -> dict[str, Predicate]:
        """Wrap, validate, and prepare predicates for compilation.

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
        wrapped_predicates: dict[str, Predicate] = {}
        for key, value in predicates.items():
            # Validate module arity if applicable
            module = self._extract_module_to_check(value)
            if module is not None and key in predicate_arities:
                self._validate_module_arity(key, module, predicate_arities[key])

            # Wrap the predicate value
            wrapped_predicates[key] = self._wrap_predicate_value(key, value)

        # Assign names, configure activation, and validate no reuse
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
            if key in predicate_arities:
                pred.configure_activation(predicate_arities[key])

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
                        f"for small symbolic sets (class labels, discrete "
                        f"choices). Max: {self.MAX_DOMAIN_SIZE}. "
                        f"Consider restructuring or sampling from domain."
                    )
                elif domain_size > self.WARN_DOMAIN_SIZE:
                    warnings.warn(
                        f"Large domain ({domain_size} elements) in "
                        f"{node.__class__.__name__} may impact performance. "
                        f"Consider using smaller domains or restructuring.",
                        UserWarning,
                        stacklevel=2,
                    )

                # Expand this quantifier
                expanded = expand_quantifier(node)
                # Recursively expand any nested quantifiers
                return _expand_recursive(expanded)

            # Recursive case: traverse children
            if hasattr(node, "args") and node.args:
                expanded_children = [
                    _expand_recursive(child) for child in node.args
                ]
                return node.func(*expanded_children)

            # Leaf node
            return node

        return _expand_recursive(expr)

    def _extract_predicate_arities(self, expr: sp.Basic) -> dict[str, int]:
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
        arities: dict[str, int] = {}

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
            for arg in getattr(e, "args", []):
                extract(arg)

        extract(expr)
        return arities

    def _extract_predicate_symbols(self, expr: sp.Basic) -> set[str]:
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

        symbols: set[str] = set()
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
        predicate_arities: dict[str, int] = {}

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
                                f"found both arity 0 (nullary) and arity "
                                f"{predicate_arities[name]} (n-ary) "
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
            for arg in getattr(e, "args", []):
                collect_usage(arg)

        collect_usage(expr)

    def _parse_predicate_application(
        self, app: PredicateApplication
    ) -> tuple[list[VariableSymbol], list[Any]]:
        """Parse PredicateApplication into free variables and constants.

        Extracts and deduplicates free variables, preserving order of first
        appearance. Collects constants separately.

        Args:
            app: PredicateApplication to parse

        Returns:
            Tuple of (free_vars, constants) where:
            - free_vars: Unique VariableSymbols in order of first appearance
            - constants: List of constant values (integers)

        Example:
            >>> P(X1, X2, 0, X1, 1)  # X1 appears twice
            >>> _parse_predicate_application(app)
            ([X1, X2], [0, 1])
        """
        # Use dict to preserve order while deduplicating
        free_vars_dict: dict[str, VariableSymbol] = {}
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
        inputs: dict[str, torch.Tensor],
        predicates: dict[str, Predicate],
        ctx: EvaluationContext,
        log_mode: bool = False,
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
                app, inputs, predicate, free_vars, constants, ctx,
                log_mode=log_mode,
            )
        else:
            # Regular callable: pass all arguments in order
            return self._evaluate_callable_predicate(
                app, inputs, predicate, ctx,
                log_mode=log_mode,
            )

    def _evaluate_callable_predicate(
        self,
        app: PredicateApplication,
        inputs: dict[str, torch.Tensor],
        predicate: Predicate,
        ctx: EvaluationContext,
        log_mode: bool = False,
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
        call_args: list[Any] = []
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

        # Create cache key - handle tensors and dicts with id()
        def _make_hashable(a: Any) -> Any:
            if isinstance(a, torch.Tensor):
                return id(a)
            elif isinstance(a, dict):
                # For dicts, use the id to avoid unhashable type error
                return ("dict", id(a))
            else:
                return a

        cache_key_base = (
            id(func), tuple(_make_hashable(a) for a in call_args)
        )
        cache_key = (
            ("log", cache_key_base) if log_mode else cache_key_base
        )

        # Call with all arguments
        if log_mode:
            result = ctx.get_or_compute(
                cache_key, lambda: predicate.log_call(*call_args)
            )
        else:
            result = ctx.get_or_compute(
                cache_key, lambda: predicate(*call_args)
            )
        return cast(torch.Tensor, result)

    def _resolve_var_inputs(
        self,
        variables: list[VariableSymbol],
        inputs: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Resolve variable symbols to their bound tensors.

        Delegates to the shared resolve_variable_inputs utility.

        Args:
            variables: List of variable symbols to resolve.
            inputs: Dict mapping variable names to tensors.

        Returns:
            List of tensors corresponding to each variable.

        Raises:
            ValueError: If a variable is missing from inputs.
        """
        return resolve_variable_inputs(variables, inputs)

    def _apply_variable_indices(
        self,
        result: torch.Tensor,
        index_vars: list[VariableSymbol],
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply per-element indexing using variable tensors.

        For each index variable, selects elements along dimension 1
        using the variable's bound tensor as per-element indices.
        For example, output[i, index[i]] for each batch element i.

        Args:
            result: Module output tensor, shape (batch, ...).
            index_vars: Variables whose values are output indices.
            inputs: Dict mapping variable names to tensors.

        Returns:
            Tensor after per-element index selection.
        """
        for var in index_vars:
            if result.dim() <= 1:
                break
            index_tensor = self._resolve_var_inputs([var], inputs)[0]
            result = result[
                torch.arange(result.shape[0]), index_tensor.long()
            ]
        return result

    def _apply_constant_indices(
        self, result: torch.Tensor, constants: list[Any]
    ) -> torch.Tensor:
        """Apply column selection using constant indices.

        For each constant, selects a fixed column (or slice) from the
        output along dimension 1.

        Args:
            result: Module output tensor, shape (batch, ...).
            constants: List of constant index values.

        Returns:
            Tensor after constant column selection.
        """
        for const in constants:
            if result.dim() <= 1:
                break
            result = result[:, const] if result.dim() == 2 else (
                result.select(dim=1, index=const)
            )
        return result

    def _evaluate_module_predicate(
        self,
        unused_app: PredicateApplication,
        inputs: dict[str, torch.Tensor],
        predicate: Predicate,
        free_vars: list[VariableSymbol],
        constants: list[Any],
        ctx: EvaluationContext,
        log_mode: bool = False,
    ) -> torch.Tensor:
        """Evaluate an nn.Module predicate.

        Passes variable inputs to the module, then uses constants and
        variable indices as output selectors.

        For multiclass modules, if the expression has more variable
        arguments than the module's forward() accepts, the extra
        variables are treated as per-element output indices. For
        example, Digit(X, Y) with a 10-class model calls model(X)
        and selects output[batch_idx, Y[batch_idx]].

        Args:
            unused_app: PredicateApplication (unused, for API consistency)
            inputs: Dict of variable bindings
            predicate: Predicate wrapping nn.Module
            free_vars: List of free variables in the application
            constants: List of constant indices
            ctx: EvaluationContext for caching

        Returns:
            Tensor of shape (batch_size,)
        """
        func = predicate.func

        # Determine how many free_vars are model inputs vs indices.
        # For multiclass modules, extra variables act as output indices.
        assert isinstance(func, nn.Module)
        model_vars, index_vars = split_model_and_index_vars(
            func, free_vars
        )

        # Resolve model input tensors
        var_inputs = self._resolve_var_inputs(model_vars, inputs)

        # Cache key based on model inputs only (indices don't affect
        # the forward pass). Use separate namespace for log_mode.
        cache_key_base = (
            id(func), tuple(id(inp) for inp in var_inputs)
        )
        cache_key = (
            ("log", cache_key_base) if log_mode
            else cache_key_base
        )

        # Call module with model inputs only
        if log_mode:
            full_output = ctx.get_or_compute(
                cache_key,
                lambda: predicate.log_call(*var_inputs),
            )
        else:
            full_output = ctx.get_or_compute(
                cache_key, lambda: predicate(*var_inputs)
            )
        full_output = cast(torch.Tensor, full_output)

        # Apply output selection (variable indices, then constants)
        if not index_vars and not constants:
            if full_output.dim() == 2 and full_output.shape[1] == 1:
                return full_output.squeeze(-1)
            return full_output

        result = self._apply_variable_indices(
            full_output, index_vars, inputs
        )
        return self._apply_constant_indices(result, constants)

    def _evaluate_boolean_constant(
        self, const: sp.Basic, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Evaluate boolean constant (sp.true or sp.false).

        Args:
            const: sp.true or sp.false
            inputs: Dict mapping variable names to tensors (for batch size)

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

    def _evaluate_expression(
        self,
        expr: sp.Basic,
        inputs: dict[str, torch.Tensor],
        predicates: dict[str, Predicate],
        ctx: EvaluationContext,
    ) -> torch.Tensor:
        """Recursively evaluate SymPy expression using logical operations.

        This is the core evaluation method shared by all compilers. Each
        compiler provides its own implementations of conjunction, disjunction,
        negation, implication, and equivalence.

        Args:
            expr: SymPy expression to evaluate
            inputs: Dict mapping variable names to tensors
            predicates: Dict of predicates
            ctx: Evaluation context for caching

        Returns:
            Tensor of shape (batch_size,) with values in [0, 1]

        Raises:
            ValueError: For unsupported expression types or bare symbols
        """
        # Base case: PredicateApplication
        if isinstance(expr, PredicateApplication):
            return self._evaluate_predicate_application(
                expr, inputs, predicates, ctx
            )

        # Reject bare symbols (nullary predicates not supported)
        if isinstance(expr, sp.Symbol):
            raise ValueError(
                f"Bare symbol '{expr}' is not supported. "
                f"All predicates must be called with at least one "
                f"variable argument "
                f"(e.g., use P(X) instead of P)."
            )

        # Boolean constants
        if expr in (sp.true, sp.false):
            return self._evaluate_boolean_constant(expr, inputs)

        # Logical operators - delegate to operator methods
        if isinstance(expr, sp.And):
            return self._evaluate_and(expr, inputs, predicates, ctx)

        if isinstance(expr, sp.Or):
            return self._evaluate_or(expr, inputs, predicates, ctx)

        if isinstance(expr, sp.Not):
            return self._evaluate_not(expr, inputs, predicates, ctx)

        if isinstance(expr, sp.Implies):
            return self._evaluate_implies(expr, inputs, predicates, ctx)

        if isinstance(expr, sp.Equivalent):
            return self._evaluate_equivalent(expr, inputs, predicates, ctx)

        raise ValueError(f"Unsupported expression type: {type(expr)}")

    def _evaluate_and(
        self,
        expr: sp.And,
        inputs: dict[str, torch.Tensor],
        predicates: dict[str, Predicate],
        ctx: EvaluationContext,
    ) -> torch.Tensor:
        """Evaluate And expression."""
        args = [
            self._evaluate_expression(a, inputs, predicates, ctx)
            for a in expr.args
        ]
        return self.conjunction(torch.stack(args))

    def _evaluate_or(
        self,
        expr: sp.Or,
        inputs: dict[str, torch.Tensor],
        predicates: dict[str, Predicate],
        ctx: EvaluationContext,
    ) -> torch.Tensor:
        """Evaluate Or expression."""
        args = [
            self._evaluate_expression(a, inputs, predicates, ctx)
            for a in expr.args
        ]
        return self.disjunction(torch.stack(args))

    def _evaluate_not(
        self,
        expr: sp.Not,
        inputs: dict[str, torch.Tensor],
        predicates: dict[str, Predicate],
        ctx: EvaluationContext,
    ) -> torch.Tensor:
        """Evaluate Not expression."""
        return self.negation(
            self._evaluate_expression(expr.args[0], inputs, predicates, ctx)
        )

    def _evaluate_implies(
        self,
        expr: sp.Implies,
        inputs: dict[str, torch.Tensor],
        predicates: dict[str, Predicate],
        ctx: EvaluationContext,
    ) -> torch.Tensor:
        """Evaluate Implies expression."""
        return self.implication(
            self._evaluate_expression(expr.args[0], inputs, predicates, ctx),
            self._evaluate_expression(expr.args[1], inputs, predicates, ctx),
        )

    def _evaluate_equivalent(
        self,
        expr: sp.Equivalent,
        inputs: dict[str, torch.Tensor],
        predicates: dict[str, Predicate],
        ctx: EvaluationContext,
    ) -> torch.Tensor:
        """Evaluate Equivalent expression."""
        return self.equivalence(
            self._evaluate_expression(expr.args[0], inputs, predicates, ctx),
            self._evaluate_expression(expr.args[1], inputs, predicates, ctx),
        )

    def _evaluate_expression_log(
        self,
        expr: sp.Basic,
        inputs: dict[str, torch.Tensor],
        predicates: dict[str, Predicate],
        ctx: EvaluationContext,
    ) -> torch.Tensor:
        """Evaluate expression in log-space for numerical stability.

        For predicate applications, uses fused log-activation ops
        (logsigmoid, log_softmax). For product conjunction (And),
        computes sum of logs instead of log of product.

        Other operators (Or, Not, Implies, Equivalent) fall back to
        linear-space evaluation then take log, since they do not
        benefit from log-space fusion.

        Args:
            expr: SymPy expression to evaluate
            inputs: Dict mapping variable names to tensors
            predicates: Dict of predicates
            ctx: Evaluation context for caching

        Returns:
            Tensor of shape (batch_size,) with log-satisfaction values
        """
        # Predicate application: use fused log ops
        if isinstance(expr, PredicateApplication):
            return self._evaluate_predicate_application(
                expr, inputs, predicates, ctx, log_mode=True
            )

        # Boolean constants
        if expr in (sp.true, sp.false):
            linear = self._evaluate_boolean_constant(expr, inputs)
            return torch.log(linear + 1e-10)

        # Product conjunction: log(prod(a_i)) = sum(log(a_i))
        # Only valid for product t-norms. Non-product (Godel,
        # Lukasiewicz) fall back to linear-space.
        if isinstance(expr, sp.And) and self._is_product_conjunction():
            args = [
                self._evaluate_expression_log(
                    a, inputs, predicates, ctx
                )
                for a in expr.args
            ]
            return torch.stack(args).sum(dim=0)

        # All other operators (and non-product And): fall back
        linear = self._evaluate_expression(
            expr, inputs, predicates, ctx
        )
        return torch.log(linear + 1e-10)

    def _collect_leaves(
        self, expr: sp.Basic
    ) -> list[PredicateApplication]:
        """Collect unique PredicateApplication leaves in evaluation order.

        Args:
            expr: SymPy expression to walk (after quantifier expansion)

        Returns:
            List of unique PredicateApplication nodes in first-occurrence
            order. Duplicate ground atoms (same predicate name and
            arguments) are deduplicated via PredicateApplication's
            structural equality/hash.
        """
        seen: dict[PredicateApplication, None] = {}

        def _walk(node: sp.Basic) -> None:
            if isinstance(node, PredicateApplication):
                seen.setdefault(node, None)
                return
            for arg in getattr(node, "args", ()):
                _walk(arg)

        _walk(expr)
        return list(seen.keys())

    def _build_combine_fn(
        self,
        expr: sp.Basic,
        leaf_order: list[PredicateApplication],
    ) -> Callable[[list[torch.Tensor]], torch.Tensor]:
        """Build a pure tensor-only function combining leaf values.

        Unlike _evaluate_expression, this function never calls a
        predicate -- it only combines already-evaluated leaf tensors
        according to the (fixed) formula structure. This is what makes
        it safe to wrap in torch.compile: its Python control flow
        depends only on the closed-over SymPy structure, never on
        tensor values or arbitrary user predicate code.

        Args:
            expr: SymPy expression (after quantifier expansion)
            leaf_order: Unique leaf atoms, as returned by
                _collect_leaves(expr)

        Returns:
            Callable taking a list of leaf tensors (in leaf_order) and
            returning the combined satisfaction tensor.
        """
        leaf_index = {atom: i for i, atom in enumerate(leaf_order)}

        def _combine(
            node: sp.Basic, leaf_values: list[torch.Tensor]
        ) -> torch.Tensor:
            if isinstance(node, PredicateApplication):
                return leaf_values[leaf_index[node]]

            if isinstance(node, sp.Symbol):
                raise ValueError(
                    f"Bare symbol '{node}' is not supported. "
                    f"All predicates must be called with at least one "
                    f"variable argument (e.g., use P(X) instead of P)."
                )

            # leaf_values[0] is safe here: an all-constant expression
            # has zero leaf atoms, which keeps use_jit False in
            # _build_evaluator, so this branch is never reached with an
            # empty leaf_values.
            if node in (sp.true, sp.false):
                return (
                    torch.ones_like(leaf_values[0])
                    if node == sp.true
                    else torch.zeros_like(leaf_values[0])
                )

            if isinstance(node, sp.And):
                return self.conjunction(
                    torch.stack(
                        [_combine(a, leaf_values) for a in node.args]
                    )
                )
            if isinstance(node, sp.Or):
                return self.disjunction(
                    torch.stack(
                        [_combine(a, leaf_values) for a in node.args]
                    )
                )
            if isinstance(node, sp.Not):
                return self.negation(_combine(node.args[0], leaf_values))
            if isinstance(node, sp.Implies):
                left, right = node.args
                return self.implication(
                    _combine(left, leaf_values),
                    _combine(right, leaf_values),
                )
            if isinstance(node, sp.Equivalent):
                left, right = node.args
                return self.equivalence(
                    _combine(left, leaf_values),
                    _combine(right, leaf_values),
                )

            raise ValueError(
                f"Unsupported expression type for jit combine: "
                f"{type(node)}"
            )

        def combine(leaf_values: list[torch.Tensor]) -> torch.Tensor:
            return _combine(expr, leaf_values)

        return combine

    def _build_evaluator(
        self,
        expanded_expr: sp.Basic,
        predicates: dict[str, Predicate],
    ) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        """Build the compiled_logic closure for a compiled expression.

        Uses the opt-in jit=True combinator path (torch.compile) when
        the formula has at least JIT_SIZE_THRESHOLD leaf atoms;
        otherwise -- including whenever jit=False, the Phase 1 default
        (TODO.md 2.21) -- returns the ordinary eager evaluator.

        Args:
            expanded_expr: SymPy expression after quantifier expansion
            predicates: Dict of wrapped predicates

        Returns:
            Callable taking a dict of variable bindings and returning
            the satisfaction tensor of shape (batch_size,).
        """
        leaf_order = self._collect_leaves(expanded_expr)
        use_jit = self.jit and len(leaf_order) >= self.JIT_SIZE_THRESHOLD

        if not use_jit:
            def compiled_logic(
                inputs: dict[str, torch.Tensor]
            ) -> torch.Tensor:
                ctx = EvaluationContext()
                return self._evaluate_expression(
                    expanded_expr, inputs, predicates, ctx
                )

            return compiled_logic

        combine = torch.compile(
            self._build_combine_fn(expanded_expr, leaf_order)
        )

        def compiled_logic_jit(
            inputs: dict[str, torch.Tensor]
        ) -> torch.Tensor:
            ctx = EvaluationContext()
            leaf_values = [
                self._evaluate_predicate_application(
                    atom, inputs, predicates, ctx
                )
                for atom in leaf_order
            ]
            return combine(leaf_values)

        return compiled_logic_jit
