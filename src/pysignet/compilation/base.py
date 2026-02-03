"""Base class for logic compilation strategies."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Set, cast
import warnings

import sympy as sp
import torch
import torch.nn as nn

from pysignet.context import EvaluationContext
from pysignet.predicate import Predicate
from pysignet.symbols import PredicateApplication
from pysignet.logic.quantifier import Quantifier
from pysignet.logic.expansion import expand_quantifier
from pysignet.logic.variable import VariableSymbol
from pysignet.compilation.arity import validate_predicate_arity
from pysignet.compilation.module_utils import infer_module_arity
from pysignet.compilation.compiled_expression import CompiledExpression


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
        predicates: Dict[str, Predicate | Callable[..., torch.Tensor]],
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
        predicates: Dict[str, Predicate | Callable[..., torch.Tensor]],
    ) -> Dict[str, Predicate]:
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
        wrapped_predicates: Dict[str, Predicate] = {}
        for key, value in predicates.items():
            # Validate module arity if applicable
            module = self._extract_module_to_check(value)
            if module is not None and key in predicate_arities:
                self._validate_module_arity(key, module, predicate_arities[key])

            # Wrap the predicate value
            wrapped_predicates[key] = self._wrap_predicate_value(key, value)

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
            for arg in getattr(e, "args", []):
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
        ctx: EvaluationContext,
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
        ctx: EvaluationContext,
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

        # Create cache key - handle tensors and dicts with id()
        def _make_hashable(a: Any) -> Any:
            if isinstance(a, torch.Tensor):
                return id(a)
            elif isinstance(a, dict):
                # For dicts, use the id to avoid unhashable type error
                return ("dict", id(a))
            else:
                return a

        cache_key = (id(func), tuple(_make_hashable(a) for a in call_args))

        # Call with all arguments
        result = ctx.get_or_compute(cache_key, lambda: predicate(*call_args))
        return cast(torch.Tensor, result)

    def _evaluate_module_predicate(
        self,
        unused_app: PredicateApplication,
        inputs: Dict[str, torch.Tensor],
        predicate: Predicate,
        free_vars: List[VariableSymbol],
        constants: List[Any],
        ctx: EvaluationContext,
    ) -> torch.Tensor:
        """Evaluate an nn.Module predicate.

        Passes variable inputs to the module, then uses constants as output
        indices to select specific channels.

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
            cache_key, lambda: predicate(*var_inputs)
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
        self, const: sp.Basic, inputs: Dict[str, torch.Tensor]
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
        inputs: Dict[str, torch.Tensor],
        predicates: Dict[str, Predicate],
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
        inputs: Dict[str, torch.Tensor],
        predicates: Dict[str, Predicate],
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
        inputs: Dict[str, torch.Tensor],
        predicates: Dict[str, Predicate],
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
        inputs: Dict[str, torch.Tensor],
        predicates: Dict[str, Predicate],
        ctx: EvaluationContext,
    ) -> torch.Tensor:
        """Evaluate Not expression."""
        return self.negation(
            self._evaluate_expression(expr.args[0], inputs, predicates, ctx)
        )

    def _evaluate_implies(
        self,
        expr: sp.Implies,
        inputs: Dict[str, torch.Tensor],
        predicates: Dict[str, Predicate],
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
        inputs: Dict[str, torch.Tensor],
        predicates: Dict[str, Predicate],
        ctx: EvaluationContext,
    ) -> torch.Tensor:
        """Evaluate Equivalent expression."""
        return self.equivalence(
            self._evaluate_expression(expr.args[0], inputs, predicates, ctx),
            self._evaluate_expression(expr.args[1], inputs, predicates, ctx),
        )
