"""Base class for logic compilation strategies."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Union, Set
import warnings

import sympy as sp
import torch
import torch.nn as nn

from ..predicate import Predicate
from ..multiclass import PredicateApplication
from ..logic import extract_variables, extract_variables_from_application
from ..logic.quantifier import Quantifier
from ..logic.expansion import expand_quantifier
from .arity import validate_predicate_arity
from .module_utils import (
    infer_module_arity,
    wrap_module_as_predicate
)


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
            predicates: Dict[str, Predicate],
    ) -> Callable[[Union[torch.Tensor, Dict[str, torch.Tensor]]], torch.Tensor]:
        """Compile a logic expression into a differentiable callable.

        Args:
            expr: SymPy logic expression (e.g., sp.And(P, sp.Or(Q, sp.Not(R))))
            predicates: Dict mapping predicate names to Predicate objects

        Returns:
            Callable that takes inputs and returns satisfaction tensor of
            shape (batch_size,) with values in [0, 1].

        Raises:
            ValueError: If symbols in expr have no corresponding predicates
        """
        pass

    def _wrap_and_validate_predicates(
        self,
        expr: sp.Basic,
        predicates: Dict[str, Predicate]
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
            module_to_check = None
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
    ) -> tuple:
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
        from ..logic.variable import VariableSymbol

        # Use dict to preserve order while deduplicating
        free_vars_dict: Dict[str, 'VariableSymbol'] = {}
        constants: list = []

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
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        predicates: Dict[str, Predicate],
        ctx
    ) -> torch.Tensor:
        """Evaluate PredicateApplication with mixed variable/constant arguments.

        This method is completely compiler-agnostic and handles:
        - Extracting free variables and constants
        - Routing to appropriate handler based on variable count
        - Backwards compatibility for old API (Digit(0) with tensor)
        - Output channel selection for constants
        - Caching via EvaluationContext

        Args:
            app: PredicateApplication to evaluate
            inputs: Single tensor or dict of tensors
            predicates: Dict of predicates
            ctx: EvaluationContext for caching

        Returns:
            Tensor of shape (batch_size,) with values in [0, 1]
        """
        from ..logic.variable import VariableSymbol

        pred_name = app.predicate_name
        predicate = predicates[pred_name]
        func = predicate.func

        # Parse application into free vars and constants
        free_vars, constants = self._parse_predicate_application(app)

        # Determine how to handle constants:
        # - nn.Module: Use constants as output indices (old behavior for multiclass)
        # - Regular callables: Pass constants as arguments (new FOL behavior)
        is_module = isinstance(func, torch.nn.Module)

        if not is_module and len(constants) > 0:
            # FOL semantics: Pass ALL arguments (free vars + constants) to callable
            # Build args in the order they appear in application_args
            from ..logic.variable import VariableSymbol

            call_args = []
            for arg in app.application_args:
                if isinstance(arg, VariableSymbol):
                    # Free variable - get from inputs
                    var_name = str(arg)
                    if isinstance(inputs, dict):
                        if var_name not in inputs:
                            raise ValueError(
                                f"Missing input for variable '{var_name}'. "
                                f"Expected key in input dict."
                            )
                        call_args.append(inputs[var_name])
                    else:
                        # Single tensor input (only valid if single free var)
                        if len(free_vars) > 1:
                            var_names = ", ".join(str(v) for v in free_vars)
                            raise ValueError(
                                f"Predicate '{pred_name}' has multiple free variables "
                                f"({var_names}) but received non-dict input."
                            )
                        call_args.append(inputs)
                else:
                    # Constant - pass as-is
                    call_args.append(arg)

            # Call with all arguments (using predicate for clamping)
            cache_key = (id(func), tuple(
                id(a) if isinstance(a, torch.Tensor) else a for a in call_args
            ))
            result = ctx.get_or_compute(cache_key, lambda: predicate(*call_args))
            return result

        # Handle nn.Module or no-constants case (old behavior)
        # Route based on number of free variables
        if len(free_vars) == 0:
            # No free variables - either nullary or constant-only predicate

            # If non-module callable with constants, pass constants as arguments
            if not is_module and len(constants) > 0:
                cache_key = (id(func), tuple(constants))
                result = ctx.get_or_compute(
                    cache_key,
                    lambda: predicate(*constants)
                )
                return result

            # True nullary predicate (no variables, no constants)
            cache_key = (id(func), "nullary")
            result = ctx.get_or_compute(cache_key, lambda: predicate())

            # Convert to tensor if needed
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result)

            # Handle constants as output indices if present (nn.Module case)
            if len(constants) > 0 and result.dim() > 0:
                for const in constants:
                    if result.dim() == 0:
                        break
                    result = (result[const] if result.dim() == 1
                             else result[:, const])

            return result

        elif len(free_vars) == 1:
            # Single free variable
            var = free_vars[0]
            var_name = str(var)

            # Get input for this variable
            if isinstance(inputs, dict):
                if var_name not in inputs:
                    raise ValueError(
                        f"Missing input for variable '{var_name}'. "
                        f"Expected key in input dict."
                    )
                var_input = inputs[var_name]
            else:
                # Single tensor input - use directly (convenience)
                var_input = inputs

            # Call predicate with single argument (using predicate for clamping)
            cache_key = (id(func), id(var_input))
            full_output = ctx.get_or_compute(
                cache_key,
                lambda: predicate(var_input)
            )

        else:
            # Multiple free variables
            # Must have dict inputs
            if not isinstance(inputs, dict):
                var_names = ", ".join(str(v) for v in free_vars)
                raise ValueError(
                    f"Predicate '{pred_name}' has multiple free variables "
                    f"({var_names}) but received non-dict input. "
                    f"Expected dict with keys for each variable."
                )

            # Extract inputs for each variable (in order of appearance)
            var_inputs = []
            for var in free_vars:
                var_name = str(var)
                if var_name not in inputs:
                    raise ValueError(
                        f"Missing input for variable '{var_name}'. "
                        f"Expected key in input dict."
                    )
                var_inputs.append(inputs[var_name])

            # Create cache key from function and input identities
            cache_key = (id(func), tuple(id(inp) for inp in var_inputs))

            # Call predicate with multiple arguments (using predicate for clamping)
            full_output = ctx.get_or_compute(
                cache_key,
                lambda: predicate(*var_inputs)
            )

        # Handle constants as output indices (nn.Module or no FOL constants)
        if len(constants) > 0:
            # Multi-output predicate - select specific output channel(s)
            result = full_output
            for const in constants:
                if result.dim() == 1:
                    # Already a batch of scalars, can't index further
                    break
                elif result.dim() == 2:
                    # Shape: (batch, num_outputs)
                    # Select column for this constant
                    result = result[:, const]
                else:
                    # Higher dimensional - index first non-batch dimension
                    result = result.select(dim=1, index=const)
            return result
        else:
            # No constants - return full output
            # Squeeze if output is (batch, 1) instead of (batch,)
            if full_output.dim() == 2 and full_output.shape[1] == 1:
                return full_output.squeeze(-1)
            return full_output

    def _evaluate_symbol(
        self,
        symbol: sp.Symbol,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        predicates: Dict[str, Predicate],
        ctx
    ) -> torch.Tensor:
        """Evaluate nullary predicate symbol.

        Args:
            symbol: SymPy symbol representing nullary predicate
            inputs: Single tensor or dict of tensors
            predicates: Dict of predicates
            ctx: EvaluationContext for caching

        Returns:
            Tensor of shape (batch_size,) with values in [0, 1]
        """
        pred_name = str(symbol)
        predicate = predicates[pred_name]
        return predicate(inputs)

    def _evaluate_boolean_constant(
        self,
        const: sp.Basic,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Evaluate boolean constant (sp.true or sp.false).

        Args:
            const: sp.true or sp.false
            inputs: Single tensor or dict of tensors (to determine batch size)

        Returns:
            Tensor of ones (true) or zeros (false) with shape (batch_size,)
        """
        # Determine batch size from inputs
        if isinstance(inputs, dict):
            sample_input = next(iter(inputs.values()))
        else:
            sample_input = inputs
        batch_size = sample_input.shape[0]

        if const == sp.true:
            return torch.ones(batch_size, device=sample_input.device)
        elif const == sp.false:
            return torch.zeros(batch_size, device=sample_input.device)
        else:
            raise ValueError(f"Expected sp.true or sp.false, got {const}")
