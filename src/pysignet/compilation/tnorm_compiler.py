"""T-norm based logic compilation strategy."""

from typing import Callable, Dict, Union, Set, Optional

import sympy as sp
import torch

from .base import LogicCompiler
from ..predicate import Predicate
from ..tnorms import TNorm, RProductTNorm
from ..context import EvaluationContext
from ..multiclass import PredicateApplication
from ..logic import extract_variables, Binding, ground


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

        # Validate predicate usage consistency (nullary vs unary)
        self._validate_predicate_usage(expr)

        # Extract free variables for FOL support
        free_vars = extract_variables(expr)

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
            # Create evaluation context for this evaluation
            # Context manages caching to avoid redundant forward passes
            ctx = EvaluationContext()

            # If no free variables, evaluate directly (propositional logic)
            if len(free_vars) == 0:
                return self._evaluate_expression(expr, inputs, wrapped_predicates, ctx)

            # FOL: Universal quantification over batch dimensions
            # ∀X∀Y...: φ(X, Y, ...) means conjunction over all batch indices
            # Get batch size from inputs
            if isinstance(inputs, dict):
                sample_input = next(iter(inputs.values()))
            else:
                sample_input = inputs
            batch_size = sample_input.shape[0]

            # For each batch index, ground all variables to that index
            # and evaluate the grounded expression
            results = []
            for batch_idx in range(batch_size):
                # Create binding: all variables map to this batch index
                binding = Binding({var: batch_idx for var in free_vars})

                # Ground expression with this binding
                grounded_expr = ground(expr, binding)

                # Evaluate grounded expression
                result = self._evaluate_expression(
                    grounded_expr, inputs, wrapped_predicates, ctx
                )
                results.append(result)

            # Stack results: (batch_size, batch_size) tensor
            # results[i][j] = satisfaction when variables are bound to index i
            #                 evaluated on batch element j
            stacked = torch.stack(results, dim=0)

            # Universal quantification: take conjunction along batch dimension
            # For each batch element j, we want: ∀i: φ(i) evaluated at j
            # This is conjunction over dimension 0
            result = stacked[0]  # Start with first batch index
            for i in range(1, batch_size):
                result = self.tnorm.conjunction(result, stacked[i])

            return result

        return compiled_logic

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

        Ensures that each predicate appears in the expression with the same
        arity (number of arguments) throughout. For example:
        - Nullary (arity 0): P, Q (used without arguments)
        - Unary (arity 1): Digit(0), Digit(1) (used with one argument)
        - Binary (arity 2): Rel(0, 1) (used with two arguments)

        Args:
            expr: SymPy expression to validate

        Raises:
            ValueError: If any predicate is used with inconsistent arity
        """
        from typing import Dict, Optional
        predicate_arities: Dict[str, int] = {}

        def collect_usage(e: sp.Basic) -> None:
            """Recursively collect predicate usage and check arity consistency."""
            if isinstance(e, sp.Symbol):
                # Used as plain symbol (nullary - arity 0)
                # But skip if it's inside a PredicateApplication's name
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
                # Used with arguments (n-ary - arity > 0)
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

    def _evaluate_expression(
        self,
        expr: sp.Basic,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        predicates: Dict[str, Predicate],
        ctx: EvaluationContext
    ) -> torch.Tensor:
        """Recursively evaluate SymPy expression using t-norms.

        Args:
            expr: SymPy expression to evaluate
            inputs: Single tensor or dict of tensors
            predicates: Dict of predicates
            ctx: Evaluation context for caching

        Returns:
            Tensor of shape (batch_size,) with values in [0, 1]
        """
        # Base case: PredicateApplication (multi-class predicate)
        if isinstance(expr, PredicateApplication):
            pred_name = expr.predicate_name
            func = predicates[pred_name].func

            # Create cache key based on function and inputs identity
            cache_key = (id(func), id(inputs))

            # Get or compute full output (cached to avoid redundant forward passes)
            full_output = ctx.get_or_compute(cache_key, lambda: func(inputs))

            # Extract the specific output index
            index = expr.application_args[0]
            if full_output.dim() == 1:
                # Single output case - just return the tensor
                return full_output
            else:
                # Multi-output case - extract the indexed column
                return full_output[:, index]

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
            result = self._evaluate_expression(expr.args[0], inputs, predicates, ctx)
            for arg in expr.args[1:]:
                result = self.tnorm.conjunction(
                    result,
                    self._evaluate_expression(arg, inputs, predicates, ctx)
                )
            return result

        if isinstance(expr, sp.Or):
            # Disjoin all arguments
            result = self._evaluate_expression(expr.args[0], inputs, predicates, ctx)
            for arg in expr.args[1:]:
                result = self.tnorm.disjunction(
                    result,
                    self._evaluate_expression(arg, inputs, predicates, ctx)
                )
            return result

        if isinstance(expr, sp.Not):
            return self.tnorm.negation(
                self._evaluate_expression(expr.args[0], inputs, predicates, ctx)
            )

        if isinstance(expr, sp.Implies):
            return self.tnorm.implication(
                self._evaluate_expression(expr.args[0], inputs, predicates, ctx),
                self._evaluate_expression(expr.args[1], inputs, predicates, ctx)
            )

        if isinstance(expr, sp.Equivalent):
            return self.tnorm.equivalence(
                self._evaluate_expression(expr.args[0], inputs, predicates, ctx),
                self._evaluate_expression(expr.args[1], inputs, predicates, ctx)
            )

        raise ValueError(f"Unsupported expression type: {type(expr)}")
