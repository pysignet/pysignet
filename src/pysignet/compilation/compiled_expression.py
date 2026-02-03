"""Compiled logical expression with variable bindings.

This module provides the CompiledExpression class, which represents a compiled
logical expression that can be evaluated with variable bindings and supports
partial binding to create new expressions with fewer free variables.

CompiledExpression performs PURE EVALUATION only - it always returns per-batch
results with shape (batch_size,). Batch reduction (quantification) and loss
computation are handled by LogicLoss, which uses BatchHandlerMixin.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Set, List, Optional, TYPE_CHECKING

import sympy as sp
import torch
import torch.nn as nn

from pysignet.predicate import Predicate

if TYPE_CHECKING:
    from pysignet.compilation.base import LogicCompiler


class CompiledExpression:
    """Represents a compiled logical expression with variable bindings.

    A compiled expression is the result of compiling a logical expression
    (e.g., And(P(X), Q(Y))) into a differentiable PyTorch computation graph.
    It supports:

    - Evaluation with variable bindings: compiled(X=x, Y=y) -> Tensor
    - Partial binding: compiled.partial(X=x) -> CompiledExpression
    - Introspection: compiled.free_variables -> Set[str]
    - Compiler access: compiled.compiler -> LogicCompiler

    CompiledExpression performs PURE EVALUATION - it always returns per-batch
    results with shape (batch_size,). No batch reduction is applied. For
    quantification over batch dimensions, wrap with LogicLoss.

    Partial binding creates a NEW computation graph with fewer free variables,
    not just a temporary store of bindings. This enables future optimizations
    like constant propagation, dead code elimination, and graph rewriting.

    IMPORTANT: All inputs must be provided as keyword arguments
    (e.g., X=tensor). Positional arguments are not supported. This ensures
    clear, unambiguous variable binding.

    Args:
        compiled_logic: Callable that evaluates the expression given variable
                       bindings as a dict mapping variable names to tensors
        free_variables: Set of variable names that must be bound for evaluation
        predicates: Dict mapping predicate names to Predicate objects
        partial_bindings: Dict of variables already bound in this expression
                         (used internally for partial binding)
        compiler: Optional LogicCompiler that produced this expression
        expr: Optional original SymPy expression (for repr/debugging)

    Example:
        >>> # Create compiled expression
        >>> compiled = compiler.compile(expr, predicates)
        >>>
        >>> # Evaluate with full bindings - returns (batch_size,)
        >>> result = compiled(X=x, Y=y)  # shape: (batch_size,)
        >>>
        >>> # Partial binding
        >>> partial = compiled.partial(X=x)
        >>> result = partial(Y=y)  # shape: (batch_size,)
        >>>
        >>> # Introspection
        >>> print(compiled.free_variables)  # {'X', 'Y'}
        >>> print(partial.free_variables)   # {'Y'}
        >>> print(compiled.compiler)        # TNormCompiler(...)
    """

    def __init__(
        self,
        compiled_logic: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
        free_variables: Set[str],
        predicates: Dict[str, Predicate],
        partial_bindings: Optional[Dict[str, torch.Tensor]] = None,
        compiler: Optional[LogicCompiler] = None,
        expr: Optional[sp.Basic] = None,
    ) -> None:
        """Initialize CompiledExpression.

        Args:
            compiled_logic: Callable that evaluates expression with inputs
                           as a dict mapping variable names to tensors.
            free_variables: Set of variable names in the original
                expression
            predicates: Dict of predicate objects
            partial_bindings: Dict of variables already bound
                (default: empty)
            compiler: Optional LogicCompiler that produced this
                expression
            expr: Optional original SymPy expression (for repr/debugging)
        """
        self._compiled_logic = compiled_logic
        self._free_variables = free_variables
        self._predicates = predicates
        self._partial_bindings = partial_bindings or {}
        self._compiler = compiler
        self._expr = expr

    @property
    def compiler(self) -> Optional[LogicCompiler]:
        """Return the LogicCompiler that produced this expression.

        Returns:
            LogicCompiler instance or None
        """
        return self._compiler

    @property
    def expr(self) -> Optional[sp.Basic]:
        """Return the original SymPy expression.

        Returns:
            SymPy expression or None
        """
        return self._expr

    def __repr__(self) -> str:
        """Return string representation of compiled expression.

        Returns:
            String showing expression, free variables, and predicates.
        """
        parts = ["CompiledExpression("]

        # Show expression if available
        if self._expr is not None:
            parts.append(f"  expr={self._expr},")
        else:
            parts.append("  expr=<compiled>,")

        # Show free variables (remaining unbound)
        free = self.free_variables
        parts.append(f"  free_variables={{{', '.join(sorted(free))}}},")

        # Show predicates
        pred_names = sorted(self._predicates.keys())
        parts.append(f"  predicates={{{', '.join(pred_names)}}},")

        # Show partial bindings if any
        if self._partial_bindings:
            bound = sorted(self._partial_bindings.keys())
            parts.append(f"  bound={{{', '.join(bound)}}},")

        # Show compiler type if available
        if self._compiler is not None:
            compiler_name = type(self._compiler).__name__
            parts.append(f"  compiler={compiler_name}")
        else:
            parts.append("  compiler=None")

        parts.append(")")
        return "\n".join(parts)

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Pretty print for IPython/Jupyter.

        Args:
            p: IPython pretty printer
            cycle: Whether we're in a cycle (unused)
        """
        del cycle  # unused
        p.text(repr(self))

    @property
    def free_variables(self) -> Set[str]:
        """Return set of variables that still need to be bound.

        Returns:
            Set of variable names not yet bound

        Example:
            >>> compiled.free_variables  # {'X', 'Y'}
            >>> partial = compiled.partial(X=x)
            >>> partial.free_variables   # {'Y'}
        """
        bound = set(self._partial_bindings.keys())
        return self._free_variables - bound

    def __call__(
        self,
        *,  # Force keyword-only arguments
        return_boolean: bool = False,
        **variable_bindings: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate compiled expression with variable bindings.

        All free variables must be bound (either via partial bindings stored
        in this expression, or provided here as keyword arguments).

        This method performs PURE EVALUATION - it always returns per-batch
        results with shape (batch_size,). No batch reduction is applied.
        For quantification over batch dimensions, wrap with LogicLoss.

        Args:
            return_boolean: If True, return boolean satisfaction using hard
                decisions (threshold at 0.5 for binary, argmax for multiclass).
                Default is False (soft satisfaction).
            **variable_bindings: Variable bindings as keyword arguments
                (e.g., X=x_tensor, Y=y_tensor)

        Returns:
            If return_boolean is False: Satisfaction tensor of shape
                (batch_size,) with values in [0, 1].
            If return_boolean is True: Boolean tensor of shape (batch_size,)
                indicating whether the formula is satisfied.

        Raises:
            ValueError: If any free variable is not bound
            ValueError: If return_boolean=True but expression is not available

        Example:
            >>> # Soft satisfaction (default)
            >>> result = compiled(X=x, Y=y)  # shape: (batch_size,)
            >>>
            >>> # Boolean satisfaction
            >>> result = compiled(X=x, Y=y, return_boolean=True)  # bool tensor
        """
        # Merge partial bindings with new bindings
        all_bindings: Dict[str, torch.Tensor] = dict(self._partial_bindings)
        all_bindings.update(variable_bindings)

        # Validate all free variables are bound
        provided = set(all_bindings.keys())
        missing = self._free_variables - provided
        if missing:
            raise ValueError(
                f"Missing input bindings for variables: "
                f"{sorted(missing)}. Expected bindings for: "
                f"{sorted(self._free_variables)}"
            )

        if return_boolean:
            return self._evaluate_boolean_satisfaction(all_bindings)

        # Evaluate the expression and return per-batch results
        # No batch reduction - LogicLoss handles quantification
        return self._compiled_logic(all_bindings)

    def _evaluate_boolean_satisfaction(
        self, bindings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Evaluate expression with boolean (hard) decisions.

        Uses ConsistencyChecker with boolean-converted predicates.

        Args:
            bindings: Dict mapping variable names to tensors

        Returns:
            Boolean tensor of shape (batch_size,)
        """
        # pylint: disable=import-outside-toplevel
        from pysignet.consistency import ConsistencyChecker

        if self._expr is None:
            raise ValueError(
                "Boolean evaluation requires the original expression. "
                "This CompiledExpression was created without storing the "
                "expression."
            )

        # Create boolean predicates that threshold soft outputs
        # Pass bindings so predicates can look up variable inputs
        boolean_predicates = self._create_boolean_predicates(bindings)

        # Use ConsistencyChecker for boolean evaluation
        # Pass bindings keyed by variable names
        checker = ConsistencyChecker(self._expr, boolean_predicates)
        return checker(bindings)

    def _create_boolean_predicates(
        self, bindings: Dict[str, torch.Tensor]
    ) -> Dict[str, Callable[[Any], torch.Tensor]]:
        """Create boolean predicate functions from soft predicates.

        Converts soft predicates to boolean by:
        - Binary (shape (batch,)): threshold at 0.5
        - Multiclass (shape (batch, n)): argmax == class_index

        Args:
            bindings: Variable bindings (used to get input tensors)

        Returns:
            Dict mapping predicate names to boolean predicate functions
        """
        boolean_predicates: Dict[str, Callable[[Any], torch.Tensor]] = {}

        for name, predicate in self._predicates.items():
            boolean_predicates[name] = self._make_boolean_predicate(
                predicate, bindings
            )

        return boolean_predicates

    def _make_boolean_predicate(
        self, predicate: Predicate, bindings: Dict[str, torch.Tensor]
    ) -> Callable[[Any], torch.Tensor]:
        """Create a boolean predicate function from a soft predicate.

        Args:
            predicate: The soft predicate to convert
            bindings: Variable bindings

        Returns:
            Callable that returns boolean tensor
        """
        def boolean_pred(*args: Any) -> torch.Tensor:
            # Extract class index if present (for multiclass predicates)
            class_idx: Optional[int] = None
            input_tensor: Optional[torch.Tensor] = None

            for arg in args:
                if isinstance(arg, int):
                    class_idx = arg
                elif isinstance(arg, torch.Tensor):
                    input_tensor = arg

            # If no tensor arg provided by ConsistencyChecker, use bindings
            if input_tensor is None:
                # Use first binding as input (common case: single variable)
                input_tensor = next(iter(bindings.values()))

            # Call predicate to get soft output
            soft_output = predicate(input_tensor)

            # Convert to boolean based on output shape
            if soft_output.dim() >= 2 and soft_output.shape[-1] > 1:
                # Multiclass output (batch, num_classes)
                if class_idx is not None:
                    # Return True if argmax == class_idx
                    return soft_output.argmax(dim=-1) == class_idx
                else:
                    # No class specified - threshold max confidence
                    return soft_output.max(dim=-1).values > 0.5
            else:
                # Binary output - threshold at 0.5
                if soft_output.dim() >= 2:
                    soft_output = soft_output.squeeze(-1)
                return soft_output > 0.5

        return boolean_pred

    def partial(self, **variable_bindings: torch.Tensor) -> CompiledExpression:
        """Create new CompiledExpression with some variables bound.

        This creates a NEW computation graph where the bound variables
        are substituted with their values. The returned
        CompiledExpression has fewer free variables.

        Future optimizations will use this to:
        - Constant-fold operations on bound variables
        - Apply common subexpression elimination
        - Prune unreachable branches
        - Rewrite the computation graph for efficiency

        Args:
            **variable_bindings: Variable bindings to fix (X=x, Y=y)

        Returns:
            New CompiledExpression with fewer free variables

        Raises:
            ValueError: If no bindings provided
            ValueError: If variable is already bound
            ValueError: If variable doesn't exist in expression

        Example:
            >>> # Bind X, leaving Y for later
            >>> partial = compiled.partial(X=x)
            >>> print(partial.free_variables)  # {'Y'}
            >>>
            >>> # Chain partial bindings
            >>> result = compiled.partial(X=x).partial(Y=y)(Z=z)
            >>>
            >>> # Bind multiple at once
            >>> partial = compiled.partial(X=x, Y=y)
        """
        if not variable_bindings:
            raise ValueError("Must provide at least one variable binding")

        # Check for duplicate bindings
        duplicates = set(variable_bindings.keys()) & set(
            self._partial_bindings.keys()
        )
        if duplicates:
            raise ValueError(
                f"Variable(s) already bound: {sorted(duplicates)}. "
                f"Cannot bind the same variable twice."
            )

        # Check for non-existent variables
        unknown = set(variable_bindings.keys()) - self._free_variables
        if unknown:
            raise ValueError(
                f"Variable(s) not in expression: {sorted(unknown)}. "
                f"Free variables are: "
                f"{sorted(self._free_variables)}"
            )

        # Create new CompiledExpression with merged bindings
        new_bindings = dict(self._partial_bindings)
        new_bindings.update(variable_bindings)

        return CompiledExpression(
            compiled_logic=self._compiled_logic,
            free_variables=self._free_variables,
            predicates=self._predicates,
            partial_bindings=new_bindings,
            compiler=self._compiler,
            expr=self._expr,
        )

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters from model-based predicates.

        Returns:
            List of torch.nn.Parameter objects from all model-based
            predicates

        Example:
            >>> params = compiled.get_trainable_parameters()
            >>> optimizer = torch.optim.Adam(params, lr=0.001)
        """
        params: List[nn.Parameter] = []
        for pred in self._predicates.values():
            if pred.is_model and hasattr(pred.func, "parameters"):
                params.extend(pred.func.parameters())
        return params
