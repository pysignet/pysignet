"""CompiledExpression - represents a compiled logical expression with variable bindings.

This module provides the CompiledExpression class, which represents a compiled
logical expression that can be evaluated with variable bindings and supports
partial binding to create new expressions with fewer free variables.

CompiledExpression performs PURE EVALUATION only - it always returns per-batch
results with shape (batch_size,). Batch reduction (quantification) and loss
computation are handled by LogicLoss, which uses BatchHandlerMixin.
"""

from typing import Callable, Dict, Set, List, Optional, TYPE_CHECKING

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

    IMPORTANT: All inputs must be provided as keyword arguments (e.g., X=tensor).
    Positional arguments are not supported. This ensures clear, unambiguous
    variable binding.

    Args:
        compiled_logic: Callable that evaluates the expression given variable
                       bindings as a dict mapping variable names to tensors
        free_variables: Set of variable names that must be bound for evaluation
        predicates: Dict mapping predicate names to Predicate objects
        partial_bindings: Dict of variables already bound in this expression
                         (used internally for partial binding)
        compiler: Optional LogicCompiler that produced this expression

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
        compiled_logic: Callable[
            [Dict[str, torch.Tensor]], torch.Tensor
        ],
        free_variables: Set[str],
        predicates: Dict[str, Predicate],
        partial_bindings: Optional[Dict[str, torch.Tensor]] = None,
        compiler: Optional['LogicCompiler'] = None
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
        """
        self._compiled_logic = compiled_logic
        self._free_variables = free_variables
        self._predicates = predicates
        self._partial_bindings = partial_bindings or {}
        self._compiler = compiler

    @property
    def compiler(self) -> Optional['LogicCompiler']:
        """Return the LogicCompiler that produced this expression.

        Returns:
            LogicCompiler instance or None
        """
        return self._compiler

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
        inputs: Optional[Dict[str, torch.Tensor]] = None,
        **variable_bindings: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate compiled expression with variable bindings.

        All free variables must be bound (either via partial bindings stored
        in this expression, or provided here).

        This method performs PURE EVALUATION - it always returns per-batch
        results with shape (batch_size,). No batch reduction is applied.
        For quantification over batch dimensions, wrap with LogicLoss.

        IMPORTANT: All inputs must be provided as keyword arguments
        (e.g., compiled(X=tensor)). Positional arguments are not supported.

        Args:
            inputs: Optional dict mapping variable names to tensors.
                   Prefer using keyword arguments (**variable_bindings)
                   instead.
            **variable_bindings: Variable bindings (X=x, Y=y, etc.)

        Returns:
            Satisfaction tensor of shape (batch_size,) with values in
            [0, 1]. Always returns per-batch results, never reduced
            to scalar.

        Raises:
            ValueError: If any free variable is not bound
            ValueError: If both inputs and variable_bindings provided

        Example:
            >>> # Keyword argument interface (required)
            >>> result = compiled(X=x, Y=y)  # shape: (batch_size,)
            >>>
            >>> # With partial bindings
            >>> partial = compiled.partial(X=x)
            >>> result = partial(Y=y)  # shape: (batch_size,)
        """
        # Merge partial bindings with new bindings
        all_bindings: Dict[str, torch.Tensor] = dict(
            self._partial_bindings
        )

        # Add new bindings from kwargs
        if variable_bindings:
            if inputs is not None:
                raise ValueError(
                    "Cannot provide both 'inputs' dict and keyword "
                    "arguments. Use one or the other."
                )
            all_bindings.update(variable_bindings)
        elif inputs is not None:
            if not isinstance(inputs, dict):
                raise ValueError(
                    "Inputs must be a dict mapping variable names "
                    "to tensors. Use keyword arguments like "
                    "compiled(X=tensor) instead of positional "
                    "arguments like compiled(tensor)."
                )
            all_bindings.update(inputs)

        # Validate all free variables are bound
        provided = set(all_bindings.keys())
        missing = self._free_variables - provided
        if missing:
            raise ValueError(
                f"Missing input bindings for variables: "
                f"{sorted(missing)}. Expected bindings for: "
                f"{sorted(self._free_variables)}"
            )

        # Evaluate the expression and return per-batch results
        # No batch reduction - LogicLoss handles quantification
        return self._compiled_logic(all_bindings)

    def partial(
        self, **variable_bindings: torch.Tensor
    ) -> 'CompiledExpression':
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
            raise ValueError(
                "Must provide at least one variable binding"
            )

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
        unknown = (
            set(variable_bindings.keys()) - self._free_variables
        )
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
            compiler=self._compiler
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
            if pred.is_model and hasattr(pred.func, 'parameters'):
                params.extend(pred.func.parameters())
        return params
