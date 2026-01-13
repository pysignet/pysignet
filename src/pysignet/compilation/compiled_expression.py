"""CompiledExpression - represents a compiled logical expression with variable bindings.

This module provides the CompiledExpression class, which represents a compiled
logical expression that can be evaluated with variable bindings and supports
partial binding to create new expressions with fewer free variables.
"""

from typing import Callable, Dict, Set, List, Union, Optional

import torch
import torch.nn as nn

from ..predicate import Predicate


class CompiledExpression:
    """Represents a compiled logical expression with variable bindings.

    A compiled expression is the result of compiling a logical expression
    (e.g., And(P(X), Q(Y))) into a differentiable PyTorch computation graph.
    It supports:

    - Evaluation with variable bindings: compiled(X=x, Y=y) → Tensor
    - Partial binding: compiled.partial(X=x) → CompiledExpression
    - Introspection: compiled.free_variables → Set[str]

    Partial binding creates a NEW computation graph with fewer free variables,
    not just a temporary store of bindings. This enables future optimizations
    like constant propagation, dead code elimination, and graph rewriting.

    Args:
        compiled_logic: Callable that evaluates the expression given variable
                       bindings as a dict
        free_variables: Set of variable names that must be bound for evaluation
        predicates: Dict mapping predicate names to Predicate objects
        partial_bindings: Dict of variables already bound in this expression
                         (used internally for partial binding)

    Example:
        >>> # Create compiled expression
        >>> compiled = compiler.compile(expr, predicates)
        >>>
        >>> # Evaluate with full bindings
        >>> result = compiled(X=x, Y=y)
        >>>
        >>> # Partial binding
        >>> partial = compiled.partial(X=x)
        >>> result = partial(Y=y)
        >>>
        >>> # Introspection
        >>> print(compiled.free_variables)  # {'X', 'Y'}
        >>> print(partial.free_variables)   # {'Y'}
    """

    def __init__(
        self,
        compiled_logic: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
        free_variables: Set[str],
        predicates: Dict[str, Predicate],
        partial_bindings: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """Initialize CompiledExpression.

        Args:
            compiled_logic: Callable that evaluates expression with bindings dict
            free_variables: Set of variable names in the original expression
            predicates: Dict of predicate objects
            partial_bindings: Dict of variables already bound (default: empty)
        """
        self._compiled_logic = compiled_logic
        self._free_variables = free_variables
        self._predicates = predicates
        self._partial_bindings = partial_bindings or {}

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
        inputs: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        **variable_bindings: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate compiled expression with variable bindings.

        All free variables must be bound (either via partial bindings stored
        in this expression, or provided here).

        Args:
            inputs: For backward compatibility - single tensor or dict.
                   Use variable_bindings kwargs instead (FOL interface).
            **variable_bindings: Variable bindings (X=x, Y=y, etc.)

        Returns:
            Satisfaction tensor of shape (batch_size,) with values in [0, 1]

        Raises:
            ValueError: If any free variable is not bound
            ValueError: If both inputs and variable_bindings provided

        Example:
            >>> # FOL interface (preferred)
            >>> result = compiled(X=x, Y=y)
            >>>
            >>> # With partial bindings
            >>> partial = compiled.partial(X=x)
            >>> result = partial(Y=y)
        """
        # Merge partial bindings with new bindings
        if self._partial_bindings or variable_bindings:
            # Start with partial bindings
            all_bindings = dict(self._partial_bindings)

            # Add new bindings
            if variable_bindings:
                if inputs is not None:
                    raise ValueError(
                        "Cannot provide both positional 'inputs' and variable "
                        "bindings (**kwargs). Use one or the other."
                    )
                all_bindings.update(variable_bindings)

            inputs = all_bindings if all_bindings else (inputs or {})
        # Handle no-input case (constant-only predicates)
        elif inputs is None:
            inputs = {}

        # Validate all free variables are bound (only when using FOL interface)
        if self._free_variables and isinstance(inputs, dict):
            provided = set(inputs.keys())
            missing = self._free_variables - provided
            if missing:
                raise ValueError(
                    f"Missing input bindings for variables: {sorted(missing)}. "
                    f"Expected bindings for: {sorted(self._free_variables)}"
                )

        return self._compiled_logic(inputs)

    def partial(self, **variable_bindings: torch.Tensor) -> 'CompiledExpression':
        """Create new CompiledExpression with some variables bound.

        This creates a NEW computation graph where the bound variables are
        substituted with their values. The returned CompiledExpression has
        fewer free variables.

        Future optimizations will use this to:
        - Constant-fold operations on bound variables
        - Apply common subexpression elimination
        - Prune unreachable branches
        - Rewrite the computation graph for efficiency

        Args:
            **variable_bindings: Variable bindings to fix (e.g., X=x, Y=y)

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
                f"Free variables are: {sorted(self._free_variables)}"
            )

        # Create new CompiledExpression with merged bindings
        new_bindings = dict(self._partial_bindings)
        new_bindings.update(variable_bindings)

        return CompiledExpression(
            compiled_logic=self._compiled_logic,
            free_variables=self._free_variables,
            predicates=self._predicates,
            partial_bindings=new_bindings
        )

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters from model-based predicates.

        Returns:
            List of torch.nn.Parameter objects from all model-based predicates

        Example:
            >>> params = compiled.get_trainable_parameters()
            >>> optimizer = torch.optim.Adam(params, lr=0.001)
        """
        params: List[nn.Parameter] = []
        for pred in self._predicates.values():
            if pred.is_model and hasattr(pred.func, 'parameters'):
                params.extend(pred.func.parameters())
        return params
