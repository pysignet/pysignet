"""LogicLoss - wraps compiled expressions with loss computation."""

from typing import Callable, Union, List, Dict

import torch

from .compilation.compiled_expression import CompiledExpression


class LogicLoss:
    """Wrapper for CompiledExpression that adds loss computation.

    LogicLoss delegates evaluation and partial binding to an underlying
    CompiledExpression, and adds loss-specific functionality (post-processing,
    reduction, parameter extraction).

    This creates a clean separation of concerns:
    - CompiledExpression: Handles computation graph and variable binding
    - LogicLoss: Adds loss-specific transformations

    Args:
        compiled_expr: CompiledExpression from LogicCompiler.compile()
        post_processing: Default post-processing mode - 'log', 'linear', or
                        callable (default: 'linear')

    Example:
        >>> # Get CompiledExpression from compiler
        >>> compiler = TNormCompiler()
        >>> compiled = compiler.compile(expr, predicates)
        >>>
        >>> # Wrap in LogicLoss for training
        >>> logic_loss = LogicLoss(compiled)
        >>> satisfaction = logic_loss(X=x, Y=y)  # Returns [0, 1]
        >>> loss = logic_loss.loss(X=x, Y=y)     # Returns loss value
        >>>
        >>> # Or use api.compile_logic() which wraps automatically
        >>> from pysignet import compile_logic
        >>> logic_loss = compile_logic(expr, predicates)
    """

    def __init__(
        self,
        compiled_expr: CompiledExpression,
        post_processing: Union[
            str, Callable[[torch.Tensor], torch.Tensor]
        ] = 'linear'
    ) -> None:
        """Initialize LogicLoss.

        Args:
            compiled_expr: CompiledExpression to wrap
            post_processing: Default post-processing ('log', 'linear', callable)
        """
        self._compiled_expr = compiled_expr
        self.default_post_processing = post_processing

    def __call__(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor], None] = None,
        **variable_bindings: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate compiled expression and return satisfaction degrees.

        Delegates to the underlying CompiledExpression.

        Args:
            inputs: For backward compatibility - single tensor or dict.
                   Use variable_bindings kwargs instead (FOL interface).
            **variable_bindings: Variable bindings (e.g., X=x, Y=y)

        Returns:
            Satisfaction tensor of shape (batch_size,) in [0, 1].
            Higher values = better satisfaction.

        Examples:
            >>> # FOL interface (preferred)
            >>> result = logic_loss(X=x, Y=y)
            >>>
            >>> # Backward compatible (single tensor)
            >>> result = logic_loss(x)
            >>>
            >>> # Constant-only predicates (no inputs)
            >>> result = logic_loss()
        """
        return self._compiled_expr(inputs=inputs, **variable_bindings)

    def loss(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor], None] = None,
        reduction: str = 'mean',
        post_processing: Union[
            str, Callable[[torch.Tensor], torch.Tensor], None
        ] = None,
        **variable_bindings: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss based on logical constraint violation.

        Args:
            inputs: For backward compatibility - single tensor or dict.
                   Use variable_bindings kwargs instead (FOL interface).
            reduction: 'mean', 'sum', or 'none' (default: 'mean')
            post_processing: Post-processing mode - 'log', 'linear', callable,
                           or None (uses default from __init__)
            **variable_bindings: Variable bindings (e.g., X=x, Y=y)

        Returns:
            Loss value (lower = better satisfaction)

        Raises:
            ValueError: If invalid post_processing or reduction mode

        Examples:
            >>> # Compute loss with default post-processing (FOL interface)
            >>> loss = logic_loss.loss(X=x, Y=y)
            >>>
            >>> # Backward compatible (single tensor)
            >>> loss = logic_loss.loss(x)
            >>>
            >>> # Custom post-processing
            >>> loss = logic_loss.loss(X=x, Y=y, post_processing='log')
            >>>
            >>> # No reduction (per-example losses)
            >>> losses = logic_loss.loss(X=x, Y=y, reduction='none')
        """
        # Compute satisfaction
        satisfaction = self._compiled_expr(inputs=inputs, **variable_bindings)

        # Determine post-processing mode
        postprocessing_type = (
            post_processing
            if post_processing is not None
            else self.default_post_processing
        )

        # Apply post-processing
        if postprocessing_type == 'log':
            # Negative log with numerical stability
            loss_values = -torch.log(satisfaction + 1e-10)
        elif postprocessing_type == 'linear':
            # Linear: 1 - satisfaction
            loss_values = 1.0 - satisfaction
        elif callable(postprocessing_type):
            # User-provided custom post-processing function
            loss_values = postprocessing_type(satisfaction)
        else:
            raise ValueError(
                f"Unknown post-processing: {postprocessing_type}. "
                f"Expected 'log', 'linear', or a callable."
            )

        # Apply reduction
        if reduction == 'mean':
            result: torch.Tensor = loss_values.mean()
            return result
        if reduction == 'sum':
            result = loss_values.sum()
            return result
        if reduction == 'none':
            return loss_values

        raise ValueError(
            f"Unknown reduction: {reduction}. "
            f"Expected 'mean', 'sum', or 'none'."
        )

    def partial(self, **variable_bindings: torch.Tensor) -> 'LogicLoss':
        """Create partially-bound LogicLoss with some variables fixed.

        Delegates to the underlying CompiledExpression and wraps the result
        in a new LogicLoss.

        This creates a NEW computation graph with fewer free variables, not
        just a temporary store of bindings. Future optimizations will use this
        to enable constant propagation, dead code elimination, etc.

        Args:
            **variable_bindings: Variable bindings to fix (e.g., X=x)

        Returns:
            New LogicLoss with fewer free variables

        Raises:
            ValueError: If no bindings provided
            ValueError: If variable is already bound
            ValueError: If variable doesn't exist in expression

        Examples:
            >>> # Bind X, leaving Y for later
            >>> partial = logic_loss.partial(X=x)
            >>> result = partial(Y=y)
            >>>
            >>> # Chain partial bindings
            >>> result = logic_loss.partial(X=x).partial(Y=y)(Z=z)
            >>>
            >>> # Bind multiple at once
            >>> partial = logic_loss.partial(X=x, Y=y)
            >>> result = partial(Z=z)
        """
        # Delegate to CompiledExpression
        partial_expr = self._compiled_expr.partial(**variable_bindings)

        # Wrap in new LogicLoss
        return LogicLoss(partial_expr, self.default_post_processing)

    @property
    def free_variables(self) -> set:
        """Return set of variables that still need to be bound.

        Delegates to the underlying CompiledExpression.

        Returns:
            Set of variable names not yet bound

        Example:
            >>> print(logic_loss.free_variables)  # {'X', 'Y'}
            >>> partial = logic_loss.partial(X=x)
            >>> print(partial.free_variables)     # {'Y'}
        """
        return self._compiled_expr.free_variables

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get all trainable parameters from model-based predicates.

        Delegates to the underlying CompiledExpression.

        Returns:
            List of torch.nn.Parameter objects from all model-based predicates

        Example:
            >>> params = logic_loss.get_trainable_parameters()
            >>> optimizer = torch.optim.Adam(params, lr=0.001)
        """
        return self._compiled_expr.get_trainable_parameters()
