"""LogicLoss - wraps compiled logic with loss computation."""

from typing import Callable, Dict, Union, Optional, List

import torch

from .predicate import Predicate


class LogicLoss:
    """Wrapper for compiled logic that provides loss computation.

    LogicLoss wraps a compiled logic expression (returned by
    LogicCompiler.compile()) and provides methods for computing
    satisfaction degrees and losses with configurable post-processing
    and reduction.

    Args:
        compiled_logic: Callable returned by LogicCompiler.compile()
        predicates: Dict mapping predicate names to Predicate objects
        post_processing: Default post-processing mode - 'log', 'linear', or
                        callable (default: 'linear')

    Example:
        >>> compiler = TNormCompiler()
        >>> compiled = compiler.compile(expr, predicates)
        >>> logic_loss = LogicLoss(compiled, predicates)
        >>> satisfaction = logic_loss(x)  # Returns [0, 1]
        >>> loss = logic_loss.loss(x)  # Returns loss value
    """

    def __init__(
        self,
        compiled_logic: Callable[
            [Union[torch.Tensor, Dict[str, torch.Tensor]]], torch.Tensor
        ],
        predicates: Dict[str, Predicate],
        post_processing: Union[
            str, Callable[[torch.Tensor], torch.Tensor]
        ] = 'linear'
    ) -> None:
        """Initialize LogicLoss.

        Args:
            compiled_logic: Compiled logic expression
            predicates: Dict of predicates for parameter extraction
            post_processing: Default post-processing ('log', 'linear', callable)
        """
        self.compiled_logic = compiled_logic
        self.predicates = predicates
        self.default_post_processing = post_processing

    def __call__(
        self,
        inputs: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        **variable_bindings: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate compiled logic and return satisfaction degrees.

        Args:
            inputs: Single tensor, dict of tensors, or None (for constant-only
                   predicates with no variables). Default: None.
            **variable_bindings: FOL variable bindings (e.g., X=x_tensor, Y=y_tensor)

        Returns:
            Satisfaction tensor of shape (batch_size,) in [0, 1].
            Higher values = better satisfaction.

        Examples:
            >>> # Traditional API (single tensor)
            >>> result = logic_loss(x)
            >>>
            >>> # Traditional API (dict)
            >>> result = logic_loss({"P": x_p, "Q": x_q})
            >>>
            >>> # FOL API (variable bindings)
            >>> result = logic_loss(X=x, Y=y)
            >>>
            >>> # Constant-only predicates (no inputs)
            >>> result = logic_loss()
        """
        # If variable bindings provided via kwargs, use them
        if variable_bindings:
            if inputs is not None:
                raise ValueError(
                    "Cannot provide both positional 'inputs' and variable "
                    "bindings (**kwargs). Use one or the other."
                )
            inputs = variable_bindings
        # Handle no-input case (constant-only predicates)
        elif inputs is None:
            inputs = {}
        return self.compiled_logic(inputs)

    def loss(
        self,
        inputs: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        reduction: str = 'mean',
        post_processing: Optional[
            Union[str, Callable[[torch.Tensor], torch.Tensor]]
        ] = None,
        **variable_bindings: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss based on logical constraint violation.

        Args:
            inputs: Inputs for predicates, or None for constant-only predicates
            reduction: 'mean', 'sum', or 'none' (default: 'mean')
            post_processing: Post-processing mode - 'log', 'linear', callable,
                           or None (uses default from __init__)
            **variable_bindings: FOL variable bindings (e.g., X=x_tensor, Y=y_tensor)

        Returns:
            Loss value (lower = better satisfaction)

        Raises:
            ValueError: If invalid post_processing or reduction mode
        """
        # Compute satisfaction
        satisfaction = self(inputs, **variable_bindings)

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

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get all trainable parameters from model-based predicates.

        Returns:
            List of torch.nn.Parameter objects from all model-based predicates

        Example:
            >>> params = logic_loss.get_trainable_parameters()
            >>> optimizer = torch.optim.Adam(params, lr=0.001)
        """
        params: List[torch.nn.Parameter] = []
        for pred in self.predicates.values():
            if pred.is_model and hasattr(pred.func, 'parameters'):
                params.extend(pred.func.parameters())
        return params
