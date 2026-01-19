"""LogicLoss - wraps compiled expressions with loss computation.

LogicLoss uses BatchHandlerMixin for t-norm aware batch reduction with
explicit quantification control:
- quantify='forall': Universal quantification (conjunction) over batch
- quantify='exists': Existential quantification (disjunction) over batch
- quantify='none': No quantification (return per-batch results)

The `reduction` parameter is only valid with `quantify='none'`.
"""

from typing import Callable, Dict, List, Literal, Optional

import torch

from pysignet.batch_handler import BatchHandlerMixin
from pysignet.compilation.compiled_expression import CompiledExpression
from pysignet.tnorms import TNorm, RProductTNorm


class LogicLoss(BatchHandlerMixin):
    """Wrapper for CompiledExpression that adds loss computation.

    LogicLoss delegates evaluation to an underlying CompiledExpression,
    then applies t-norm aware batch reduction via BatchHandlerMixin,
    and adds loss-specific functionality (post-processing, reduction).

    This creates a clean separation of concerns:
    - CompiledExpression: Pure evaluation, returns (batch_size,)
    - LogicLoss: Batch quantification, loss computation

    Args:
        compiled_expr: CompiledExpression from LogicCompiler.compile()
        post_processing: Default post-processing mode - 'log', 'linear', or
                        callable (default: 'linear')
        tnorm: T-norm instance for batch quantification. If None, uses
              RProductTNorm (default).

    Example:
        >>> # Get CompiledExpression from compiler
        >>> compiler = TNormCompiler()
        >>> compiled = compiler.compile(expr, predicates)
        >>>
        >>> # Wrap in LogicLoss for training
        >>> logic_loss = LogicLoss(compiled, tnorm=RProductTNorm())
        >>> satisfaction = logic_loss(X=x)  # Returns scalar (forall)
        >>> loss = logic_loss.loss(X=x)     # Returns loss value
        >>>
        >>> # Or use api.compile_logic() which wraps automatically
        >>> from pysignet import compile_logic
        >>> logic_loss = compile_logic(expr, predicates)
    """

    def __init__(
        self,
        compiled_expr: CompiledExpression,
        post_processing: str | Callable[[torch.Tensor], torch.Tensor] = 'linear',
        tnorm: Optional[TNorm] = None
    ) -> None:
        """Initialize LogicLoss.

        Args:
            compiled_expr: CompiledExpression to wrap
            post_processing: Default post-processing ('log', 'linear', callable)
            tnorm: T-norm for batch quantification (default: RProductTNorm)
        """
        self._compiled_expr = compiled_expr
        self.default_post_processing = post_processing
        self._tnorm = tnorm or RProductTNorm()

    def __call__(
        self,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
        quantify: Literal['forall', 'exists', 'none'] = 'forall',
        **variable_bindings: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate compiled expression and return satisfaction degrees.

        IMPORTANT: All inputs must be provided as keyword arguments
        (e.g., logic_loss(X=tensor)). Positional arguments are not supported.

        Args:
            inputs: Optional dict mapping variable names to tensors.
                   Prefer using keyword arguments (**variable_bindings) instead.
            quantify: Batch quantification mode:
                - 'forall': Universal quantification (conjunction) -> scalar
                - 'exists': Existential quantification (disjunction) -> scalar
                - 'none': No quantification -> (batch_size,)
            **variable_bindings: Variable bindings (e.g., X=x, Y=y)

        Returns:
            Satisfaction tensor in [0, 1]:
            - Scalar if quantify='forall' or 'exists'
            - (batch_size,) if quantify='none'

        Raises:
            ValueError: If quantify is invalid

        Examples:
            >>> # Keyword argument interface (required)
            >>> result = logic_loss(X=x)  # scalar (forall default)
            >>>
            >>> # Existential quantification
            >>> result = logic_loss(X=x, quantify='exists')  # scalar
            >>>
            >>> # No quantification (per-batch)
            >>> result = logic_loss(X=x, quantify='none')  # (batch_size,)
        """
        # Validate quantify parameter
        valid_quantifiers = ('forall', 'exists', 'none')
        if quantify not in valid_quantifiers:
            raise ValueError(
                f"Invalid quantify value '{quantify}'. "
                f"Must be one of {valid_quantifiers}."
            )

        # Get per-batch satisfaction from compiled expression
        per_batch = self._compiled_expr(inputs=inputs, **variable_bindings)

        # Apply batch quantification using BatchHandlerMixin
        # Use 'linear' space to return satisfaction in [0, 1]
        # (For log-space results, use log_satisfaction method)
        return self._reduce_batch(per_batch, quantifier=quantify, space='linear')

    def log_satisfaction(
        self,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
        quantify: Literal['forall', 'exists', 'none'] = 'forall',
        **variable_bindings: torch.Tensor
    ) -> torch.Tensor:
        """Compute log-satisfaction for numerical stability.

        For product t-norms with large batches, computing satisfaction
        in linear space can underflow to zero. This method computes
        log(satisfaction) directly in log space for stability.

        Args:
            inputs: Optional dict mapping variable names to tensors.
                   Prefer using keyword arguments (**variable_bindings) instead.
            quantify: Batch quantification mode (same as __call__)
            **variable_bindings: Variable bindings (e.g., X=x, Y=y)

        Returns:
            Log-satisfaction tensor in (-inf, 0]:
            - Scalar if quantify='forall' or 'exists'
            - (batch_size,) if quantify='none'

        Examples:
            >>> # Stable computation for large batches
            >>> log_sat = logic_loss.log_satisfaction(X=x)
            >>> # log_sat is sum of log(satisfaction_i) for forall
        """
        valid_quantifiers = ('forall', 'exists', 'none')
        if quantify not in valid_quantifiers:
            raise ValueError(
                f"Invalid quantify value '{quantify}'. "
                f"Must be one of {valid_quantifiers}."
            )

        # Get per-batch satisfaction from compiled expression
        per_batch = self._compiled_expr(inputs=inputs, **variable_bindings)

        # Always use log space for numerical stability
        return self._reduce_batch(per_batch, quantifier=quantify, space='log')

    def loss(
        self,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
        quantify: Literal['forall', 'exists', 'none'] = 'forall',
        reduction: Literal['mean', 'sum', 'none'] = 'none',
        post_processing: str | Callable[[torch.Tensor], torch.Tensor] | None = None,
        **variable_bindings: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss based on logical constraint violation.

        IMPORTANT: All inputs must be provided as keyword arguments
        (e.g., logic_loss.loss(X=tensor)). Positional arguments are not supported.

        Args:
            inputs: Optional dict mapping variable names to tensors.
                   Prefer using keyword arguments (**variable_bindings) instead.
            quantify: Batch quantification mode:
                - 'forall': Universal quantification -> scalar loss
                - 'exists': Existential quantification -> scalar loss
                - 'none': No quantification -> per-batch losses
            reduction: Loss reduction mode (only valid with quantify='none'):
                - 'mean': Mean of per-batch losses
                - 'sum': Sum of per-batch losses
                - 'none': Return per-batch losses
            post_processing: Post-processing mode - 'log', 'linear', callable,
                           or None (uses default from __init__)
            **variable_bindings: Variable bindings (e.g., X=x, Y=y)

        Returns:
            Loss value (lower = better satisfaction):
            - Scalar if quantify='forall'/'exists', or reduction='mean'/'sum'
            - (batch_size,) if quantify='none' and reduction='none'

        Raises:
            ValueError: If invalid quantify, reduction, or post_processing
            ValueError: If reduction != 'none' with quantify='forall'/'exists'

        Examples:
            >>> # Default: forall quantification
            >>> loss = logic_loss.loss(X=x)  # scalar
            >>>
            >>> # Per-batch losses with mean reduction
            >>> loss = logic_loss.loss(X=x, quantify='none', reduction='mean')
            >>>
            >>> # Per-batch losses without reduction
            >>> losses = logic_loss.loss(X=x, quantify='none', reduction='none')
        """
        # Validate quantify
        valid_quantifiers = ('forall', 'exists', 'none')
        if quantify not in valid_quantifiers:
            raise ValueError(
                f"Invalid quantify value '{quantify}'. "
                f"Must be one of {valid_quantifiers}."
            )

        # Validate reduction is only used with quantify='none'
        if quantify != 'none' and reduction != 'none':
            raise ValueError(
                f"reduction='{reduction}' is invalid with quantify='{quantify}'. "
                f"When using quantify='forall' or 'exists', the result is already "
                f"a scalar. Use reduction='none' or omit the reduction parameter."
            )

        # Get per-batch satisfaction from compiled expression
        per_batch = self._compiled_expr(inputs=inputs, **variable_bindings)

        # Apply batch quantification
        # For loss computation, we need satisfaction in linear space
        # (post-processing expects [0, 1] values)
        if quantify == 'none':
            satisfaction = per_batch
        else:
            # Use linear space for satisfaction (forall/exists)
            satisfaction = self._reduce_batch(
                per_batch, quantifier=quantify, space='linear'
            )

        # Determine post-processing mode
        postprocessing_type = (
            post_processing
            if post_processing is not None
            else self.default_post_processing
        )

        # Apply post-processing to convert satisfaction to loss
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

        # Apply reduction (only meaningful for quantify='none')
        return self._apply_reduction(loss_values, reduction=reduction)

    def partial(self, **variable_bindings: torch.Tensor) -> 'LogicLoss':
        """Create partially-bound LogicLoss with some variables fixed.

        Delegates to the underlying CompiledExpression and wraps the result
        in a new LogicLoss.

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
        """
        # Delegate to CompiledExpression
        partial_expr = self._compiled_expr.partial(**variable_bindings)

        # Wrap in new LogicLoss with same settings
        return LogicLoss(
            partial_expr,
            self.default_post_processing,
            tnorm=self._tnorm
        )

    @property
    def free_variables(self) -> set[str]:
        """Return set of variables that still need to be bound.

        Delegates to the underlying CompiledExpression.

        Returns:
            Set of variable names not yet bound
        """
        return self._compiled_expr.free_variables

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get all trainable parameters from model-based predicates.

        Delegates to the underlying CompiledExpression.

        Returns:
            List of torch.nn.Parameter objects from all model-based predicates
        """
        return self._compiled_expr.get_trainable_parameters()
