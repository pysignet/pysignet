"""LogicLoss - wraps compiled expressions with loss computation.

LogicLoss provides explicit methods for computing loss and satisfaction:
- satisfaction(): Get soft satisfaction values
- loss(): Get loss values (for training)
- log_satisfaction(): Get log-space satisfaction (numerical stability)
- partial(): Create partially-bound expression

Batch quantification control:
- quantify='forall': Universal quantification (conjunction) over batch
- quantify='exists': Existential quantification (disjunction) over batch
- quantify='none': No quantification (return per-batch results)
"""

from typing import Callable, List, Literal

import torch

from pysignet.batch_handler import BatchHandlerMixin
from pysignet.compilation.compiled_expression import CompiledExpression
from pysignet.compilation.tnorm_compiler import TNormCompiler
from pysignet.tnorms import SProductTNorm


class LogicLoss(BatchHandlerMixin):
    """Wrapper for CompiledExpression that adds loss computation.

    LogicLoss delegates evaluation to an underlying CompiledExpression,
    then applies compiler-aware batch reduction via BatchHandlerMixin,
    and adds loss-specific functionality (post-processing, reduction).

    Methods:
        satisfaction(): Get soft satisfaction values in [0, 1]
        loss(): Get loss values for training (lower = better)
        log_satisfaction(): Get log-space satisfaction for stability
        partial(): Create partially-bound expression

    Properties:
        free_variables: Set of unbound variable names
        trainable_parameters: List of trainable nn.Parameters

    Args:
        compiled_expr: CompiledExpression from compiler.compile()
        post_processing: Default post-processing mode - 'log', 'linear',
            or callable. If None, uses the compiler's recommendation.

    Example:
        >>> from pysignet import logic_to_loss
        >>> logic_loss = logic_to_loss(expr, predicates)
        >>>
        >>> sat = logic_loss.satisfaction(X=x)  # Soft truth values
        >>> loss = logic_loss.loss(X=x)         # For training
        >>> params = logic_loss.trainable_parameters
    """

    def __init__(
        self,
        compiled_expr: CompiledExpression,
        post_processing: (
            str
            | Callable[[torch.Tensor], torch.Tensor]
            | None
        ) = None,
    ) -> None:
        """Initialize LogicLoss.

        Args:
            compiled_expr: CompiledExpression to wrap
            post_processing: Default post-processing
                ('log', 'linear', callable). If None, uses the
                compiler's recommended mode.
        """
        self._compiled_expr = compiled_expr

        self._compiler = self._get_compiler(compiled_expr)

        pp = self._compiler.recommended_postprocessing
        self.default_post_processing: (
            str | Callable[[torch.Tensor], torch.Tensor]
        ) = (
            post_processing if post_processing is not None
            else pp
        )

        # Cache whether batch compiler uses product conjunction
        self._product_conjunction = (
            isinstance(self._compiler, TNormCompiler)
            and isinstance(
                self._compiler.tnorm, SProductTNorm
            )
        )

    def satisfaction(
        self,
        quantify: Literal["forall", "exists", "none"] = "forall",
        **variable_bindings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute soft satisfaction of the logical expression.

        Args:
            quantify: Batch quantification mode:
                - 'forall': Universal quantification -> scalar
                - 'exists': Existential quantification -> scalar
                - 'none': No quantification -> (batch_size,)
            **variable_bindings: Variable bindings as keyword arguments
                (e.g., X=x_tensor, Y=y_tensor)

        Returns:
            Satisfaction tensor in [0, 1]:
            - Scalar if quantify='forall' or 'exists'
            - (batch_size,) if quantify='none'

        Raises:
            ValueError: If quantify is invalid

        Examples:
            >>> sat = logic_loss.satisfaction(X=x)  # scalar (forall)
            >>> sat = logic_loss.satisfaction(X=x, quantify='exists')
            >>> sat = logic_loss.satisfaction(X=x, quantify='none')
        """
        # Validate quantify parameter
        valid_quantifiers = ("forall", "exists", "none")
        if quantify not in valid_quantifiers:
            raise ValueError(
                f"Invalid quantify value '{quantify}'. "
                f"Must be one of {valid_quantifiers}."
            )

        # Get per-batch satisfaction from compiled expression
        per_batch = self._compiled_expr(
            return_boolean=False, log_mode=False,
            **variable_bindings,
        )

        # Apply batch quantification using BatchHandlerMixin
        return self._reduce_batch(per_batch, quantifier=quantify)

    def log_satisfaction(
        self,
        quantify: Literal["forall", "exists", "none"] = "forall",
        **variable_bindings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-satisfaction for numerical stability.

        For product-based compilers with large batches, computing
        satisfaction in linear space can underflow to zero. When
        using product conjunction with forall quantification, this
        method computes sum(log(p_i)) directly instead of
        log(product(p_i)), which is mathematically identical but
        avoids underflow.

        Args:
            quantify: Batch quantification mode (same as satisfaction())
            **variable_bindings: Variable bindings as keyword arguments
                (e.g., X=x_tensor, Y=y_tensor)

        Returns:
            Log-satisfaction tensor in (-inf, 0]:
            - Scalar if quantify='forall' or 'exists'
            - (batch_size,) if quantify='none'

        Examples:
            >>> log_sat = logic_loss.log_satisfaction(X=x)
        """
        valid_quantifiers = ("forall", "exists", "none")
        if quantify not in valid_quantifiers:
            raise ValueError(
                f"Invalid quantify value '{quantify}'. "
                f"Must be one of {valid_quantifiers}."
            )

        # Log-space shortcut for product forall:
        # log(prod(p_i)) = sum(log(p_i)), avoids underflow.
        # Uses fused log-activation ops (logsigmoid, log_softmax)
        # when available for better numerical stability.
        if quantify == "forall" and self._product_conjunction:
            per_batch = self._compiled_expr(
                return_boolean=False, log_mode=True,
                **variable_bindings,
            )
            return per_batch.sum()

        # Fallback: compute satisfaction, then take log
        sat = self.satisfaction(quantify=quantify, **variable_bindings)
        return torch.log(sat + 1e-10)

    def loss(
        self,
        quantify: Literal["forall", "exists", "none"] = "forall",
        reduction: Literal["mean", "sum", "none"] = "none",
        post_processing: (
            str
            | Callable[[torch.Tensor], torch.Tensor]
            | None
        ) = None,
        **variable_bindings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss based on logical constraint violation.

        Args:
            quantify: Batch quantification mode:
                - 'forall': Universal quantification -> scalar loss
                - 'exists': Existential quantification -> scalar loss
                - 'none': No quantification -> per-batch losses
            reduction: Loss reduction mode (only with quantify='none'):
                - 'mean': Mean of per-batch losses
                - 'sum': Sum of per-batch losses
                - 'none': Return per-batch losses
            post_processing: Post-processing mode - 'log', 'linear',
                callable, or None (uses default from __init__)
            **variable_bindings: Variable bindings as keyword arguments
                (e.g., X=x_tensor, Y=y_tensor)

        Returns:
            Loss value (lower = better satisfaction):
            - Scalar if quantify='forall'/'exists', or
              reduction='mean'/'sum'
            - (batch_size,) if quantify='none' and reduction='none'

        Raises:
            ValueError: If invalid quantify, reduction, or
                post_processing
            ValueError: If reduction != 'none' with
                quantify='forall'/'exists'

        Examples:
            >>> loss = logic_loss.loss(X=x)  # scalar
            >>> loss = logic_loss.loss(
            ...     X=x, quantify='none', reduction='mean'
            ... )
        """
        # Validate quantify
        valid_quantifiers = ("forall", "exists", "none")
        if quantify not in valid_quantifiers:
            raise ValueError(
                f"Invalid quantify value '{quantify}'. "
                f"Must be one of {valid_quantifiers}."
            )

        # Validate reduction is only used with quantify='none'
        if quantify != "none" and reduction != "none":
            raise ValueError(
                f"reduction='{reduction}' is invalid with "
                f"quantify='{quantify}'. When using "
                f"quantify='forall' or 'exists', the result is "
                f"already a scalar. Use reduction='none' or omit "
                f"the reduction parameter."
            )

        # Determine post-processing mode
        postprocessing_type = (
            post_processing
            if post_processing is not None
            else self.default_post_processing
        )

        # For log post-processing, delegate to log_satisfaction
        # which handles log-space computation to avoid underflow
        if postprocessing_type == "log":
            log_sat = self.log_satisfaction(
                quantify=quantify, **variable_bindings
            )
            loss_values = -log_sat
            return self._apply_reduction(loss_values, reduction=reduction)

        # Get per-batch satisfaction from compiled expression
        per_batch = self._compiled_expr(
            return_boolean=False, log_mode=False,
            **variable_bindings,
        )

        # Apply batch quantification
        if quantify == "none":
            satisfaction = per_batch
        else:
            satisfaction = self._reduce_batch(per_batch, quantifier=quantify)

        # Apply post-processing to convert satisfaction to loss
        if postprocessing_type == "linear":
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

    def partial(self, **variable_bindings: torch.Tensor) -> "LogicLoss":
        """Create partially-bound LogicLoss with some variables fixed.

        Delegates to the underlying CompiledExpression and wraps the
        result in a new LogicLoss.

        Args:
            **variable_bindings: Variable bindings to fix (e.g., X=x)

        Returns:
            New LogicLoss with fewer free variables

        Raises:
            ValueError: If no bindings provided
            ValueError: If variable is already bound
            ValueError: If variable doesn't exist in expression

        Examples:
            >>> partial = logic_loss.partial(X=x)
            >>> result = partial(Y=y)
        """
        # Delegate to CompiledExpression
        partial_expr = self._compiled_expr.partial(**variable_bindings)

        # Wrap in new LogicLoss with same settings
        return LogicLoss(
            partial_expr,
            self.default_post_processing,
        )

    @property
    def free_variables(self) -> set[str]:
        """Return set of variables that still need to be bound.

        Delegates to the underlying CompiledExpression.

        Returns:
            Set of variable names not yet bound
        """
        return self._compiled_expr.free_variables

    @property
    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        """All trainable parameters from model-based predicates.

        Delegates to the underlying CompiledExpression.

        Returns:
            List of torch.nn.Parameter objects from all model-based
            predicates

        Example:
            >>> params = logic_loss.trainable_parameters
            >>> optimizer = torch.optim.Adam(params, lr=0.001)
        """
        return self._compiled_expr.get_trainable_parameters()
