"""Batch handling mixin for t-norm aware reduction.

This module provides a reusable mixin for batch reduction that is aware of
t-norm computation spaces. Product t-norms use log-space for numerical
stability, while other t-norms use linear space.

The mixin consolidates all batch reduction logic, previously scattered across
CompiledExpression and LogicLoss, into a single reusable component.
"""

from typing import Literal

import torch

from pysignet.tnorms.base import TNorm


class BatchHandlerMixin:
    """Mixin providing t-norm aware batch reduction.

    This mixin provides methods for reducing batches of satisfaction values
    using forall (conjunction) or exists (disjunction) quantification.

    For product t-norms (RProduct, SProduct), log-space computation is used
    for numerical stability with large batches. For other t-norms (Lukasiewicz,
    Godel), linear-space computation is used.

    Subclasses must have a `_tnorm` attribute containing a TNorm instance.

    Attributes:
        _tnorm: The t-norm instance used for determining computation space.

    Example:
        >>> class MyLogicHandler(BatchHandlerMixin):
        ...     def __init__(self, tnorm):
        ...         self._tnorm = tnorm
        ...
        >>> handler = MyLogicHandler(RProductTNorm())
        >>> result = handler._reduce_batch(tensor, quantifier='forall')
    """

    _tnorm: TNorm

    def _is_product_tnorm(self) -> bool:
        """Check if the t-norm is a product t-norm.

        Product t-norms (RProduct, SProduct) use log-space for numerical
        stability, while other t-norms use linear space.

        Returns:
            True if the t-norm recommends log post-processing (product t-norm).
        """
        return self._tnorm.recommended_postprocessing == 'log'

    def _reduce_forall_linear(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce batch using forall (conjunction) in linear space.

        Computes the product of all values in the tensor, which corresponds
        to the conjunction of all satisfaction values.

        Args:
            tensor: 1D tensor of satisfaction values in [0, 1].

        Returns:
            Scalar tensor with the product of all values.
            Returns 1.0 for empty tensor (vacuous truth).
        """
        if tensor.numel() == 0:
            return torch.tensor(1.0, dtype=tensor.dtype, device=tensor.device)
        return tensor.prod()

    def _reduce_forall_log(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce batch using forall (conjunction) in log space.

        Computes the sum of log values, which is numerically stable for
        product t-norms with large batches.

        Note: This returns log-satisfaction, not satisfaction.
        log(prod(x_i)) = sum(log(x_i))

        Args:
            tensor: 1D tensor of satisfaction values in [0, 1].

        Returns:
            Scalar tensor with the sum of log values (in (-inf, 0]).
            Returns 0.0 for empty tensor (log(1) = 0).
        """
        if tensor.numel() == 0:
            return torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device)
        # Add small epsilon to avoid log(0)
        return torch.log(tensor + 1e-10).sum()

    def _reduce_exists_linear(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce batch using exists (disjunction) in linear space.

        Computes the probabilistic OR of all values, which corresponds
        to the disjunction of all satisfaction values.

        P(A or B) = P(A) + P(B) - P(A)*P(B)

        For multiple values, this is applied iteratively:
        result = 1 - prod(1 - x_i)

        Args:
            tensor: 1D tensor of satisfaction values in [0, 1].

        Returns:
            Scalar tensor with the probabilistic OR of all values.
            Returns 0.0 for empty tensor (no witnesses).
        """
        if tensor.numel() == 0:
            return torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device)
        # exists: 1 - prod(1 - x_i)
        return 1.0 - (1.0 - tensor).prod()

    def _reduce_batch(
        self,
        tensor: torch.Tensor,
        quantifier: Literal['forall', 'exists', 'none'] = 'forall',
        space: Literal['linear', 'log', 'auto'] = 'auto'
    ) -> torch.Tensor:
        """Reduce batch using the specified quantifier and computation space.

        Args:
            tensor: 1D tensor of satisfaction values.
            quantifier: The quantifier to use:
                - 'forall': Conjunction (AND) over all values
                - 'exists': Disjunction (OR) over all values
                - 'none': No reduction, return tensor unchanged
            space: The computation space:
                - 'linear': Compute in [0, 1] space
                - 'log': Compute in log space (returns log-satisfaction)
                - 'auto': Use log for product t-norms, linear otherwise

        Returns:
            Reduced tensor (scalar for forall/exists, original for none).

        Raises:
            ValueError: If quantifier or space is invalid.
        """
        valid_quantifiers = ('forall', 'exists', 'none')
        if quantifier not in valid_quantifiers:
            raise ValueError(
                f"Invalid quantifier '{quantifier}'. "
                f"Must be one of {valid_quantifiers}."
            )

        valid_spaces = ('linear', 'log', 'auto')
        if space not in valid_spaces:
            raise ValueError(
                f"Invalid space '{space}'. Must be one of {valid_spaces}."
            )

        if quantifier == 'none':
            return tensor

        # Determine actual space to use
        if space == 'auto':
            actual_space = 'log' if self._is_product_tnorm() else 'linear'
        else:
            actual_space = space

        if quantifier == 'forall':
            if actual_space == 'log':
                return self._reduce_forall_log(tensor)
            else:
                return self._reduce_forall_linear(tensor)
        else:  # exists
            # Exists always uses linear space
            return self._reduce_exists_linear(tensor)

    def _apply_reduction(
        self,
        tensor: torch.Tensor,
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ) -> torch.Tensor:
        """Apply final reduction (mean/sum/none) to tensor.

        This is for loss aggregation, separate from quantifier-based reduction.

        Args:
            tensor: Tensor to reduce.
            reduction: Reduction mode:
                - 'mean': Return mean of tensor
                - 'sum': Return sum of tensor
                - 'none': Return tensor unchanged

        Returns:
            Reduced tensor.

        Raises:
            ValueError: If reduction is invalid.
        """
        valid_reductions = ('mean', 'sum', 'none')
        if reduction not in valid_reductions:
            raise ValueError(
                f"Invalid reduction '{reduction}'. "
                f"Must be one of {valid_reductions}."
            )

        if reduction == 'mean':
            return tensor.mean() if tensor.numel() > 0 else tensor
        elif reduction == 'sum':
            return tensor.sum() if tensor.numel() > 0 else tensor
        else:  # none
            return tensor
