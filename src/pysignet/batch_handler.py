"""Batch handling mixin for compiler-aware reduction.

This module provides a reusable mixin for batch reduction that delegates
conjunction and disjunction to the compiler that produced the expression.
This ensures batch reduction (forall/exists) uses the same logical operations
as expression evaluation.
"""

from typing import Literal, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pysignet.compilation.base import LogicCompiler


class BatchHandlerMixin:
    """Mixin providing compiler-aware batch reduction.

    This mixin provides methods for reducing batches of satisfaction values
    using forall (conjunction) or exists (disjunction) quantification,
    delegating to the compiler's conjunction/disjunction operations.

    Subclasses must have a `_compiler` attribute containing a LogicCompiler.

    Attributes:
        _compiler: The LogicCompiler used for batch reduction operations.

    Example:
        >>> class MyLogicHandler(BatchHandlerMixin):
        ...     def __init__(self, compiler):
        ...         self._compiler = compiler
        ...
        >>> handler = MyLogicHandler(compiler)
        >>> result = handler._reduce_batch(tensor, quantifier='forall')
    """

    _compiler: 'LogicCompiler'

    def _reduce_batch(
        self,
        tensor: torch.Tensor,
        quantifier: Literal['forall', 'exists', 'none'] = 'forall',
    ) -> torch.Tensor:
        """Reduce batch using the specified quantifier.

        Delegates to the compiler's conjunction (forall) or disjunction
        (exists) operations.

        Args:
            tensor: 1D tensor of satisfaction values.
            quantifier: The quantifier to use:
                - 'forall': Conjunction (AND) over all values
                - 'exists': Disjunction (OR) over all values
                - 'none': No reduction, return tensor unchanged

        Returns:
            Reduced tensor (scalar for forall/exists, original for
            none).

        Raises:
            ValueError: If quantifier is invalid.
        """
        valid_quantifiers = ('forall', 'exists', 'none')
        if quantifier not in valid_quantifiers:
            raise ValueError(
                f"Invalid quantifier '{quantifier}'. "
                f"Must be one of {valid_quantifiers}."
            )

        if quantifier == 'none':
            return tensor

        if tensor.numel() == 0:
            if quantifier == 'forall':
                return torch.tensor(
                    1.0, dtype=tensor.dtype, device=tensor.device
                )
            else:
                return torch.tensor(
                    0.0, dtype=tensor.dtype, device=tensor.device
                )

        if quantifier == 'forall':
            return self._compiler.conjunction(tensor)
        else:  # exists
            return self._compiler.disjunction(tensor)

    def _apply_reduction(
        self,
        tensor: torch.Tensor,
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ) -> torch.Tensor:
        """Apply final reduction (mean/sum/none) to tensor.

        This is for loss aggregation, separate from quantifier-based
        reduction.

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
