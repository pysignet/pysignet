"""Gödel t-norm implementation."""

import torch

from .base import TNorm


class GodelTNorm(TNorm):
    """Gödel t-norm (minimum/maximum).

    - AND: min(a, b)
    - OR: max(a, b)

    Most conservative option, but can have gradient issues.
    """

    @property
    def recommended_postprocessing(self) -> str:
        """Gödel recommends linear post-processing."""
        return 'linear'

    def conjunction(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """Gödel conjunction."""
        return torch.minimum(a, b)

    def disjunction(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """Gödel disjunction."""
        return torch.maximum(a, b)
