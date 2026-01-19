"""Łukasiewicz t-norm implementation."""

import torch

from pysignet.tnorms.base import TNorm


class LukasiewiczTNorm(TNorm):
    """Łukasiewicz t-norm (bounded difference).

    - AND: max(0, a + b - 1)
    - OR: min(1, a + b)

    Good for enforcing stricter logical constraints.
    """

    @property
    def recommended_postprocessing(self) -> str:
        """Łukasiewicz recommends linear post-processing."""
        return 'linear'

    def conjunction(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """Łukasiewicz conjunction."""
        return torch.clamp(a + b - 1.0, min=0.0)

    def disjunction(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """Łukasiewicz disjunction."""
        return torch.clamp(a + b, max=1.0)
