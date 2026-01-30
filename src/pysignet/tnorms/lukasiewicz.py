"""Lukasiewicz t-norm implementation."""

import torch

from pysignet.tnorms.base import TNorm


class LukasiewiczTNorm(TNorm):
    """Lukasiewicz t-norm (bounded difference).

    - AND: max(0, sum(values) - (n - 1))
    - OR: min(1, sum(values))

    Good for enforcing stricter logical constraints.
    """

    @property
    def recommended_postprocessing(self) -> str:
        """Lukasiewicz recommends linear post-processing."""
        return 'linear'

    def conjunction(self, values: torch.Tensor) -> torch.Tensor:
        """Lukasiewicz conjunction: max(0, sum - (n-1))."""
        n = values.shape[0]
        return torch.clamp(
            values.sum(dim=0) - (n - 1), min=0.0
        )

    def disjunction(self, values: torch.Tensor) -> torch.Tensor:
        """Lukasiewicz disjunction: min(1, sum)."""
        return torch.clamp(values.sum(dim=0), max=1.0)
