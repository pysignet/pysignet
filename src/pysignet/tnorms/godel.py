"""Godel t-norm implementation."""

import torch

from pysignet.tnorms.base import TNorm


class GodelTNorm(TNorm):
    """Godel t-norm (minimum/maximum).

    - AND: min(values) along dim=0
    - OR: max(values) along dim=0

    Most conservative option, but can have gradient issues.
    """

    @property
    def recommended_postprocessing(self) -> str:
        """Godel recommends linear post-processing."""
        return 'linear'

    def conjunction(self, values: torch.Tensor) -> torch.Tensor:
        """Godel conjunction: min along dim=0."""
        return values.min(dim=0).values

    def disjunction(self, values: torch.Tensor) -> torch.Tensor:
        """Godel disjunction: max along dim=0."""
        return values.max(dim=0).values
