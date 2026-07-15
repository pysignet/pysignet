"""Godel t-norm implementation."""

import torch

from pysignet.tnorms.base import TNorm


class GodelTNorm(TNorm):
    """Godel t-norm (minimum/maximum).

    - AND: min(values) along dim=0
    - OR: max(values) along dim=0

    Most conservative option, but can have gradient issues.

    Uses torch.amin/amax rather than values.min(dim=0).values /
    values.max(dim=0).values: the latter also compute and discard an
    unused argmin/argmax index, which is far more expensive for no
    benefit (~60x slower, measured on small batches) and, for tied
    values, gives the entire gradient to a single arbitrary
    (first-occurring) index rather than splitting it fairly across
    every tied element.
    """

    @property
    def recommended_postprocessing(self) -> str:
        """Godel recommends linear post-processing."""
        return "linear"

    def conjunction(self, values: torch.Tensor) -> torch.Tensor:
        """Godel conjunction: min along dim=0."""
        return torch.amin(values, dim=0)

    def disjunction(self, values: torch.Tensor) -> torch.Tensor:
        """Godel disjunction: max along dim=0."""
        return torch.amax(values, dim=0)
