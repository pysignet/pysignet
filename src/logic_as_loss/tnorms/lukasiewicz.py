"""Łukasiewicz t-norm implementation."""

import torch

from .base import TNorm


class LukasiewiczTNorm(TNorm):
    """Łukasiewicz t-norm (bounded difference).

    - AND: max(0, a + b - 1)
    - OR: min(1, a + b)

    Good for enforcing stricter logical constraints.
    """

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
