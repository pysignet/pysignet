"""Base class for t-norm implementations."""

from abc import ABC, abstractmethod

import torch


class TNorm(ABC):
    """Base class for t-norms (continuous relaxations of logic ops).

    Conjunction and disjunction operate on a tensor, reducing along
    dim=0. This supports both n-ary expression evaluation (stacking
    multiple per-batch tensors) and batch reduction (reducing a 1D
    tensor to a scalar).

    Args:
        values: Tensor of shape (n, ...) where n >= 1.

    Returns:
        Tensor of shape (...) with the operation applied along dim=0.
    """

    @property
    @abstractmethod
    def recommended_postprocessing(self) -> str:
        """Return recommended loss post-processing mode.

        Returns:
            'log' for -log(satisfaction) or 'linear' for 1 - satisfaction
        """

    @abstractmethod
    def conjunction(self, values: torch.Tensor) -> torch.Tensor:
        """Relaxed AND operation, reducing along dim=0.

        Args:
            values: Tensor of shape (n, ...) with values in [0, 1].

        Returns:
            Tensor of shape (...) with conjunction applied.
        """

    @abstractmethod
    def disjunction(self, values: torch.Tensor) -> torch.Tensor:
        """Relaxed OR operation, reducing along dim=0.

        Args:
            values: Tensor of shape (n, ...) with values in [0, 1].

        Returns:
            Tensor of shape (...) with disjunction applied.
        """

    def negation(self, a: torch.Tensor) -> torch.Tensor:
        """Relaxed NOT operation (standard across all t-norms)."""
        result: torch.Tensor = 1.0 - a
        return result

    def implication(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """Relaxed IMPLIES operation: a -> b = NOT(a) OR b."""
        return self.disjunction(torch.stack([self.negation(a), b]))

    def equivalence(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """Relaxed EQUIVALENCE: a <-> b = (a -> b) AND (b -> a)."""
        return self.conjunction(torch.stack([
            self.implication(a, b),
            self.implication(b, a)
        ]))
