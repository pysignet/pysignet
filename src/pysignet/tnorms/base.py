"""Base class for t-norm implementations."""

from abc import ABC, abstractmethod

import torch


class TNorm(ABC):
    """Base class for t-norms (continuous relaxations of logic ops)."""

    @property
    @abstractmethod
    def recommended_postprocessing(self) -> str:
        """Return recommended loss post-processing mode.

        Returns:
            'log' for -log(satisfaction) or 'linear' for 1 - satisfaction
        """

    @abstractmethod
    def conjunction(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """Relaxed AND operation."""

    @abstractmethod
    def disjunction(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """Relaxed OR operation."""

    def negation(self, a: torch.Tensor) -> torch.Tensor:
        """Relaxed NOT operation (standard across all t-norms)."""
        result: torch.Tensor = 1.0 - a
        return result

    def implication(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """Relaxed IMPLIES operation: a → b ≡ ¬a ∨ b."""
        return self.disjunction(self.negation(a), b)

    def equivalence(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """Relaxed EQUIVALENCE: a ↔ b ≡ (a → b) ∧ (b → a)."""
        # pylint: disable=arguments-out-of-order
        return self.conjunction(
            self.implication(a, b),
            self.implication(b, a)
        )
