"""T-norm and t-conorm implementations for relaxing logic operators."""

from abc import ABC, abstractmethod

import torch


class TNorm(ABC):
    """Base class for t-norms (continuous relaxations of logic ops)."""

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


class SProductTNorm(TNorm):
    """S-Product t-norm (S-logics: implication as disjunction).

    - AND: a * b
    - OR: a + b - a * b
    - IMPLIES: 1 - a + a * b (treats implication as NOT(a) OR b)

    S-Product uses the standard implication-as-disjunction approach.
    According to "Evaluating Relaxations of Logic for Neural Networks",
    S-Product is less consistent than R-Product and performs worse
    empirically, but is equivalent to cross-entropy for labeled data.
    """

    def conjunction(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """Product conjunction."""
        return a * b

    def disjunction(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """Product disjunction."""
        return a + b - a * b


class RProductTNorm(SProductTNorm):
    """R-Product t-norm (R-logics: axiomatic residuum-based implication).

    - AND: a * b (inherited from S-Product)
    - OR: a + b - a * b (inherited from S-Product)
    - IMPLIES: 1 if a <= b else b/a (residuum, overrides S-Product)

    R-Product defines implication axiomatically using residua rather than
    treating it as disjunction. According to "Evaluating Relaxations of
    Logic for Neural Networks" (2107.13646v1.pdf):
    - R-Product empirically outperforms all other t-norms (Tables 3-9)
    - R-Product is self-consistent for all formulas (Proposition 1)
    - R-Product is the recommended default t-norm for neural networks

    This is the default t-norm used by LogicLoss.
    """

    def implication(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """Relaxed IMPLIES using R-Product residuum.

        R-Product implication: 1 if a <= b else b/a

        This axiomatic definition makes R-Product self-consistent and
        more suitable for neural network training than S-Product.

        Args:
            a: Antecedent tensor (values in [0, 1])
            b: Consequent tensor (values in [0, 1])

        Returns:
            Implication result: 1 where a <= b, else b/a
        """
        # Use torch.where for differentiable conditional
        # Add small epsilon to avoid division by zero
        result: torch.Tensor = torch.where(
            a <= b,
            torch.ones_like(a),
            b / torch.clamp(a, min=1e-10)
        )
        return result


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


class GodelTNorm(TNorm):
    """Gödel t-norm (minimum/maximum).

    - AND: min(a, b)
    - OR: max(a, b)

    Most conservative option, but can have gradient issues.
    """

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
