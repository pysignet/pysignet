"""Product t-norm implementations (S-Product and R-Product)."""

import torch

from .base import TNorm


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
