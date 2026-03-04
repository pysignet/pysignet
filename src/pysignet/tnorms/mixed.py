"""MixedTNorm - uses Godel for large arities, RProduct otherwise.

MixedTNorm addresses numerical stability issues with product t-norms when
many values are combined (e.g., 0.9^10 = 0.35, causing gradient vanishing).
For large conjunctions/disjunctions, Godel (min/max) is more stable.
"""

import torch

from pysignet.tnorms.base import TNorm
from pysignet.tnorms.godel import GodelTNorm
from pysignet.tnorms.product import RProductTNorm


class MixedTNorm(TNorm):
    """Mixed t-norm: Godel for large arities, RProduct for small arities.

    This t-norm switches behavior based on the number of arguments:
    - For arities <= threshold: uses RProduct (product/probabilistic sum)
    - For arities > threshold: uses Godel (min/max)

    This addresses numerical stability issues with product t-norms when
    combining many values. For example, 0.9^20 = 0.12, which can cause
    gradient vanishing during training. Godel (min/max) is stable for
    large arities.

    Binary operations (implication, equivalence) always use RProduct since
    they only involve 2 operands.

    Args:
        threshold: Maximum arity for RProduct. Arities > threshold use Godel.
            Default is 4.

    Example:
        ```python
        tnorm = MixedTNorm(threshold=4)
        # Small conjunction (3 args) -> RProduct
        values = torch.tensor([[0.8], [0.7], [0.6]])
        tnorm.conjunction(values)  # 0.8 * 0.7 * 0.6 = 0.336
        # Large conjunction (6 args) -> Godel
        values = torch.tensor([[0.9], [0.8], [0.7], [0.6], [0.5], [0.4]])
        tnorm.conjunction(values)  # min = 0.4
        ```
    """

    def __init__(self, threshold: int = 4) -> None:
        """Initialize MixedTNorm.

        Args:
            threshold: Maximum arity for RProduct. Arities > threshold
                use Godel. Default is 4.
        """
        self.threshold = threshold
        self._godel = GodelTNorm()
        self._rproduct = RProductTNorm()

    @property
    def recommended_postprocessing(self) -> str:
        """Return recommended loss post-processing mode.

        Returns 'log' since RProduct is used for small arities and
        binary operations (implication, equivalence).
        """
        return "log"

    def conjunction(self, values: torch.Tensor) -> torch.Tensor:
        """Relaxed AND operation, reducing along dim=0.

        Uses RProduct (product) for small arities (<=threshold),
        Godel (min) for large arities (>threshold).

        Args:
            values: Tensor of shape (n, ...) with values in [0, 1].

        Returns:
            Tensor of shape (...) with conjunction applied.
        """
        if values.shape[0] > self.threshold:
            return self._godel.conjunction(values)
        return self._rproduct.conjunction(values)

    def disjunction(self, values: torch.Tensor) -> torch.Tensor:
        """Relaxed OR operation, reducing along dim=0.

        Uses RProduct (probabilistic sum) for small arities (<=threshold),
        Godel (max) for large arities (>threshold).

        Args:
            values: Tensor of shape (n, ...) with values in [0, 1].

        Returns:
            Tensor of shape (...) with disjunction applied.
        """
        if values.shape[0] > self.threshold:
            return self._godel.disjunction(values)
        return self._rproduct.disjunction(values)

    def implication(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Relaxed IMPLIES using RProduct residuum.

        Implication is always binary, so RProduct is used for
        better gradient properties.

        Args:
            a: Antecedent tensor (values in [0, 1])
            b: Consequent tensor (values in [0, 1])

        Returns:
            Implication result using RProduct.
        """
        return self._rproduct.implication(a, b)

    def equivalence(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Relaxed EQUIVALENCE using RProduct.

        Equivalence is conjunction of two implications (binary),
        so RProduct is used for better gradient properties.

        Args:
            a: First operand tensor (values in [0, 1])
            b: Second operand tensor (values in [0, 1])

        Returns:
            Equivalence result using RProduct.
        """
        return self._rproduct.equivalence(a, b)
