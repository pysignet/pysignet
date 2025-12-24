"""
T-norm and t-conorm implementations for relaxing logical operators.
"""

import torch
from abc import ABC, abstractmethod


class TNorm(ABC):
    """Base class for t-norms (continuous relaxations of logical operators)."""
    
    @abstractmethod
    def conjunction(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Relaxed AND operation."""
        pass
    
    @abstractmethod
    def disjunction(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Relaxed OR operation."""
        pass
    
    def negation(self, a: torch.Tensor) -> torch.Tensor:
        """Relaxed NOT operation (standard across all t-norms)."""
        return 1.0 - a
    
    def implication(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Relaxed IMPLIES operation: a → b ≡ ¬a ∨ b."""
        return self.disjunction(self.negation(a), b)
    
    def equivalence(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Relaxed EQUIVALENCE operation: a ↔ b ≡ (a → b) ∧ (b → a)."""
        return self.conjunction(
            self.implication(a, b),
            self.implication(b, a)
        )


class ProductTNorm(TNorm):
    """
    Product t-norm (probabilistic semantics).
    
    - AND: a * b
    - OR: a + b - a * b
    
    This is the most commonly used t-norm in neural-symbolic learning.
    """
    
    def conjunction(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b
    
    def disjunction(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b - a * b


class LukasiewiczTNorm(TNorm):
    """
    Łukasiewicz t-norm (bounded difference).
    
    - AND: max(0, a + b - 1)
    - OR: min(1, a + b)
    
    Good for enforcing stricter logical constraints.
    """
    
    def conjunction(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.clamp(a + b - 1.0, min=0.0)
    
    def disjunction(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.clamp(a + b, max=1.0)


class GodelTNorm(TNorm):
    """
    Gödel t-norm (minimum/maximum).
    
    - AND: min(a, b)
    - OR: max(a, b)
    
    Most conservative option, but can have gradient issues.
    """
    
    def conjunction(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.minimum(a, b)
    
    def disjunction(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.maximum(a, b)
