"""Base class for logic compilation strategies."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Union

import sympy as sp
import torch

from ..predicate import Predicate


class LogicCompiler(ABC):
    """Abstract base class for compiling logic expressions into differentiable
    computations.

    LogicCompiler defines the interface for different compilation strategies
    (t-norms, semantic loss, etc.). Each strategy compiles a SymPy logic
    expression into a PyTorch callable that returns satisfaction degrees.

    The compiled callable can be used directly or wrapped in a LogicLoss for
    loss computation.
    """

    @abstractmethod
    def compile(
            self,
            expr: sp.Basic,
            predicates: Dict[str, Predicate],
    ) -> Callable[[Union[torch.Tensor, Dict[str, torch.Tensor]]], torch.Tensor]:
        """Compile a logic expression into a differentiable callable.

        Args:
            expr: SymPy logic expression (e.g., sp.And(P, sp.Or(Q, sp.Not(R))))
            predicates: Dict mapping predicate names to Predicate objects

        Returns:
            Callable that takes inputs and returns satisfaction tensor of
            shape (batch_size,) with values in [0, 1].

        Raises:
            ValueError: If symbols in expr have no corresponding predicates
        """
        pass
