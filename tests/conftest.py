"""Shared pytest fixtures for pysignet tests.

This module provides reusable fixtures for testing the pysignet library.
Fixtures include common predicates, sample expressions, and test inputs.
"""

from typing import Dict

import pytest
import sympy as sp
import torch
import torch.nn as nn

from pysignet import Predicate


@pytest.fixture
def simple_predicates() -> Dict[str, Predicate]:
    """Simple deterministic predicates with constant values.

    Returns:
        Dictionary mapping predicate names to Predicate instances
        with fixed satisfaction values.
    """
    return {
        "P": Predicate( lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate( lambda x: torch.ones(x.shape[0]) * 0.6),
        "R": Predicate( lambda x: torch.ones(x.shape[0]) * 0.5),
    }


@pytest.fixture
def neural_predicates() -> Dict[str, Predicate]:
    """Neural network-based predicates.

    Returns:
        Dictionary mapping predicate names to Predicate instances
        wrapping simple neural networks.
    """
    model_p = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
    model_q = nn.Sequential(
        nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 1), nn.Sigmoid()
    )

    return {
        "P": Predicate( lambda x: model_p(x).squeeze(-1)),
        "Q": Predicate( lambda x: model_q(x).squeeze(-1)),
    }


@pytest.fixture
def dynamic_predicates() -> Dict[str, Predicate]:
    """Input-dependent predicates.

    Returns:
        Dictionary of predicates that compute values based on input.
    """
    return {
        "P": Predicate( lambda x: torch.sigmoid(x.sum(dim=-1))),
        "Q": Predicate( lambda x: torch.sigmoid(x.mean(dim=-1))),
        "R": Predicate( lambda x: (x > 0).float().mean(dim=-1)),
    }


@pytest.fixture
def batch_inputs() -> Dict[str, torch.Tensor]:
    """Various batch sizes for testing.

    Returns:
        Dictionary mapping size names to random tensors.
    """
    return {
        "empty": torch.randn(0, 5),
        "single": torch.randn(1, 5),
        "small": torch.randn(5, 5),
        "medium": torch.randn(32, 5),
        "large": torch.randn(100, 5),
    }


@pytest.fixture
def special_values() -> Dict[str, torch.Tensor]:
    """Tensors with special values (NaN, Inf, etc.).

    Returns:
        Dictionary of tensors containing edge case values.
    """
    return {
        "nan": torch.tensor([[float("nan"), 1.0, 2.0]]),
        "pos_inf": torch.tensor([[float("inf"), 1.0, 2.0]]),
        "neg_inf": torch.tensor([[float("-inf"), 1.0, 2.0]]),
        "zeros": torch.zeros(5, 3),
        "ones": torch.ones(5, 3),
    }
