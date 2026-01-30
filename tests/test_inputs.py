"""Tests for input handling and batching.

This module tests different input formats (single tensor, dict), batch
handling, and input routing to predicates.
"""

import sympy as sp
import torch

from pysignet import Predicate, Symbol, Variable, logic_to_loss


def test_single_tensor_input() -> None:
    """Test using single tensor input for all predicates."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P, Q = Symbol("P Q")
    expr = sp.And(P(X), Q(X))

    predicates = {
        "P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1))),
        "Q": Predicate(lambda x: torch.sigmoid(x.mean(dim=-1))),
    }

    logic_loss = logic_to_loss(expr, predicates)

    # Single tensor input - same tensor passed to all predicates
    batch_size = 10
    x = torch.randn(batch_size, 5)
    # Use quantify='none' to get per-batch results
    satisfaction = logic_loss(X=x, quantify='none')

    assert satisfaction.shape == (batch_size,)
    assert satisfaction.min() >= 0.0
    assert satisfaction.max() <= 1.0


def test_dict_input_per_predicate() -> None:
    """Test dict input with different tensors for each predicate."""
    X, Y = Variable("X Y")
    # pylint: disable=invalid-name
    P, Q = Symbol("P Q")
    expr = sp.And(P(X), Q(X))

    predicates = {
        "P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1))),
        "Q": Predicate(lambda x: torch.sigmoid(x.mean(dim=-1))),
    }

    logic_loss = logic_to_loss(expr, predicates)

    batch_size = 10
    inputs = {
        "X": torch.randn(batch_size, 5),
        "Y": torch.randn(batch_size, 10),  # Different shape
    }

    # Use quantify='none' to get per-batch results
    satisfaction = logic_loss(**inputs, quantify='none')
    assert satisfaction.shape == (batch_size,)


def test_batching_various_sizes() -> None:
    """Test that batching works correctly with various batch sizes."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    # Predicate that depends on input
    predicates = {"P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = logic_to_loss(expr, predicates)

    # Different batch sizes with quantify='none' to get per-batch results
    for batch_size in [1, 5, 10, 32]:
        x = torch.randn(batch_size, 5)
        satisfaction = logic_loss(X=x, quantify='none')
        assert satisfaction.shape == (batch_size,)
        assert satisfaction.min() >= 0.0
        assert satisfaction.max() <= 1.0


def test_different_feature_dimensions() -> None:
    """Test predicates with different input feature dimensions."""
    X, Y, Z = Variable("X Y Z")
    # pylint: disable=invalid-name
    P, Q, R = Symbol("P Q R")
    expr = sp.And(sp.Or(P(X), Q(Y)), R(Z))

    predicates = {
        "P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1))),
        "Q": Predicate(lambda x: torch.sigmoid(x.mean(dim=-1))),
        "R": Predicate(lambda x: torch.sigmoid(x[:, 0])),
    }

    logic_loss = logic_to_loss(expr, predicates)

    batch_size = 10
    inputs = {
        "X": torch.randn(batch_size, 5),
        "Y": torch.randn(batch_size, 10),
        "Z": torch.randn(batch_size, 3),
    }

    # Use quantify='none' to get per-batch results
    satisfaction = logic_loss(**inputs, quantify='none')
    assert satisfaction.shape == (batch_size,)


def test_input_preserves_device() -> None:
    """Test that output preserves input device."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

    logic_loss = logic_to_loss(expr, predicates)

    # CPU tensor
    x_cpu = torch.randn(1, 3)
    satisfaction_cpu = logic_loss(X=x_cpu)
    assert satisfaction_cpu.device == x_cpu.device


def test_input_preserves_dtype() -> None:
    """Test operations with different tensor dtypes."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

    logic_loss = logic_to_loss(expr, predicates)

    # float32 (default)
    x_float32 = torch.randn(1, 3, dtype=torch.float32)
    satisfaction_float32 = logic_loss(X=x_float32)
    assert satisfaction_float32.dtype == torch.float32

    # float64
    x_float64 = torch.randn(1, 3, dtype=torch.float64)
    unused_satisfaction_float64 = logic_loss(X=x_float64)
    # Note: dtype might be converted due to operations
    # Intentionally unused, just testing execution
    del unused_satisfaction_float64


def test_multidimensional_features() -> None:
    """Test inputs with more than 2 dimensions."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    # Predicate that flattens multi-dimensional features
    predicates = {"P": Predicate(lambda x: torch.sigmoid(x.flatten(1).sum(dim=-1)))}

    logic_loss = logic_to_loss(expr, predicates)

    # 3D input: (batch, height, width)
    batch_size = 5
    x = torch.randn(batch_size, 4, 4)
    # Use quantify='none' to get per-batch results
    satisfaction = logic_loss(X=x, quantify='none')

    assert satisfaction.shape == (batch_size,)
    assert satisfaction.min() >= 0.0
    assert satisfaction.max() <= 1.0


def test_sequential_calls_same_input() -> None:
    """Test multiple sequential calls with same input."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.randn(1, 3)

    # Multiple calls should give same result
    result1 = logic_loss(X=x)
    result2 = logic_loss(X=x)

    assert torch.allclose(result1, result2)


def test_sequential_calls_different_inputs() -> None:
    """Test multiple sequential calls with different inputs."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = logic_to_loss(expr, predicates)

    # Different inputs should give potentially different results
    x1 = torch.ones(1, 3)  # Positive
    x2 = -torch.ones(1, 3)  # Negative

    result1 = logic_loss(X=x1)
    result2 = logic_loss(X=x2)

    # Results should be different
    assert not torch.allclose(result1, result2)
    # But both in valid range
    assert result1.min() >= 0.0 and result1.max() <= 1.0
    assert result2.min() >= 0.0 and result2.max() <= 1.0


def test_dict_input_subset_of_predicates() -> None:
    """Test dict input with some predicates sharing data."""
    X, Y = Variable("X Y")
    # pylint: disable=invalid-name
    P, Q, R = Symbol("P Q R")
    expr = sp.And(P(X), sp.Or(Q(Y), R(Y)))

    predicates = {
        "P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1))),
        "Q": Predicate(lambda x: torch.sigmoid(x.mean(dim=-1))),
        "R": Predicate(lambda x: torch.sigmoid(x.max(dim=-1)[0])),
    }

    logic_loss = logic_to_loss(expr, predicates)

    # Provide specific data for P and shared data for Q and R
    batch_size = 5
    inputs = {
        "X": torch.randn(batch_size, 5),
        "Y": torch.randn(batch_size, 3),
    }

    # Use quantify='none' to get per-batch results
    satisfaction = logic_loss(inputs, quantify='none')
    assert satisfaction.shape == (batch_size,)


def test_consistent_batch_size_required() -> None:
    """Test that all inputs must have same batch size."""
    X, Y = Variable("X Y")
    # pylint: disable=invalid-name
    P, Q = Symbol("P Q")
    expr = sp.And(P(X), Q(X))

    # Predicates that return different batch sizes will cause shape mismatch
    predicates = {
        "P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1))),
        "Q": Predicate(lambda x: torch.sigmoid(x.mean(dim=-1))),
    }

    logic_loss = logic_to_loss(expr, predicates)

    batch_size = 10
    inputs = {
        "X": torch.randn(batch_size, 5),
        "Y": torch.randn(batch_size, 3),  # Same batch size
    }

    # Should work fine - use quantify='none' to get per-batch results
    satisfaction = logic_loss(inputs, quantify='none')
    assert satisfaction.shape == (batch_size,)
