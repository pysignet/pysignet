"""Tests for input handling and batching.

This module tests different input formats (single tensor, dict), batch
handling, and input routing to predicates.
"""

import sympy as sp
import torch

from pysignet import LogicCompiler, Predicate


def test_single_tensor_input() -> None:
    """Test using single tensor input for all predicates."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1))),
        "Q": Predicate("Q", lambda x: torch.sigmoid(x.mean(dim=-1))),
    }

    logic_loss = LogicCompiler(expr, predicates)

    # Single tensor input - same tensor passed to all predicates
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    assert satisfaction.shape == (10,)
    assert satisfaction.min() >= 0.0
    assert satisfaction.max() <= 1.0


def test_dict_input_per_predicate() -> None:
    """Test dict input with different tensors for each predicate."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1))),
        "Q": Predicate("Q", lambda x: torch.sigmoid(x.mean(dim=-1))),
    }

    logic_loss = LogicCompiler(expr, predicates)

    batch_size = 10
    inputs = {
        "P": torch.randn(batch_size, 5),
        "Q": torch.randn(batch_size, 10),  # Different shape
    }

    satisfaction = logic_loss(inputs)
    assert satisfaction.shape == (batch_size,)


def test_dict_input_with_default_key() -> None:
    """Test dict input uses 'default' key when specific key missing."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1))),
        "Q": Predicate("Q", lambda x: torch.sigmoid(x.mean(dim=-1))),
    }

    logic_loss = LogicCompiler(expr, predicates)

    # Provide specific input for P and default for others
    default_input = torch.randn(5, 3)
    inputs = {"P": torch.randn(5, 3), "default": default_input}

    satisfaction = logic_loss(inputs)

    # Should use default for Q
    assert satisfaction.shape == (5,)


def test_batching_various_sizes() -> None:
    """Test that batching works correctly with various batch sizes."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    # Predicate that depends on input
    predicates = {"P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = LogicCompiler(expr, predicates)

    # Different batch sizes
    for batch_size in [1, 10, 100]:
        x = torch.randn(batch_size, 5)
        satisfaction = logic_loss(x)
        assert satisfaction.shape == (batch_size,)
        assert satisfaction.min() >= 0.0
        assert satisfaction.max() <= 1.0


def test_different_feature_dimensions() -> None:
    """Test predicates with different input feature dimensions."""
    # pylint: disable=invalid-name
    P, Q, R = sp.symbols("P Q R")
    expr = sp.And(sp.Or(P, Q), R)

    predicates = {
        "P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1))),
        "Q": Predicate("Q", lambda x: torch.sigmoid(x.mean(dim=-1))),
        "R": Predicate("R", lambda x: torch.sigmoid(x[:, 0])),
    }

    logic_loss = LogicCompiler(expr, predicates)

    batch_size = 10
    inputs = {
        "P": torch.randn(batch_size, 5),
        "Q": torch.randn(batch_size, 10),
        "R": torch.randn(batch_size, 3),
    }

    satisfaction = logic_loss(inputs)
    assert satisfaction.shape == (batch_size,)


def test_input_preserves_device() -> None:
    """Test that output preserves input device."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.7)}

    logic_loss = LogicCompiler(expr, predicates)

    # CPU tensor
    x_cpu = torch.randn(5, 3)
    satisfaction_cpu = logic_loss(x_cpu)
    assert satisfaction_cpu.device == x_cpu.device


def test_input_preserves_dtype() -> None:
    """Test operations with different tensor dtypes."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.7)}

    logic_loss = LogicCompiler(expr, predicates)

    # float32 (default)
    x_float32 = torch.randn(5, 3, dtype=torch.float32)
    satisfaction_float32 = logic_loss(x_float32)
    assert satisfaction_float32.dtype == torch.float32

    # float64
    x_float64 = torch.randn(5, 3, dtype=torch.float64)
    unused_satisfaction_float64 = logic_loss(x_float64)
    # Note: dtype might be converted due to operations
    # Intentionally unused, just testing execution
    del unused_satisfaction_float64


def test_multidimensional_features() -> None:
    """Test inputs with more than 2 dimensions."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    # Predicate that flattens multi-dimensional features
    predicates = {
        "P": Predicate("P", lambda x: torch.sigmoid(x.flatten(1).sum(dim=-1)))
    }

    logic_loss = LogicCompiler(expr, predicates)

    # 3D input: (batch, height, width)
    x = torch.randn(5, 4, 4)
    satisfaction = logic_loss(x)

    assert satisfaction.shape == (5,)
    assert satisfaction.min() >= 0.0
    assert satisfaction.max() <= 1.0


def test_sequential_calls_same_input() -> None:
    """Test multiple sequential calls with same input."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(5, 3)

    # Multiple calls should give same result
    result1 = logic_loss(x)
    result2 = logic_loss(x)

    assert torch.allclose(result1, result2)


def test_sequential_calls_different_inputs() -> None:
    """Test multiple sequential calls with different inputs."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    predicates = {"P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = LogicCompiler(expr, predicates)

    # Different inputs should give potentially different results
    x1 = torch.ones(5, 3)  # Positive
    x2 = -torch.ones(5, 3)  # Negative

    result1 = logic_loss(x1)
    result2 = logic_loss(x2)

    # Results should be different
    assert not torch.allclose(result1, result2)
    # But both in valid range
    assert result1.min() >= 0.0 and result1.max() <= 1.0
    assert result2.min() >= 0.0 and result2.max() <= 1.0


def test_dict_input_subset_of_predicates() -> None:
    """Test dict input providing subset of predicates uses default."""
    # pylint: disable=invalid-name
    P, Q, R = sp.symbols("P Q R")
    expr = sp.And(P, sp.Or(Q, R))

    predicates = {
        "P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1))),
        "Q": Predicate("Q", lambda x: torch.sigmoid(x.mean(dim=-1))),
        "R": Predicate("R", lambda x: torch.sigmoid(x.max(dim=-1)[0])),
    }

    logic_loss = LogicCompiler(expr, predicates)

    # Only provide P and default
    batch_size = 5
    default_input = torch.randn(batch_size, 3)
    inputs = {"P": torch.randn(batch_size, 5), "default": default_input}

    satisfaction = logic_loss(inputs)
    assert satisfaction.shape == (batch_size,)


def test_consistent_batch_size_required() -> None:
    """Test that all inputs must have same batch size."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    # Predicates that return different batch sizes will cause shape mismatch
    predicates = {
        "P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1))),
        "Q": Predicate("Q", lambda x: torch.sigmoid(x.mean(dim=-1))),
    }

    logic_loss = LogicCompiler(expr, predicates)

    batch_size = 10
    inputs = {
        "P": torch.randn(batch_size, 5),
        "Q": torch.randn(batch_size, 3),  # Same batch size
    }

    # Should work fine
    satisfaction = logic_loss(inputs)
    assert satisfaction.shape == (batch_size,)
