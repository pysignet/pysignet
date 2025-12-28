"""Tests for edge cases and error handling.

This module tests edge cases like empty batches, special values (NaN, Inf),
error conditions, and boundary scenarios.
"""

import sympy as sp
import torch

from logic_as_loss import LogicCompiler, Predicate


def test_empty_batch() -> None:
    """Test handling of empty batches (batch_size=0)."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    predicates = {"P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(0, 5)  # Empty batch
    satisfaction = logic_loss(x)

    assert satisfaction.shape == (0,)
    assert satisfaction.numel() == 0


def test_single_element_batch() -> None:
    """Test handling of single-element batches."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.7),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.5),
    }

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(1, 5)  # Single element
    satisfaction = logic_loss(x)

    assert satisfaction.shape == (1,)
    assert torch.allclose(satisfaction, torch.tensor([0.35]), atol=1e-5)


def test_very_large_batch() -> None:
    """Test handling of very large batches (10000+ elements)."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    predicates = {"P": Predicate("P", lambda x: torch.sigmoid(x.mean(dim=-1)))}

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(10000, 5)
    satisfaction = logic_loss(x)

    assert satisfaction.shape == (10000,)
    assert satisfaction.min() >= 0.0
    assert satisfaction.max() <= 1.0


def test_nan_handling() -> None:
    """Test handling of NaN values in inputs."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    predicates = {"P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.tensor([[float("nan"), 1.0, 2.0]])
    satisfaction = logic_loss(x)

    # sigmoid(nan) = nan, clamped to [0,1] doesn't fix NaN
    assert satisfaction.shape == (1,)
    # NaN should propagate through
    assert torch.isnan(satisfaction).any() or (
        satisfaction.min() >= 0.0 and satisfaction.max() <= 1.0
    )


def test_inf_handling() -> None:
    """Test handling of Inf values in inputs."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    predicates = {"P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = LogicCompiler(expr, predicates)

    # Test positive infinity
    x_pos_inf = torch.tensor([[float("inf"), 1.0, 2.0]])
    satisfaction_pos = logic_loss(x_pos_inf)
    assert satisfaction_pos.shape == (1,)
    # sigmoid(+inf) = 1.0
    assert torch.allclose(satisfaction_pos, torch.tensor([1.0]), atol=1e-5)

    # Test negative infinity
    x_neg_inf = torch.tensor([[float("-inf"), 1.0, 2.0]])
    satisfaction_neg = logic_loss(x_neg_inf)
    assert satisfaction_neg.shape == (1,)
    # sigmoid(-inf) = 0.0
    assert torch.allclose(satisfaction_neg, torch.tensor([0.0]), atol=1e-5)


def test_missing_predicate_raises_error() -> None:
    """Test that missing predicates raise ValueError."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    # Only provide predicate for P, not Q
    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.5)}

    try:
        LogicCompiler(expr, predicates)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Missing predicates" in str(e)
        assert "Q" in str(e)


def test_unsupported_expression_raises_error() -> None:
    """Test that unsupported SymPy expressions raise ValueError."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")

    # Use an unsupported operation
    expr = P + P  # Arithmetic, not logic

    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.5)}

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(5, 3)

    try:
        logic_loss(x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported expression type" in str(e)


def test_zero_dimension_input() -> None:
    """Test handling of inputs with zero features."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    # Predicate that doesn't depend on input dimension
    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.5)}

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(5, 0)  # 5 samples, 0 features
    satisfaction = logic_loss(x)

    assert satisfaction.shape == (5,)
    assert torch.allclose(satisfaction, torch.tensor(0.5))


def test_very_small_values() -> None:
    """Test handling of very small predicate values (near 0)."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 1e-10),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 1e-10),
    }

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # Product t-norm: 1e-10 * 1e-10 = 1e-20
    assert satisfaction.min() >= 0.0
    assert satisfaction.max() <= 1.0
    assert torch.allclose(satisfaction, torch.tensor(1e-20), atol=1e-25)


def test_values_near_one() -> None:
    """Test handling of predicate values very close to 1."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.9999999),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.9999999),
    }

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    assert satisfaction.min() >= 0.0
    assert satisfaction.max() <= 1.0
    expected = 0.9999999 * 0.9999999
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-6)


def test_deeply_nested_expression() -> None:
    """Test handling of deeply nested logical expressions."""
    # pylint: disable=invalid-name
    P, Q, R, S = sp.symbols("P Q R S")

    # Create a deeply nested expression
    expr = sp.And(
        sp.Or(sp.And(P, Q), sp.Not(R)), sp.Implies(S, sp.Or(P, sp.Not(Q)))
    )

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
        "R": Predicate("R", lambda x: torch.ones(x.shape[0]) * 0.5),
        "S": Predicate("S", lambda x: torch.ones(x.shape[0]) * 0.7),
    }

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # Should compute without error
    assert satisfaction.shape == (10,)
    assert satisfaction.min() >= 0.0
    assert satisfaction.max() <= 1.0


def test_many_predicates() -> None:
    """Test handling expressions with many predicates."""
    # pylint: disable=invalid-name
    symbols_list = sp.symbols("P Q R S T U V W X Y Z")

    # Create expression using all predicates
    expr = sp.And(*symbols_list[:10])

    predicates = {
        str(sym): Predicate(str(sym), lambda x: torch.ones(x.shape[0]) * 0.9)
        for sym in symbols_list[:10]
    }

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # Product t-norm: 0.9^10 ≈ 0.3487
    assert satisfaction.shape == (5,)
    expected = 0.9**10
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-4)


def test_single_predicate_expression() -> None:
    """Test expression with single predicate (no operators)."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P  # Just the symbol itself

    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.75)}

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # Should just return predicate value
    assert torch.allclose(satisfaction, torch.tensor(0.75))


def test_mixed_batch_dimensions() -> None:
    """Test with varying batch sizes across calls."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    predicates = {"P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = LogicCompiler(expr, predicates)

    # Test different batch sizes sequentially
    for batch_size in [1, 5, 10, 100]:
        x = torch.randn(batch_size, 5)
        satisfaction = logic_loss(x)
        assert satisfaction.shape == (batch_size,)
        assert satisfaction.min() >= 0.0
        assert satisfaction.max() <= 1.0
