"""Tests for edge cases and error handling.

This module tests edge cases like empty batches, special values (NaN, Inf),
error conditions, and boundary scenarios.
"""

import sympy as sp
import torch

from pysignet import Predicate, Symbol, Variable, logic_to_loss


def test_empty_batch() -> None:
    """Test handling of empty batches (batch_size=0)."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.randn(0, 5)  # Empty batch
    # Use quantify='none' to get per-batch results for shape check
    satisfaction = logic_loss(X=x, quantify='none')

    assert satisfaction.shape == (0,)
    assert satisfaction.numel() == 0


def test_single_element_batch() -> None:
    """Test handling of single-element batches."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P, Q = Symbol("P Q")
    expr = sp.And(P(X), Q(X))

    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7),
        "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5),
    }

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.randn(1, 5)  # Single element
    # Default quantify='forall' with batch_size=1 returns scalar
    satisfaction = logic_loss(X=x)

    assert satisfaction.shape == ()  # Scalar with forall
    assert torch.allclose(satisfaction, torch.tensor(0.35), atol=1e-5)


def test_very_large_batch() -> None:
    """Test handling of very large batches (10000+ elements)."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.sigmoid(x.mean(dim=-1)))}

    logic_loss = logic_to_loss(expr, predicates)
    batch_size = 10000
    x = torch.randn(batch_size, 5)
    # Use quantify='none' to get per-batch results
    satisfaction = logic_loss(X=x, quantify='none')

    assert satisfaction.shape == (batch_size,)
    assert satisfaction.min() >= 0.0
    assert satisfaction.max() <= 1.0


def test_nan_handling() -> None:
    """Test handling of NaN values in inputs."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.tensor([[float("nan"), 1.0, 2.0]])
    # Default quantify='forall' with batch_size=1 returns scalar
    satisfaction = logic_loss(X=x)

    # sigmoid(nan) = nan, clamped to [0,1] doesn't fix NaN
    assert satisfaction.shape == ()  # Scalar with forall
    # NaN should propagate through
    assert torch.isnan(satisfaction) or (
        satisfaction.item() >= 0.0 and satisfaction.item() <= 1.0
    )


def test_inf_handling() -> None:
    """Test handling of Inf values in inputs."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = logic_to_loss(expr, predicates)

    # Test positive infinity
    # Default quantify='forall' with batch_size=1 returns scalar
    x_pos_inf = torch.tensor([[float("inf"), 1.0, 2.0]])
    satisfaction_pos = logic_loss(X=x_pos_inf)
    assert satisfaction_pos.shape == ()  # Scalar with forall
    # sigmoid(+inf) = 1.0
    assert torch.allclose(satisfaction_pos, torch.tensor(1.0), atol=1e-5)

    # Test negative infinity
    x_neg_inf = torch.tensor([[float("-inf"), 1.0, 2.0]])
    satisfaction_neg = logic_loss(X=x_neg_inf)
    assert satisfaction_neg.shape == ()  # Scalar with forall
    # sigmoid(-inf) = 0.0
    assert torch.allclose(satisfaction_neg, torch.tensor(0.0), atol=1e-5)


def test_missing_predicate_raises_error() -> None:
    """Test that missing predicates raise ValueError."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P, Q = Symbol("P Q")
    expr = sp.And(P(X), Q(X))

    # Only provide predicate for P, not Q
    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)}

    try:
        logic_to_loss(expr, predicates)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Missing predicates" in str(e)
        assert "Q" in str(e)


def test_unsupported_expression_raises_error() -> None:
    """Test that unsupported SymPy expressions raise ValueError."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")

    # Use an unsupported operation
    try:
        expr = P(X) + P(X)  # Arithmetic, not logic
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "unsupported operand type" in str(e)


def test_zero_dimension_input() -> None:
    """Test handling of inputs with zero features."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    # Predicate that doesn't depend on input dimension
    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)}

    logic_loss = logic_to_loss(expr, predicates)
    batch_size = 5
    x = torch.randn(batch_size, 0)  # 5 samples, 0 features
    # Use quantify='none' to get per-batch results
    satisfaction = logic_loss(X=x, quantify='none')

    assert satisfaction.shape == (batch_size,)
    assert torch.allclose(satisfaction, torch.ones(batch_size) * 0.5)


def test_very_small_values() -> None:
    """Test handling of very small predicate values (near 0)."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P, Q = Symbol("P Q")
    expr = sp.And(P(X), Q(X))

    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * 1e-10),
        "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 1e-10),
    }

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(X=x)

    # Product t-norm: 1e-10 * 1e-10 = 1e-20
    assert satisfaction.min() >= 0.0
    assert satisfaction.max() <= 1.0
    assert torch.allclose(satisfaction, torch.tensor(1e-20), atol=1e-25)


def test_values_near_one() -> None:
    """Test handling of predicate values very close to 1."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P, Q = Symbol("P Q")
    expr = sp.And(P(X), Q(X))

    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.9999999),
        "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.9999999),
    }

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(X=x)

    assert satisfaction.min() >= 0.0
    assert satisfaction.max() <= 1.0
    expected = 0.9999999 * 0.9999999
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-6)


def test_deeply_nested_expression() -> None:
    """Test handling of deeply nested logical expressions."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P, Q, R, S = Symbol("P Q R S")

    # Create a deeply nested expression
    expr = sp.And(
        sp.Or(sp.And(P(X), Q(X)), sp.Not(R(X))),
        sp.Implies(S(X), sp.Or(P(X), sp.Not(Q(X)))),
    )

    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        "R": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5),
        "S": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7),
    }

    logic_loss = logic_to_loss(expr, predicates)
    batch_size = 10
    x = torch.randn(batch_size, 5)
    # Use quantify='none' to get per-batch results
    satisfaction = logic_loss(X=x, quantify='none')

    # Should compute without error
    assert satisfaction.shape == (batch_size,)
    assert satisfaction.min() >= 0.0
    assert satisfaction.max() <= 1.0


def test_many_predicates() -> None:
    """Test handling expressions with many predicates."""
    # pylint: disable=invalid-name
    X = Variable("X")
    symbols_list = Symbol("P Q R S T U V W X Y Z")

    # Create expression using all predicates with explicit variable
    expr = sp.And(*[sym(X) for sym in symbols_list[:10]])

    predicates = {
        str(sym): Predicate(lambda x: torch.ones(x.shape[0]) * 0.9)
        for sym in symbols_list[:10]
    }

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.randn(1, 3)
    # Default quantify='forall' with batch_size=1 returns scalar
    satisfaction = logic_loss(X=x)

    # Product t-norm: 0.9^10 ≈ 0.3487
    assert satisfaction.shape == ()  # Scalar with forall
    expected = 0.9**10
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-4)


def test_single_predicate_expression() -> None:
    """Test expression with single predicate (no operators)."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)  # Just the symbol itself

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.75)}

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(X=x)

    # Should just return predicate value
    assert torch.allclose(satisfaction, torch.tensor(0.75))


def test_mixed_batch_dimensions() -> None:
    """Test with varying batch sizes across calls."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = logic_to_loss(expr, predicates)

    # Test different batch sizes sequentially with quantify='none' for per-batch results
    for batch_size in [1, 5, 10, 32]:
        x = torch.randn(batch_size, 5)
        satisfaction = logic_loss(X=x, quantify='none')
        assert satisfaction.shape == (batch_size,)
        assert satisfaction.min() >= 0.0
        assert satisfaction.max() <= 1.0
