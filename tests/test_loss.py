"""Tests for loss computation and reduction modes.

This module tests the loss() method and its different reduction modes
(mean, sum, none).
"""

import sympy as sp
import torch

from pysignet import Predicate, Symbol, Variable, compile_logic


def test_loss_mean_reduction() -> None:
    """Test loss with mean reduction (default)."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(10, 5)

    # Mean reduction (default) with linear post-processing
    loss = logic_loss.loss(x, reduction="mean", post_processing="linear")

    assert loss.shape == ()  # Scalar
    assert loss >= 0.0
    assert loss <= 1.0


def test_loss_sum_reduction() -> None:
    """Test loss with sum reduction."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(10, 5)

    # Sum reduction
    loss = logic_loss.loss(x, reduction="sum")

    assert loss.shape == ()  # Scalar
    assert loss >= 0.0


def test_loss_none_reduction() -> None:
    """Test loss with no reduction."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(10, 5)

    # No reduction with linear post-processing
    loss = logic_loss.loss(x, reduction="none", post_processing="linear")

    assert loss.shape == (10,)  # Per-sample losses
    assert loss.min() >= 0.0
    assert loss.max() <= 1.0


def test_loss_reduction_consistency() -> None:
    """Test that different reductions are mathematically consistent."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(10, 5)

    # Get all three reduction modes
    loss_mean = logic_loss.loss(x, reduction="mean")
    loss_sum = logic_loss.loss(x, reduction="sum")
    loss_none = logic_loss.loss(x, reduction="none")

    # sum = mean * batch_size
    assert torch.allclose(loss_sum, loss_mean * 10)

    # mean = average of per-sample losses
    assert torch.allclose(loss_mean, loss_none.mean())

    # sum = sum of per-sample losses
    assert torch.allclose(loss_sum, loss_none.sum())


def test_invalid_reduction_mode() -> None:
    """Test that invalid reduction mode raises ValueError."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(5, 3)

    try:
        logic_loss.loss(x, reduction="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown reduction" in str(e)


def test_loss_is_one_minus_satisfaction() -> None:
    """Test that loss = 1 - satisfaction with linear post-processing."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(10, 5)

    satisfaction = logic_loss(x)
    loss_none = logic_loss.loss(x, reduction="none", post_processing="linear")

    # loss = 1 - satisfaction (with linear post-processing)
    assert torch.allclose(loss_none, 1.0 - satisfaction)


def test_loss_with_perfect_satisfaction() -> None:
    """Test loss with perfect satisfaction (satisfaction = 1.0)."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    # Predicate always returns 1.0
    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]))}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(5, 3)

    loss = logic_loss.loss(x, reduction="mean")

    # Perfect satisfaction => loss = 0
    assert torch.allclose(loss, torch.tensor(0.0))


def test_loss_with_zero_satisfaction() -> None:
    """Test loss with zero satisfaction (satisfaction = 0.0)."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    # Predicate always returns 0.0
    predicates = {"P": Predicate(lambda x: torch.zeros(x.shape[0]))}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(5, 3)

    loss = logic_loss.loss(x, reduction="mean", post_processing="linear")

    # Zero satisfaction => loss = 1 (with linear post-processing)
    assert torch.allclose(loss, torch.tensor(1.0))


def test_loss_with_complex_expression() -> None:
    """Test loss computation with complex expressions."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P, Q, R = Symbol("P Q R")
    expr = sp.And(sp.Or(P(X), Q(X)), sp.Not(R(X)))

    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5),
        "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5),
        "R": Predicate(lambda x: torch.ones(x.shape[0]) * 0.3),
    }

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(10, 5)

    # Compute satisfaction manually
    # (P | Q) & ~R
    # P | Q = 0.5 + 0.5 - 0.25 = 0.75
    # ~R = 0.7
    # Result = 0.75 * 0.7 = 0.525
    p_or_q = 0.5 + 0.5 - 0.5 * 0.5
    not_r = 0.7
    expected_satisfaction = p_or_q * not_r
    expected_loss = 1.0 - expected_satisfaction

    loss = logic_loss.loss(x, reduction="mean", post_processing="linear")
    assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-5)


def test_loss_default_reduction() -> None:
    """Test that default reduction is 'mean'."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(10, 5)

    # Default should be mean
    loss_default = logic_loss.loss(x)
    loss_mean = logic_loss.loss(x, reduction="mean")

    assert torch.allclose(loss_default, loss_mean)
    assert loss_default.shape == ()


def test_loss_with_varying_batch_sizes() -> None:
    """Test loss computation with different batch sizes."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8)}

    logic_loss = compile_logic(expr, predicates)

    # Test different batch sizes with linear post-processing
    for batch_size in [1, 5, 10, 100]:
        x = torch.randn(batch_size, 5)

        loss_mean = logic_loss.loss(x, reduction="mean", post_processing="linear")
        loss_sum = logic_loss.loss(x, reduction="sum", post_processing="linear")
        loss_none = logic_loss.loss(x, reduction="none", post_processing="linear")

        assert loss_mean.shape == ()
        assert loss_sum.shape == ()
        assert loss_none.shape == (batch_size,)

        # Mean should be constant (0.2) regardless of batch size
        assert torch.allclose(loss_mean, torch.tensor(0.2))

        # Sum should scale with batch size
        assert torch.allclose(loss_sum, torch.tensor(0.2 * batch_size))


def test_loss_with_dict_input() -> None:
    """Test loss computation with dict input."""
    X, Y = Variable("X Y")
    # pylint: disable=invalid-name
    P, Q = Symbol("P Q")
    expr = sp.And(P(X), Q(Y))

    predicates = {
        "P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1))),
        "Q": Predicate(lambda x: torch.sigmoid(x.mean(dim=-1))),
    }

    logic_loss = compile_logic(expr, predicates)

    batch_size = 10
    inputs = {
        "X": torch.randn(batch_size, 5),
        "Y": torch.randn(batch_size, 10),
    }

    loss = logic_loss.loss(inputs, reduction="mean", post_processing="linear")

    assert loss.shape == ()
    assert loss >= 0.0
    assert loss <= 1.0


def test_loss_numerics_stability() -> None:
    """Test loss computation with extreme satisfaction values."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = P(X)

    # Very high satisfaction (near 1)
    predicates_high = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.9999999)}
    logic_loss_high = compile_logic(expr, predicates_high)
    x = torch.randn(5, 3)
    loss_high = logic_loss_high.loss(x, reduction="mean", post_processing="linear")

    # Loss should be very small (near 0) with linear post-processing
    assert loss_high >= 0.0
    assert loss_high < 1e-6

    # Very low satisfaction (near 0)
    predicates_low = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 1e-7)}
    logic_loss_low = compile_logic(expr, predicates_low)
    loss_low = logic_loss_low.loss(x, reduction="mean", post_processing="linear")

    # Loss should be very high (near 1) with linear post-processing
    assert loss_low > 0.999999
    assert loss_low <= 1.0
