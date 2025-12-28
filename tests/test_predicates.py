"""Tests for Predicate class behavior.

This module tests the Predicate wrapper class, including deterministic
vs. model predicates, return value handling, and value clamping.
"""

import sympy as sp
import torch
import torch.nn as nn

from logic_as_loss import LogicCompiler, Predicate


def test_deterministic_predicate() -> None:
    """Test deterministic (non-model) predicate."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    def deterministic_func(x: torch.Tensor) -> torch.Tensor:
        """Deterministic predicate function."""
        return (x.sum(dim=-1) > 0).float()

    predicates = {"P": Predicate("P", deterministic_func, is_model=False)}

    logic_loss = LogicCompiler(expr, predicates)

    x_pos = torch.ones(5, 3)
    x_neg = -torch.ones(5, 3)

    assert (logic_loss(x_pos) == 1.0).all()
    assert (logic_loss(x_neg) == 0.0).all()


def test_model_predicate_auto_detection() -> None:
    """Test automatic detection of nn.Module as model."""
    # Should auto-detect nn.Module as model
    model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
    predicate = Predicate("P", model)

    assert predicate.is_model is True


def test_function_predicate_auto_detection() -> None:
    """Test automatic detection of function as non-model."""
    # Should auto-detect lambda as non-model
    predicate = Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1)))

    assert predicate.is_model is False


def test_get_trainable_parameters() -> None:
    """Test getting trainable parameters from models."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    model_p = nn.Linear(5, 1)

    predicates = {
        "P": Predicate("P", model_p),
        "Q": Predicate("Q", lambda x: (x > 0).float().mean(dim=-1)),
    }

    logic_loss = LogicCompiler(expr, predicates)
    params = logic_loss.get_trainable_parameters()

    # Should only get parameters from model_p
    assert len(params) == 2  # weight and bias
    assert all(p.requires_grad for p in params)


def test_get_trainable_parameters_no_models() -> None:
    """Test get_trainable_parameters with no model predicates."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = LogicCompiler(expr, predicates)
    params = logic_loss.get_trainable_parameters()

    # Should return empty list
    assert len(params) == 0


def test_non_tensor_predicate_return() -> None:
    """Test predicate that returns non-tensor (float)."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    # Predicate that returns a Python float (will be converted to tensor)
    predicates = {"P": Predicate("P", lambda x: 0.75)}

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # Should convert 0.75 to tensor
    assert isinstance(satisfaction, torch.Tensor)
    assert torch.allclose(satisfaction, torch.tensor(0.75))


def test_predicate_clamping_above_one() -> None:
    """Test that predicates returning values >1 are clamped to [0,1]."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    # Predicate that returns values > 1
    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 2.5)}

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # Should be clamped to 1.0
    assert torch.allclose(satisfaction, torch.tensor(1.0))
    assert satisfaction.max() <= 1.0


def test_predicate_clamping_below_zero() -> None:
    """Test that predicates returning values <0 are clamped to [0,1]."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    # Predicate that returns values < 0
    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * -1.5)}

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # Should be clamped to 0.0
    assert torch.allclose(satisfaction, torch.tensor(0.0))
    assert satisfaction.min() >= 0.0


def test_predicate_with_neural_network() -> None:
    """Test predicate wrapping a neural network."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    # Create a simple neural network
    model = nn.Sequential(
        nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid()
    )

    predicates = {"P": Predicate("P", lambda x: model(x).squeeze(-1))}

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # Should return values in [0, 1] due to sigmoid
    assert satisfaction.shape == (10,)
    assert satisfaction.min() >= 0.0
    assert satisfaction.max() <= 1.0


def test_predicate_with_multiple_models() -> None:
    """Test multiple model-based predicates."""
    # pylint: disable=invalid-name
    P, Q, R = sp.symbols("P Q R")
    expr = sp.And(sp.Or(P, Q), R)

    model_p = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
    model_q = nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 1),
                            nn.Sigmoid())
    model_r = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())

    predicates = {
        "P": Predicate("P", lambda x: model_p(x).squeeze(-1)),
        "Q": Predicate("Q", lambda x: model_q(x).squeeze(-1)),
        "R": Predicate("R", lambda x: model_r(x).squeeze(-1)),
    }

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # Should compute correctly with all models
    assert satisfaction.shape == (10,)
    assert satisfaction.min() >= 0.0
    assert satisfaction.max() <= 1.0


def test_predicate_name_attribute() -> None:
    """Test that predicate stores its name correctly."""
    # pylint: disable=invalid-name
    predicate_p = Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.5)
    predicate_q = Predicate("CustomName", lambda x: torch.ones(x.shape[0]) *
                            0.7)

    assert predicate_p.name == "P"
    assert predicate_q.name == "CustomName"


def test_predicate_callable_behavior() -> None:
    """Test that predicates are callable and pass through arguments."""
    # pylint: disable=invalid-name
    def custom_func(x: torch.Tensor, y: int = 1) -> torch.Tensor:
        """Custom function with multiple arguments."""
        return torch.ones(x.shape[0]) * y * 0.5

    predicate = Predicate("P", custom_func)

    x = torch.randn(5, 3)

    # Test with default argument
    result1 = predicate(x)
    assert torch.allclose(result1, torch.tensor(0.5))

    # Test with custom argument
    result2 = predicate(x, y=2)
    assert torch.allclose(result2, torch.tensor(1.0))


def test_predicate_explicit_is_model_flag() -> None:
    """Test explicit is_model flag overrides auto-detection."""
    # pylint: disable=invalid-name
    model = nn.Linear(5, 1)

    # Explicitly mark as non-model (override auto-detection)
    predicate = Predicate("P", model, is_model=False)
    assert predicate.is_model is False

    # Explicitly mark lambda as model
    predicate2 = Predicate("Q", lambda x: x.sum(), is_model=True)
    assert predicate2.is_model is True
