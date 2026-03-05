"""Tests for Predicate class behavior.

This module tests the Predicate wrapper class, including deterministic
vs. model predicates, return value handling, and value clamping.
"""

import pytest
import sympy as sp
import torch
import torch.nn as nn

from pysignet import Symbol, Variable, compile_logic, logic_to_loss, Predicate


def test_deterministic_predicate() -> None:
    """Test deterministic (non-model) predicate."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P = Symbol("P")

    expr = P(X)

    def deterministic_func(x: torch.Tensor) -> torch.Tensor:
        """Deterministic predicate function."""
        return (x.sum(dim=-1) > 0).float()

    predicates = {"P": Predicate( deterministic_func, is_model=False)}

    compiled = compile_logic(expr, predicates)

    x_pos = torch.ones(1, 3)
    x_neg = -torch.ones(1, 3)

    assert (compiled(X=x_pos) == 1.0).all()
    assert (compiled(X=x_neg) == 0.0).all()


def test_model_predicate_auto_detection() -> None:
    """Test automatic detection of nn.Module as model."""
    # Should auto-detect nn.Module as model
    model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
    predicate = Predicate( model)

    assert predicate.is_model is True


def test_function_predicate_auto_detection() -> None:
    """Test automatic detection of function as non-model."""
    # Should auto-detect lambda as non-model
    predicate = Predicate( lambda x: torch.sigmoid(x.sum(dim=-1)))

    assert predicate.is_model is False


def test_get_trainable_parameters() -> None:
    """Test getting trainable parameters from models."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q = Symbol("P Q")

    expr = sp.And(P(X), Q(X))

    model_p = nn.Linear(5, 1)

    predicates = {
        "P": Predicate( model_p),
        "Q": Predicate( lambda x: (x > 0).float().mean(dim=-1)),
    }

    compiled = compile_logic(expr, predicates)
    params = compiled.get_trainable_parameters()

    # Should only get parameters from model_p
    assert len(params) == 2  # weight and bias
    assert all(p.requires_grad for p in params)


def test_get_trainable_parameters_no_models() -> None:
    """Test get_trainable_parameters with no model predicates."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q = Symbol("P Q")

    expr = sp.And(P(X), Q(X))

    predicates = {
        "P": Predicate( lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate( lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    compiled = compile_logic(expr, predicates)
    params = compiled.get_trainable_parameters()

    # Should return empty list
    assert len(params) == 0


def test_non_tensor_predicate_return() -> None:
    """Test predicate that returns non-tensor (float)."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P = Symbol("P")

    expr = P(X)

    # Predicate that returns a Python float (will be converted to tensor)
    predicates = {"P": Predicate( lambda x: 0.75)}

    compiled = compile_logic(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = compiled(X=x)

    # Should convert 0.75 to tensor
    assert isinstance(satisfaction, torch.Tensor)
    assert torch.allclose(satisfaction, torch.tensor(0.75))


def test_predicate_above_one_raises() -> None:
    """Test that non-module predicates returning >1 raise ValueError."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P = Symbol("P")

    expr = P(X)

    # Predicate that returns values > 1
    predicates = {"P": Predicate( lambda x: torch.ones(x.shape[0]) * 2.5)}

    compiled = compile_logic(expr, predicates)
    x = torch.randn(1, 3)

    # Should raise ValueError instead of silently clamping
    with pytest.raises(ValueError, match="outside.*0.*1"):
        compiled(X=x)


def test_predicate_below_zero_raises() -> None:
    """Test that non-module predicates returning <0 raise ValueError."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P = Symbol("P")

    expr = P(X)

    # Predicate that returns values < 0
    predicates = {"P": Predicate( lambda x: torch.ones(x.shape[0]) * -1.5)}

    compiled = compile_logic(expr, predicates)
    x = torch.randn(1, 3)

    # Should raise ValueError instead of silently clamping
    with pytest.raises(ValueError, match="outside.*0.*1"):
        compiled(X=x)


def test_predicate_with_neural_network() -> None:
    """Test predicate wrapping a neural network."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P = Symbol("P")

    expr = P(X)

    # Create a simple neural network
    model = nn.Sequential(
        nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid()
    )

    predicates = {"P": Predicate( lambda x: model(x).squeeze(-1))}

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.randn(1, 5)
    # Default quantify='forall' with batch_size=1 returns scalar
    satisfaction = logic_loss.satisfaction(X=x)

    # Should return values in [0, 1] due to sigmoid
    assert satisfaction.shape == ()  # Scalar with forall quantification
    assert satisfaction.item() >= 0.0
    assert satisfaction.item() <= 1.0


def test_predicate_with_multiple_models() -> None:
    """Test multiple model-based predicates."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q, R = Symbol("P Q R")
    expr = sp.And(sp.Or(P(X), Q(X)), R(X))

    model_p = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
    model_q = nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 1),
                            nn.Sigmoid())
    model_r = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())

    predicates = {
        "P": Predicate( lambda x: model_p(x).squeeze(-1)),
        "Q": Predicate( lambda x: model_q(x).squeeze(-1)),
        "R": Predicate( lambda x: model_r(x).squeeze(-1)),
    }

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.randn(1, 5)
    # Default quantify='forall' with batch_size=1 returns scalar
    satisfaction = logic_loss.satisfaction(X=x)

    # Should compute correctly with all models
    assert satisfaction.shape == ()  # Scalar with forall quantification
    assert satisfaction.item() >= 0.0
    assert satisfaction.item() <= 1.0


def test_predicate_name_attribute() -> None:
    """Test that predicate name is assigned by compiler."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q = Symbol("P Q")

    expr = sp.And(P(X), Q(X))

    predicate_p = Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)
    predicate_q = Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)

    # Names should be None before compilation
    assert predicate_p.name is None
    assert predicate_q.name is None

    # Compile to assign names
    predicates = {"P": predicate_p, "Q": predicate_q}
    compile_logic(expr, predicates)

    # Names should now be assigned
    assert predicate_p.name == "P"
    assert predicate_q.name == "Q"


def test_predicate_callable_behavior() -> None:
    """Test that predicates are callable and pass through arguments."""
    # pylint: disable=invalid-name
    def custom_func(x: torch.Tensor, y: int = 1) -> torch.Tensor:
        """Custom function with multiple arguments."""
        return torch.ones(x.shape[0]) * y * 0.5

    predicate = Predicate( custom_func)

    x = torch.randn(1, 3)

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
    predicate = Predicate( model, is_model=False)
    assert predicate.is_model is False

    # Explicitly mark lambda as model
    predicate2 = Predicate( lambda x: x.sum(), is_model=True)
    assert predicate2.is_model is True


def test_predicate_name_assignment_from_dict_key() -> None:
    """Test that predicate names are assigned from dict keys."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P = Symbol("P")

    expr = P(X)

    predicate = Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)

    # Dict key determines the name
    predicates = {"P": predicate}

    compile_logic(expr, predicates)

    # Predicate should now have name 'P' from dict key
    assert predicate.name == "P"
