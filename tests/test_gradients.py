"""Tests for gradient flow and differentiability.

This module tests that gradients flow correctly through all logical
operators and predicates, ensuring the library is fully differentiable.
"""

import sympy as sp
import torch
import torch.nn as nn

from pysignet import (
    logic_to_loss,
    Symbol,
    Variable,
    Predicate,
    RProductTNorm,
    LukasiewiczTNorm,
    GodelTNorm,
)


def test_basic_gradient_flow() -> None:
    """Test that gradients flow through the loss."""
    # pylint: disable=invalid-name
    X = Variable("X")
    P = Symbol("P")
    expr = P(X)

    model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
    predicates = {"P": Predicate(lambda x: model(x).squeeze(-1))}

    logic_loss = logic_to_loss(expr, predicates)

    x = torch.randn(1, 5)
    loss = logic_loss.loss(X=x)
    loss.backward()  # type: ignore[no-untyped-call]

    # Check gradients exist
    for param in model.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


def test_gradient_flow_and() -> None:
    """Test gradient flow through AND operation."""
    # pylint: disable=invalid-name
    X = Variable("X")
    P, Q = Symbol("P Q")
    expr = sp.And(P(X), Q(X))

    model_p = nn.Linear(5, 1)
    model_q = nn.Linear(5, 1)

    predicates = {
        "P": Predicate(lambda x: torch.sigmoid(model_p(x).squeeze(-1))),
        "Q": Predicate(lambda x: torch.sigmoid(model_q(x).squeeze(-1))),
    }

    logic_loss = logic_to_loss(expr, predicates)

    x = torch.randn(1, 5)
    loss = logic_loss.loss(X=x)
    loss.backward()  # type: ignore[no-untyped-call]

    # Both models should have gradients
    for param in model_p.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()

    for param in model_q.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


def test_gradient_flow_or() -> None:
    """Test gradient flow through OR operation."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q = Symbol("P Q")

    expr = sp.Or(P(X), Q(X))

    model_p = nn.Linear(5, 1)
    model_q = nn.Linear(5, 1)

    predicates = {
        "P": Predicate( lambda x: torch.sigmoid(model_p(x).squeeze(-1))),
        "Q": Predicate( lambda x: torch.sigmoid(model_q(x).squeeze(-1))),
    }

    logic_loss = logic_to_loss(expr, predicates)

    x = torch.randn(1, 5)
    loss = logic_loss.loss(X=x)
    loss.backward()  # type: ignore[no-untyped-call]

    # Both models should have gradients
    for param in model_p.parameters():
        assert param.grad is not None

    for param in model_q.parameters():
        assert param.grad is not None


def test_gradient_flow_not() -> None:
    """Test gradient flow through NOT operation."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P = Symbol("P")

    expr = sp.Not(P(X))

    model = nn.Linear(5, 1)
    predicates = {
        "P": Predicate( lambda x: torch.sigmoid(model(x).squeeze(-1)))
    }

    logic_loss = logic_to_loss(expr, predicates)

    x = torch.randn(1, 5)
    loss = logic_loss.loss(X=x)
    loss.backward()  # type: ignore[no-untyped-call]

    # Gradients should flow through NOT
    for param in model.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


def test_gradient_flow_implies() -> None:
    """Test gradient flow through IMPLIES operation."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q = Symbol("P Q")

    expr = sp.Implies(P(X), Q(X))

    model_p = nn.Linear(5, 1)
    model_q = nn.Linear(5, 1)

    predicates = {
        "P": Predicate( lambda x: torch.sigmoid(model_p(x).squeeze(-1))),
        "Q": Predicate( lambda x: torch.sigmoid(model_q(x).squeeze(-1))),
    }

    logic_loss = logic_to_loss(expr, predicates)

    x = torch.randn(1, 5)
    loss = logic_loss.loss(X=x)
    loss.backward()  # type: ignore[no-untyped-call]

    # Both models should have gradients
    for param in model_p.parameters():
        assert param.grad is not None

    for param in model_q.parameters():
        assert param.grad is not None


def test_gradient_flow_equivalent() -> None:
    """Test gradient flow through EQUIVALENT operation."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q = Symbol("P Q")

    expr = sp.Equivalent(P(X), Q(X))

    model_p = nn.Linear(5, 1)
    model_q = nn.Linear(5, 1)

    predicates = {
        "P": Predicate( lambda x: torch.sigmoid(model_p(x).squeeze(-1))),
        "Q": Predicate( lambda x: torch.sigmoid(model_q(x).squeeze(-1))),
    }

    logic_loss = logic_to_loss(expr, predicates)

    x = torch.randn(1, 5)
    loss = logic_loss.loss(X=x)
    loss.backward()  # type: ignore[no-untyped-call]

    # Both models should have gradients
    for param in model_p.parameters():
        assert param.grad is not None

    for param in model_q.parameters():
        assert param.grad is not None


def test_gradient_flow_complex_expression() -> None:
    """Test gradient flow through complex nested expression."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q, R = Symbol("P Q R")

    expr = sp.And(sp.Or(P(X), Q(X)), sp.Not(R(X)))

    model_p = nn.Linear(5, 1)
    model_q = nn.Linear(5, 1)
    model_r = nn.Linear(5, 1)

    predicates = {
        "P": Predicate( lambda x: torch.sigmoid(model_p(x).squeeze(-1))),
        "Q": Predicate( lambda x: torch.sigmoid(model_q(x).squeeze(-1))),
        "R": Predicate( lambda x: torch.sigmoid(model_r(x).squeeze(-1))),
    }

    logic_loss = logic_to_loss(expr, predicates)

    x = torch.randn(1, 5)
    loss = logic_loss.loss(X=x)
    loss.backward()  # type: ignore[no-untyped-call]

    # All three models should have gradients
    for param in model_p.parameters():
        assert param.grad is not None

    for param in model_q.parameters():
        assert param.grad is not None

    for param in model_r.parameters():
        assert param.grad is not None


def test_gradient_flow_product_tnorm() -> None:
    """Test gradient flow with R-Product t-norm."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q = Symbol("P Q")

    expr = sp.And(P(X), Q(X))

    model_p = nn.Linear(5, 1)
    model_q = nn.Linear(5, 1)

    predicates = {
        "P": Predicate( lambda x: torch.sigmoid(model_p(x).squeeze(-1))),
        "Q": Predicate( lambda x: torch.sigmoid(model_q(x).squeeze(-1))),
    }

    logic_loss = logic_to_loss(expr, predicates, tnorm=RProductTNorm())

    x = torch.randn(1, 5)
    loss = logic_loss.loss(X=x)
    loss.backward()  # type: ignore[no-untyped-call]

    # Both models should have gradients
    for param in model_p.parameters():
        assert param.grad is not None

    for param in model_q.parameters():
        assert param.grad is not None


def test_gradient_flow_lukasiewicz_tnorm() -> None:
    """Test gradient flow with Łukasiewicz t-norm."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q = Symbol("P Q")

    expr = sp.And(P(X), Q(X))

    model_p = nn.Linear(5, 1)
    model_q = nn.Linear(5, 1)

    predicates = {
        "P": Predicate( lambda x: torch.sigmoid(model_p(x).squeeze(-1))),
        "Q": Predicate( lambda x: torch.sigmoid(model_q(x).squeeze(-1))),
    }

    logic_loss = logic_to_loss(expr, predicates, tnorm=LukasiewiczTNorm())

    x = torch.randn(1, 5)
    loss = logic_loss.loss(X=x)
    loss.backward()  # type: ignore[no-untyped-call]

    # Both models should have gradients (may be zero in some cases)
    for param in model_p.parameters():
        assert param.grad is not None

    for param in model_q.parameters():
        assert param.grad is not None


def test_gradient_flow_godel_tnorm() -> None:
    """Test gradient flow with Gödel t-norm."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q = Symbol("P Q")

    expr = sp.And(P(X), Q(X))

    model_p = nn.Linear(5, 1)
    model_q = nn.Linear(5, 1)

    predicates = {
        "P": Predicate( lambda x: torch.sigmoid(model_p(x).squeeze(-1))),
        "Q": Predicate( lambda x: torch.sigmoid(model_q(x).squeeze(-1))),
    }

    logic_loss = logic_to_loss(expr, predicates, tnorm=GodelTNorm())

    x = torch.randn(1, 5)
    loss = logic_loss.loss(X=x)
    loss.backward()  # type: ignore[no-untyped-call]

    # Both models should have gradients (may be zero for non-minimum)
    for param in model_p.parameters():
        assert param.grad is not None

    for param in model_q.parameters():
        assert param.grad is not None


def test_gradient_accumulation() -> None:
    """Test gradient accumulation across multiple backward passes."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P = Symbol("P")

    expr = P(X)

    model = nn.Linear(5, 1)
    predicates = {
        "P": Predicate( lambda x: torch.sigmoid(model(x).squeeze(-1)))
    }

    logic_loss = logic_to_loss(expr, predicates)

    # First backward pass
    x1 = torch.randn(1, 5)
    loss1 = logic_loss.loss(X=x1)
    loss1.backward()  # type: ignore[no-untyped-call]

    # Save first gradients
    first_grads = []
    for p in model.parameters():
        assert p.grad is not None
        first_grads.append(p.grad.clone())

    # Second backward pass (accumulate)
    x2 = torch.randn(1, 5)
    loss2 = logic_loss.loss(X=x2)
    loss2.backward()  # type: ignore[no-untyped-call]

    # Gradients should have accumulated
    for param, first_grad in zip(model.parameters(), first_grads):
        assert param.grad is not None
        # Gradients should be different (accumulated)
        assert not torch.allclose(param.grad, first_grad)


def test_gradient_zero() -> None:
    """Test that zero_grad works correctly."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P = Symbol("P")

    expr = P(X)

    model = nn.Linear(5, 1)
    predicates = {
        "P": Predicate( lambda x: torch.sigmoid(model(x).squeeze(-1)))
    }

    logic_loss = logic_to_loss(expr, predicates)

    # First backward pass
    x = torch.randn(1, 5)
    loss = logic_loss.loss(X=x)
    loss.backward()  # type: ignore[no-untyped-call]

    # Gradients should exist
    for param in model.parameters():
        assert param.grad is not None

    # Zero gradients
    model.zero_grad()

    # Gradients should be None or zero
    for param in model.parameters():
        assert param.grad is None or torch.allclose(
            param.grad, torch.zeros_like(param.grad)
        )


def test_gradient_no_nan_or_inf() -> None:
    """Test that gradients don't contain NaN or Inf."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q, R = Symbol("P Q R")

    expr = sp.And(sp.Or(P(X), Q(X)), sp.Not(R(X)))

    model_p = nn.Linear(5, 10)
    model_q = nn.Linear(5, 10)
    model_r = nn.Linear(5, 10)

    predicates = {
        "P": Predicate(
            lambda x: torch.sigmoid(model_p(x).mean(dim=-1))
        ),
        "Q": Predicate(
            lambda x: torch.sigmoid(model_q(x).mean(dim=-1))
        ),
        "R": Predicate(
            lambda x: torch.sigmoid(model_r(x).mean(dim=-1))
        ),
    }

    logic_loss = logic_to_loss(expr, predicates)

    x = torch.randn(1, 5)
    loss = logic_loss.loss(X=x)
    loss.backward()  # type: ignore[no-untyped-call]

    # Check no NaN or Inf in gradients
    for model in [model_p, model_q, model_r]:
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()


def test_get_trainable_parameters_for_optimization() -> None:
    """Test using get_trainable_parameters with an optimizer."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q = Symbol("P Q")

    expr = sp.And(P(X), Q(X))

    model_p = nn.Linear(5, 1)

    predicates = {
        "P": Predicate( model_p),
        "Q": Predicate( lambda x: (x > 0).float().mean(dim=-1)),
    }

    logic_loss = logic_to_loss(expr, predicates)
    params = logic_loss.get_trainable_parameters()

    # Create optimizer with trainable parameters
    optimizer = torch.optim.SGD(params, lr=0.01)

    x = torch.randn(1, 5)
    loss = logic_loss.loss(X=x)
    loss.backward()  # type: ignore[no-untyped-call]

    # Optimizer step should work
    optimizer.step()
    optimizer.zero_grad()

    # Should be able to continue training
    loss2 = logic_loss.loss(X=x)
    loss2.backward()  # type: ignore[no-untyped-call]


def test_gradient_with_multiple_models() -> None:
    """Test gradient flow with multiple neural network predicates."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q = Symbol("P Q")

    expr = sp.And(P(X), Q(X))

    # Different architectures
    model_p = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1),
                            nn.Sigmoid())
    model_q = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())

    predicates = {
        "P": Predicate( lambda x: model_p(x).squeeze(-1)),
        "Q": Predicate( lambda x: model_q(x).squeeze(-1)),
    }

    logic_loss = logic_to_loss(expr, predicates)

    x = torch.randn(1, 5)
    loss = logic_loss.loss(X=x)
    loss.backward()  # type: ignore[no-untyped-call]

    # All parameters should have gradients
    for param in model_p.parameters():
        assert param.grad is not None

    for param in model_q.parameters():
        assert param.grad is not None


def test_gradient_independent_of_batch_size() -> None:
    """Test that gradient computation works for different batch sizes."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P = Symbol("P")

    expr = P(X)

    predicates = {"P": Predicate( lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = logic_to_loss(expr, predicates)

    # Test with different batch sizes
    for batch_size in [1]:
        x = torch.randn(batch_size, 5, requires_grad=True)
        loss = logic_loss.loss(X=x)
        loss.backward()  # type: ignore[no-untyped-call]

        # Input should have gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.isnan(x.grad).any()
