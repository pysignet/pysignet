"""Tests for basic logical operations (AND, OR, NOT, IMPLIES, EQUIVALENT).

This module tests the core logical operators provided by the library,
ensuring they work correctly with the default RProductTNorm.

Note: These tests use batch_size=1, so forall quantification (default)
returns the same value as per-batch - just as a scalar.
"""

import sympy as sp
import torch

from pysignet import (
    Predicate,
    Symbol,
    Variable,
    compile_logic,
    logic_to_loss
)


def test_basic_and() -> None:
    """Test basic AND operation."""
    # pylint: disable=invalid-name
    X = Variable("X")
    P, Q = Symbol("P Q")
    expr = sp.And(P(X), Q(X))

    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.randn(1, 5)

    # Default quantify='forall' with batch_size=1 returns scalar
    satisfaction = logic_loss(X=x)

    # Product t-norm: 0.8 * 0.6 = 0.48
    assert satisfaction.shape == ()  # Scalar
    assert torch.allclose(satisfaction, torch.tensor(0.48), atol=1e-5)


def test_basic_and_per_batch() -> None:
    """Test basic AND operation with per-batch results."""
    # pylint: disable=invalid-name
    X = Variable("X")
    P, Q = Symbol("P Q")
    expr = sp.And(P(X), Q(X))

    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.randn(10, 5)

    # Use quantify='none' for per-batch results
    satisfaction = logic_loss(X=x, quantify='none')

    # Product t-norm: 0.8 * 0.6 = 0.48 for each sample
    assert satisfaction.shape == (10,)
    assert torch.allclose(satisfaction, torch.ones(10) * 0.48, atol=1e-5)


def test_basic_or() -> None:
    """Test basic OR operation."""
    # pylint: disable=invalid-name
    X = Variable("X")
    P, Q = Symbol("P Q")
    expr = sp.Or(P(X), Q(X))

    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.randn(1, 5)
    satisfaction = logic_loss(X=x)

    # Product t-conorm: 0.8 + 0.6 - 0.8*0.6 = 0.92
    expected = 0.8 + 0.6 - 0.8 * 0.6
    assert satisfaction.shape == ()  # Scalar
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


def test_negation() -> None:
    """Test NOT operation."""
    # pylint: disable=invalid-name
    X = Variable("X")
    P = Symbol("P")
    expr = sp.Not(P(X))

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.randn(1, 5)
    satisfaction = logic_loss(X=x)

    assert satisfaction.shape == ()  # Scalar
    assert torch.allclose(satisfaction, torch.tensor(0.3), atol=1e-5)


def test_implication() -> None:
    """Test IMPLIES operation with R-Product (default)."""
    # pylint: disable=invalid-name
    X = Variable("X")
    P, Q = Symbol("P Q")
    expr = sp.Implies(P(X), Q(X))

    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.randn(1, 5)
    satisfaction = logic_loss(X=x)

    # R-Product: P -> Q = (1 if P <= Q else Q/P)
    # 0.8 > 0.6, so result = 0.6/0.8 = 0.75
    expected = 0.6 / 0.8
    assert satisfaction.shape == ()  # Scalar
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


def test_equivalence_operator() -> None:
    """Test EQUIVALENCE (biconditional) operator with R-Product (default)."""
    # pylint: disable=invalid-name
    X = Variable("X")
    P, Q = Symbol("P Q")
    expr = sp.Equivalent(P(X), Q(X))

    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(X=x)

    # P <-> Q = (P -> Q) AND (Q -> P)
    # R-Product:
    # P -> Q: 0.8 > 0.6, so 0.6/0.8 = 0.75
    # Q -> P: 0.6 <= 0.8, so 1.0
    # Result = 0.75 * 1.0 = 0.75
    p_implies_q = 0.6 / 0.8
    q_implies_p = 1.0
    expected = p_implies_q * q_implies_p

    assert satisfaction.shape == ()  # Scalar
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


def test_complex_expression() -> None:
    """Test complex nested expression."""
    # pylint: disable=invalid-name
    X = Variable("X")
    P, Q, R = Symbol("P Q R")
    expr = sp.And(sp.Or(P(X), Q(X)), sp.Not(R(X)))

    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5),
        "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5),
        "R": Predicate(lambda x: torch.ones(x.shape[0]) * 0.3),
    }

    logic_loss = logic_to_loss(expr, predicates)
    x = torch.randn(1, 5)
    satisfaction = logic_loss(X=x)

    # (P | Q) & ~R
    # P | Q = 0.5 + 0.5 - 0.25 = 0.75
    # ~R = 0.7
    # Result = 0.75 * 0.7 = 0.525
    p_or_q = 0.5 + 0.5 - 0.5 * 0.5
    not_r = 0.7
    expected = p_or_q * not_r
    assert satisfaction.shape == ()  # Scalar
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)
