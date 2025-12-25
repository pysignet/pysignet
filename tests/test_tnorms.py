"""Tests for different t-norm implementations.

This module tests all t-norm implementations (S-Product, Łukasiewicz,
Gödel) across all logical operators to ensure correct behavior.

Note: R-Product has its own dedicated test file (test_r_product.py).
S-Product tests here verify the implication-as-disjunction semantics.
"""

import sympy as sp
import torch

from logic_as_loss import (
    LogicLoss,
    Predicate,
    SProductTNorm,
    LukasiewiczTNorm,
    GodelTNorm,
)


# Product T-Norm Tests


def test_product_and() -> None:
    """Test Product t-norm AND operation."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = LogicLoss(expr, predicates, tnorm=SProductTNorm())
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # Product t-norm AND: 0.8 * 0.6 = 0.48
    assert torch.allclose(satisfaction, torch.tensor(0.48), atol=1e-5)


def test_product_or() -> None:
    """Test Product t-norm OR operation."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.Or(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = LogicLoss(expr, predicates, tnorm=SProductTNorm())
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # Product t-conorm OR: 0.8 + 0.6 - 0.8*0.6 = 0.92
    expected = 0.8 + 0.6 - 0.8 * 0.6
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


def test_product_not() -> None:
    """Test Product t-norm NOT operation."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = sp.Not(P)

    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.7)}

    logic_loss = LogicLoss(expr, predicates, tnorm=SProductTNorm())
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # NOT: 1 - 0.7 = 0.3
    assert torch.allclose(satisfaction, torch.tensor(0.3), atol=1e-5)


def test_product_implies() -> None:
    """Test Product t-norm IMPLIES operation."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.Implies(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = LogicLoss(expr, predicates, tnorm=SProductTNorm())
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # P -> Q = ~P | Q = 0.2 | 0.6
    not_p = 0.2
    expected = not_p + 0.6 - not_p * 0.6
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


def test_product_equivalent() -> None:
    """Test Product t-norm EQUIVALENT operation."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.Equivalent(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = LogicLoss(expr, predicates, tnorm=SProductTNorm())
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # P <-> Q = (P -> Q) AND (Q -> P)
    # P -> Q = NOT P OR Q = 0.2 + 0.6 - 0.12 = 0.68
    # Q -> P = NOT Q OR P = 0.4 + 0.8 - 0.32 = 0.88
    # Result = 0.68 * 0.88 = 0.5984
    p_implies_q = 0.2 + 0.6 - 0.2 * 0.6
    q_implies_p = 0.4 + 0.8 - 0.4 * 0.8
    expected = p_implies_q * q_implies_p

    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


# Łukasiewicz T-Norm Tests


def test_lukasiewicz_and() -> None:
    """Test Łukasiewicz t-norm AND operation."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = LogicLoss(expr, predicates, tnorm=LukasiewiczTNorm())
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # Łukasiewicz AND: max(0, 0.8 + 0.6 - 1) = 0.4
    assert torch.allclose(satisfaction, torch.tensor(0.4), atol=1e-5)


def test_lukasiewicz_or() -> None:
    """Test Łukasiewicz t-norm OR operation."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.Or(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.7),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.5),
    }

    logic_loss = LogicLoss(expr, predicates, tnorm=LukasiewiczTNorm())
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # Łukasiewicz OR: min(1, 0.7 + 0.5) = min(1, 1.2) = 1.0
    assert torch.allclose(satisfaction, torch.tensor(1.0), atol=1e-5)


def test_lukasiewicz_not() -> None:
    """Test Łukasiewicz t-norm NOT operation."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = sp.Not(P)

    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.7)}

    logic_loss = LogicLoss(expr, predicates, tnorm=LukasiewiczTNorm())
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # NOT: 1 - 0.7 = 0.3 (same across all t-norms)
    assert torch.allclose(satisfaction, torch.tensor(0.3), atol=1e-5)


def test_lukasiewicz_implies() -> None:
    """Test Łukasiewicz t-norm IMPLIES operation."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.Implies(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.5),
    }

    logic_loss = LogicLoss(expr, predicates, tnorm=LukasiewiczTNorm())
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # P -> Q = ~P | Q = NOT(0.8) | 0.5 = 0.2 | 0.5
    # Łukasiewicz OR: min(1, 0.2 + 0.5) = 0.7
    expected = min(1.0, 0.2 + 0.5)
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


def test_lukasiewicz_equivalent() -> None:
    """Test Łukasiewicz t-norm EQUIVALENT operation."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.Equivalent(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = LogicLoss(expr, predicates, tnorm=LukasiewiczTNorm())
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # P <-> Q = (P -> Q) AND (Q -> P)
    # P -> Q = min(1, 0.2 + 0.6) = 0.8
    # Q -> P = min(1, 0.4 + 0.8) = 1.0
    # Result = max(0, 0.8 + 1.0 - 1) = 0.8
    p_implies_q = min(1.0, 0.2 + 0.6)
    q_implies_p = min(1.0, 0.4 + 0.8)
    expected = max(0.0, p_implies_q + q_implies_p - 1.0)

    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


# Gödel T-Norm Tests


def test_godel_and() -> None:
    """Test Gödel t-norm AND operation."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.7),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.5),
    }

    logic_loss = LogicLoss(expr, predicates, tnorm=GodelTNorm())
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # Gödel AND: min(0.7, 0.5) = 0.5
    assert torch.allclose(satisfaction, torch.tensor(0.5), atol=1e-5)


def test_godel_or() -> None:
    """Test Gödel t-norm OR operation."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.Or(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.7),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.5),
    }

    logic_loss = LogicLoss(expr, predicates, tnorm=GodelTNorm())
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # Gödel OR: max(0.7, 0.5) = 0.7
    assert torch.allclose(satisfaction, torch.tensor(0.7), atol=1e-5)


def test_godel_not() -> None:
    """Test Gödel t-norm NOT operation."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = sp.Not(P)

    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.7)}

    logic_loss = LogicLoss(expr, predicates, tnorm=GodelTNorm())
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # NOT: 1 - 0.7 = 0.3 (same across all t-norms)
    assert torch.allclose(satisfaction, torch.tensor(0.3), atol=1e-5)


def test_godel_implies() -> None:
    """Test Gödel t-norm IMPLIES operation."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.Implies(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.5),
    }

    logic_loss = LogicLoss(expr, predicates, tnorm=GodelTNorm())
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # P -> Q = ~P | Q = 0.2 | 0.5
    # Gödel OR: max(0.2, 0.5) = 0.5
    expected = max(0.2, 0.5)
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


def test_godel_equivalent() -> None:
    """Test Gödel t-norm EQUIVALENT operation."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.Equivalent(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = LogicLoss(expr, predicates, tnorm=GodelTNorm())
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # P <-> Q = (P -> Q) AND (Q -> P)
    # P -> Q = max(0.2, 0.6) = 0.6
    # Q -> P = max(0.4, 0.8) = 0.8
    # Result = min(0.6, 0.8) = 0.6
    p_implies_q = max(0.2, 0.6)
    q_implies_p = max(0.4, 0.8)
    expected = min(p_implies_q, q_implies_p)

    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


# Multi-argument tests


def test_product_multi_and() -> None:
    """Test Product t-norm with 3+ arguments in AND."""
    # pylint: disable=invalid-name
    P, Q, R = sp.symbols("P Q R")
    expr = sp.And(P, Q, R)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
        "R": Predicate("R", lambda x: torch.ones(x.shape[0]) * 0.5),
    }

    logic_loss = LogicLoss(expr, predicates, tnorm=SProductTNorm())
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # Product AND: 0.8 * 0.6 * 0.5 = 0.24
    assert torch.allclose(satisfaction, torch.tensor(0.24), atol=1e-5)


def test_lukasiewicz_multi_or() -> None:
    """Test Łukasiewicz t-norm with 3+ arguments in OR."""
    # pylint: disable=invalid-name
    P, Q, R = sp.symbols("P Q R")
    expr = sp.Or(P, Q, R)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.3),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.4),
        "R": Predicate("R", lambda x: torch.ones(x.shape[0]) * 0.5),
    }

    logic_loss = LogicLoss(expr, predicates, tnorm=LukasiewiczTNorm())
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # Łukasiewicz OR is associative:
    # 0.3 | 0.4 = min(1, 0.7) = 0.7
    # 0.7 | 0.5 = min(1, 1.2) = 1.0
    assert torch.allclose(satisfaction, torch.tensor(1.0), atol=1e-5)


def test_godel_multi_and() -> None:
    """Test Gödel t-norm with 3+ arguments in AND."""
    # pylint: disable=invalid-name
    P, Q, R = sp.symbols("P Q R")
    expr = sp.And(P, Q, R)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
        "R": Predicate("R", lambda x: torch.ones(x.shape[0]) * 0.5),
    }

    logic_loss = LogicLoss(expr, predicates, tnorm=GodelTNorm())
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # Gödel AND: min(0.8, 0.6, 0.5) = 0.5
    assert torch.allclose(satisfaction, torch.tensor(0.5), atol=1e-5)
