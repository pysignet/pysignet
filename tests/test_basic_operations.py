"""Tests for basic logical operations (AND, OR, NOT, IMPLIES, EQUIVALENT).

This module tests the core logical operators provided by the library,
ensuring they work correctly with the default RProductTNorm.
"""

import sympy as sp
import torch

from pysignet import LogicCompiler, Predicate


def test_basic_and() -> None:
    """Test basic AND operation."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # Product t-norm: 0.8 * 0.6 = 0.48
    assert satisfaction.shape == (10,)
    assert torch.allclose(satisfaction, torch.tensor(0.48), atol=1e-5)


def test_basic_or() -> None:
    """Test basic OR operation."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.Or(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # Product t-conorm: 0.8 + 0.6 - 0.8*0.6 = 0.92
    expected = 0.8 + 0.6 - 0.8 * 0.6
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


def test_negation() -> None:
    """Test NOT operation."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = sp.Not(P)

    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.7)}

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    assert torch.allclose(satisfaction, torch.tensor(0.3), atol=1e-5)


def test_implication() -> None:
    """Test IMPLIES operation with R-Product (default)."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.Implies(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # R-Product: P -> Q = (1 if P <= Q else Q/P)
    # 0.8 > 0.6, so result = 0.6/0.8 = 0.75
    expected = 0.6 / 0.8
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


def test_equivalence_operator() -> None:
    """Test EQUIVALENCE (biconditional) operator with R-Product (default)."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.Equivalent(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # P <-> Q = (P -> Q) AND (Q -> P)
    # R-Product:
    # P -> Q: 0.8 > 0.6, so 0.6/0.8 = 0.75
    # Q -> P: 0.6 <= 0.8, so 1.0
    # Result = 0.75 * 1.0 = 0.75
    p_implies_q = 0.6 / 0.8
    q_implies_p = 1.0
    expected = p_implies_q * q_implies_p

    assert satisfaction.shape == (5,)
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


def test_complex_expression() -> None:
    """Test complex nested expression."""
    # pylint: disable=invalid-name
    P, Q, R = sp.symbols("P Q R")
    expr = sp.And(sp.Or(P, Q), sp.Not(R))

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.5),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.5),
        "R": Predicate("R", lambda x: torch.ones(x.shape[0]) * 0.3),
    }

    logic_loss = LogicCompiler(expr, predicates)
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # (P | Q) & ~R
    # P | Q = 0.5 + 0.5 - 0.25 = 0.75
    # ~R = 0.7
    # Result = 0.75 * 0.7 = 0.525
    p_or_q = 0.5 + 0.5 - 0.5 * 0.5
    not_r = 0.7
    expected = p_or_q * not_r
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)
