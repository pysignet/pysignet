"""Tests for boolean constants (sp.true and sp.false).

This module tests handling of SymPy boolean constants (true/false)
in logical expressions.
"""

import sympy as sp
import torch

from pysignet import Predicate, Symbol, Variable, compile_logic


def test_true_constant() -> None:
    """Test sp.true evaluates to 1.0."""
    expr = sp.true  # Just the constant

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(x)

    # true should always evaluate to 1.0
    assert torch.allclose(satisfaction, torch.tensor(1.0))


def test_false_constant() -> None:
    """Test sp.false evaluates to 0.0."""
    expr = sp.false  # Just the constant

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(x)

    # false should always evaluate to 0.0
    assert torch.allclose(satisfaction, torch.tensor(0.0))


def test_and_with_true() -> None:
    """Test P AND true = P."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = sp.And(P(X), sp.true)

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(x)

    # P AND true = P
    assert torch.allclose(satisfaction, torch.tensor(0.6), atol=1e-5)


def test_and_with_false() -> None:
    """Test P AND false = false."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = sp.And(P(X), sp.false)

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(x)

    # P AND false = false = 0
    assert torch.allclose(satisfaction, torch.tensor(0.0), atol=1e-5)


def test_or_with_true() -> None:
    """Test P OR true = true."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = sp.Or(P(X), sp.true)

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(x)

    # P OR true = true = 1
    assert torch.allclose(satisfaction, torch.tensor(1.0), atol=1e-5)


def test_or_with_false() -> None:
    """Test P OR false = P."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = sp.Or(P(X), sp.false)

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(x)

    # P OR false = P = 0.6
    assert torch.allclose(satisfaction, torch.tensor(0.6), atol=1e-5)


def test_not_true() -> None:
    """Test NOT true = false."""
    expr = sp.Not(sp.true)

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(x)

    # NOT true = false = 0
    assert torch.allclose(satisfaction, torch.tensor(0.0))


def test_not_false() -> None:
    """Test NOT false = true."""
    expr = sp.Not(sp.false)

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(x)

    # NOT false = true = 1
    assert torch.allclose(satisfaction, torch.tensor(1.0))


def test_implies_with_true_antecedent() -> None:
    """Test true -> P = P."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = sp.Implies(sp.true, P(X))

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(x)

    # true -> P = NOT(true) OR P = false OR P = P
    assert torch.allclose(satisfaction, torch.tensor(0.6), atol=1e-5)


def test_implies_with_false_antecedent() -> None:
    """Test false -> P = true."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = sp.Implies(sp.false, P(X))

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(x)

    # false -> P = NOT(false) OR P = true OR P = true
    assert torch.allclose(satisfaction, torch.tensor(1.0), atol=1e-5)


def test_implies_with_true_consequent() -> None:
    """Test P -> true = true."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = sp.Implies(P(X), sp.true)

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(x)

    # P -> true = NOT(P) OR true = true
    assert torch.allclose(satisfaction, torch.tensor(1.0), atol=1e-5)


def test_implies_with_false_consequent() -> None:
    """Test P -> false = NOT P."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = sp.Implies(P(X), sp.false)

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(x)

    # P -> false = NOT(P) OR false = NOT(P)
    assert torch.allclose(satisfaction, torch.tensor(0.4), atol=1e-5)


def test_constants_with_dict_input() -> None:
    """Test boolean constants with dict input."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    # Test with true constant
    expr_true = sp.Or(P(X), sp.true)
    logic_loss_true = compile_logic(expr_true, predicates)
    inputs = {"X": torch.randn(1, 3)}
    satisfaction_true = logic_loss_true(inputs)

    # P OR true = true = 1
    assert torch.allclose(satisfaction_true, torch.tensor(1.0), atol=1e-5)

    # Test with false constant
    expr_false = sp.Or(P(X), sp.false)
    logic_loss_false = compile_logic(expr_false, predicates)
    satisfaction_false = logic_loss_false(inputs)

    # P OR false = P = 0.6
    assert torch.allclose(satisfaction_false, torch.tensor(0.6), atol=1e-5)


def test_complex_with_constants() -> None:
    """Test complex expressions mixing predicates and constants."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P, Q = Symbol("P Q")

    # (P AND true) OR (Q AND false) = P OR false = P
    expr = sp.Or(sp.And(P(X), sp.true), sp.And(Q(X), sp.false))

    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7),
        "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5),
    }

    logic_loss = compile_logic(expr, predicates)
    x = torch.randn(1, 3)
    satisfaction = logic_loss(x)

    # Should simplify to P
    assert torch.allclose(satisfaction, torch.tensor(0.7), atol=1e-5)


def test_constants_preserve_batch_size() -> None:
    """Test that constants preserve batch size correctly."""
    # pylint: disable=invalid-name
    X = Variable("X")
    P = Symbol("P")

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = compile_logic(sp.And(P(X), sp.true), predicates)

    # Test with different batch sizes using quantify='none' to get per-batch results
    for batch_size in [1, 5, 10]:
        x = torch.randn(batch_size, 5)
        satisfaction = logic_loss({"X": x}, quantify='none')
        assert satisfaction.shape == (batch_size,)
        assert torch.allclose(satisfaction, torch.ones(batch_size) * 0.6)


def test_equivalent_with_constants() -> None:
    """Test EQUIVALENT with boolean constants."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8)}

    # P <-> true = P
    expr_true = sp.Equivalent(P(X), sp.true)
    logic_loss_true = compile_logic(expr_true, predicates)
    x = torch.randn(1, 3)
    satisfaction_true = logic_loss_true(x)

    # P <-> true = (P -> true) AND (true -> P) = true AND P = P
    assert torch.allclose(satisfaction_true, torch.tensor(0.8), atol=1e-5)

    # P <-> false = NOT P
    expr_false = sp.Equivalent(P(X), sp.false)
    logic_loss_false = compile_logic(expr_false, predicates)
    satisfaction_false = logic_loss_false(x)

    # P <-> false = (P -> false) AND (false -> P)
    #              = NOT(P) AND true = NOT(P)
    assert torch.allclose(satisfaction_false, torch.tensor(0.2), atol=1e-5)


def test_false_constant_with_dict_input() -> None:
    """Test sp.false with dict input."""
    expr = sp.false

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = compile_logic(expr, predicates)
    inputs = {"P": torch.randn(1, 3)}
    satisfaction = logic_loss(inputs)

    # false should always evaluate to 0.0
    assert torch.allclose(satisfaction, torch.tensor(0.0))
