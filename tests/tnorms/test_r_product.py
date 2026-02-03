"""Tests for R-Product t-norm implementation.

This module tests the R-Product (residuated Product) t-norm which uses
axiomatic implication via residua: implication(x, y) = 1 if x <= y else y/x.

R-Product differs from S-Product only in the implication operator:
- R-Product: axiomatic residuum-based implication
- S-Product: treats implication as disjunction (NOT x OR y)

According to "Evaluating Relaxations of Logic for Neural Networks", R-Product
empirically outperforms all other t-norms.

"""

import sympy as sp
import torch
import torch.nn as nn

from pysignet import Predicate, RProductTNorm, Symbol, Variable, logic_to_loss


def test_r_product_implication_when_antecedent_less() -> None:
    """Test R-Product implication when x < y returns 1."""
    tnorm = RProductTNorm()

    # When x < y, implication should be 1
    x = torch.tensor([0.3, 0.5, 0.7])
    y = torch.tensor([0.8, 0.9, 1.0])

    result = tnorm.implication(x, y)

    assert torch.allclose(result, torch.ones_like(x))


def test_r_product_implication_when_antecedent_equal() -> None:
    """Test R-Product implication when x == y returns 1."""
    tnorm = RProductTNorm()

    # When x == y, implication should be 1
    x = torch.tensor([0.3, 0.5, 0.7, 1.0])
    y = x.clone()

    result = tnorm.implication(x, y)

    assert torch.allclose(result, torch.ones_like(x))


def test_r_product_implication_when_antecedent_greater() -> None:
    """Test R-Product implication when x > y returns y/x."""
    tnorm = RProductTNorm()

    # When x > y, implication should be y/x
    x = torch.tensor([0.8, 0.9, 1.0])
    y = torch.tensor([0.3, 0.5, 0.7])

    result = tnorm.implication(x, y)
    expected = y / x

    assert torch.allclose(result, expected, atol=1e-6)


def test_r_product_implication_boundary_cases() -> None:
    """Test R-Product implication with boundary values."""
    tnorm = RProductTNorm()

    # x=0, y=any: should return 1 (0 <= any)
    x_zero = torch.tensor([0.0, 0.0, 0.0])
    y_any = torch.tensor([0.0, 0.5, 1.0])
    result_zero = tnorm.implication(x_zero, y_any)
    assert torch.allclose(result_zero, torch.ones_like(x_zero))

    # x=1, y=1: should return 1 (1 <= 1)
    x_one = torch.ones(3)
    y_one = torch.ones(3)
    result_one_one = tnorm.implication(x_one, y_one)
    assert torch.allclose(result_one_one, torch.ones_like(x_one))

    # x=1, y=0: should return 0 (0/1 = 0)
    x_one = torch.ones(3)
    y_zero = torch.zeros(3)
    result_one_zero = tnorm.implication(x_one, y_zero)
    assert torch.allclose(result_one_zero, torch.zeros_like(x_one))


def test_r_product_implication_numerical_stability() -> None:
    """Test R-Product implication doesn't produce NaN or Inf."""
    tnorm = RProductTNorm()

    # Very small x values (avoid division by zero)
    x_small = torch.tensor([1e-8, 1e-6, 1e-4])
    y = torch.tensor([0.5, 0.5, 0.5])

    result = tnorm.implication(x_small, y)

    # Should not produce NaN or Inf
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()

    # When x is very small and x < y, result should be 1
    # When x > y (not applicable here), result should be y/x
    # All x_small < y, so all should be 1
    assert torch.allclose(result, torch.ones_like(x_small))


def test_r_product_and_same_as_product() -> None:
    """Test R-Product AND is same as standard product: x * y."""
    tnorm = RProductTNorm()

    x = torch.tensor([0.3, 0.5, 0.7, 0.9])
    y = torch.tensor([0.4, 0.6, 0.8, 1.0])

    result = tnorm.conjunction(torch.stack([x, y]))
    expected = x * y

    assert torch.allclose(result, expected)


def test_r_product_or_same_as_product() -> None:
    """Test R-Product OR is same as probabilistic sum: x + y - x*y."""
    tnorm = RProductTNorm()

    x = torch.tensor([0.3, 0.5, 0.7, 0.9])
    y = torch.tensor([0.4, 0.6, 0.8, 1.0])

    result = tnorm.disjunction(torch.stack([x, y]))
    expected = x + y - x * y

    assert torch.allclose(result, expected)


def test_r_product_not() -> None:
    """Test R-Product NOT is standard negation: 1 - x."""
    tnorm = RProductTNorm()

    x = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])

    result = tnorm.negation(x)
    expected = 1.0 - x

    assert torch.allclose(result, expected)


def test_r_product_self_consistency() -> None:
    """Test R-Product is self-consistent: P <-> P = 1 for all P.

    This is Proposition 1 from the paper: R-Product satisfies
    self-consistency for all formulas.
    """
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = sp.Equivalent(P(X), P(X))

    # Test with various predicate values
    for p_value in [0.0, 0.3, 0.5, 0.7, 1.0]:
        predicates = {
            "P": Predicate("P", lambda x, val=p_value: torch.ones(x.shape[0]) * val)
        }

        logic_loss = logic_to_loss(expr, predicates, tnorm=RProductTNorm())
        x = torch.randn(1, 5)
        satisfaction = logic_loss.satisfaction(X=x)

        # P <-> P should always be 1 (perfect satisfaction)
        assert torch.allclose(satisfaction, torch.tensor(1.0), atol=1e-5)


def test_r_product_implication_tautology() -> None:
    """Test R-Product satisfies P -> P = 1 for all P."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")
    expr = sp.Implies(P(X), P(X))

    # Test with various predicate values
    for p_value in [0.0, 0.3, 0.5, 0.7, 1.0]:
        predicates = {
            "P": Predicate("P", lambda x, val=p_value: torch.ones(x.shape[0]) * val)
        }

        logic_loss = logic_to_loss(expr, predicates, tnorm=RProductTNorm())
        x = torch.randn(1, 5)
        satisfaction = logic_loss.satisfaction(X=x)

        # P -> P should always be 1
        assert torch.allclose(satisfaction, torch.tensor(1.0), atol=1e-5)


def test_r_product_modus_ponens() -> None:
    """Test R-Product satisfies modus ponens: (P ∧ (P → Q)) → Q."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P, Q = Symbol("P Q")
    # Modus ponens: ((P AND (P -> Q)) -> Q) should be a tautology
    expr = sp.Implies(sp.And(P(X), sp.Implies(P(X), Q(X))), Q(X))

    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = logic_to_loss(expr, predicates, tnorm=RProductTNorm())
    x = torch.randn(1, 5)
    satisfaction = logic_loss.satisfaction(X=x)

    # Modus ponens should be satisfied
    assert (satisfaction >= 0.9).all()  # Very high satisfaction expected


def test_r_product_gradient_flow_implication() -> None:
    """Test gradients flow through R-Product implication."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q = Symbol("P Q")

    expr = sp.Implies(P(X), Q(X))

    model_p = nn.Linear(5, 1)
    model_q = nn.Linear(5, 1)

    predicates = {
        "P": Predicate(lambda x: torch.sigmoid(model_p(x).squeeze(-1))),
        "Q": Predicate(lambda x: torch.sigmoid(model_q(x).squeeze(-1))),
    }

    logic_loss = logic_to_loss(expr, predicates, tnorm=RProductTNorm())

    x = torch.randn(1, 5)
    loss = logic_loss.loss(X=x)
    loss.backward()  # type: ignore[no-untyped-call]

    # Both models should have gradients
    for param in model_p.parameters():
        assert param.grad is not None
        # Gradients may be zero in some cases due to conditional

    for param in model_q.parameters():
        assert param.grad is not None


def test_r_product_gradient_flow_complex() -> None:
    """Test gradients flow through complex expression with R-Product."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q, R = Symbol("P Q R")
    expr = sp.And(sp.Implies(P(X), Q(X)), sp.Implies(Q(X), R(X)))

    model_p = nn.Linear(5, 1)
    model_q = nn.Linear(5, 1)
    model_r = nn.Linear(5, 1)

    predicates = {
        "P": Predicate(lambda x: torch.sigmoid(model_p(x).squeeze(-1))),
        "Q": Predicate(lambda x: torch.sigmoid(model_q(x).squeeze(-1))),
        "R": Predicate(lambda x: torch.sigmoid(model_r(x).squeeze(-1))),
    }

    logic_loss = logic_to_loss(expr, predicates, tnorm=RProductTNorm())

    x = torch.randn(1, 5)
    loss = logic_loss.loss(X=x)
    loss.backward()  # type: ignore[no-untyped-call]

    # All models should have gradients (may be zero in some cases)
    for param in model_p.parameters():
        assert param.grad is not None

    for param in model_q.parameters():
        assert param.grad is not None

    for param in model_r.parameters():
        assert param.grad is not None


def test_r_product_with_batch_dimensions() -> None:
    """Test R-Product works correctly with different batch sizes."""
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q = Symbol("P Q")

    expr = sp.Implies(P(X), Q(X))

    predicates = {
        "P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1))),
        "Q": Predicate(lambda x: torch.sigmoid(x.mean(dim=-1))),
    }

    logic_loss = logic_to_loss(expr, predicates, tnorm=RProductTNorm())

    # Test with different batch sizes using quantify='none' for per-batch results
    for batch_size in [1, 5, 10]:
        x = torch.randn(batch_size, 5)
        satisfaction = logic_loss.satisfaction(X=x, quantify='none')

        assert satisfaction.shape == (batch_size,)
        assert satisfaction.min() >= 0.0
        assert satisfaction.max() <= 1.0


def test_r_product_implication_with_constants() -> None:
    """Test R-Product implication with boolean constants."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P = Symbol("P")

    predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

    x = torch.randn(1, 3)

    # true -> P: should equal P (since true=1, 1 <= 0.7 is false, so 0.7/1)
    # Wait, let me recalculate: true=1, P=0.7
    # implication(1, 0.7): 1 <= 0.7? No, so return 0.7/1 = 0.7
    expr_true_p = sp.Implies(sp.true, P(X))
    logic_loss_true_p = logic_to_loss(expr_true_p, predicates, tnorm=RProductTNorm())
    satisfaction_true_p = logic_loss_true_p.satisfaction(X=x)
    assert torch.allclose(satisfaction_true_p, torch.tensor(0.7), atol=1e-5)

    # false -> P: should be 1 (since false=0, 0 <= P is true)
    expr_false_p = sp.Implies(sp.false, P(X))
    logic_loss_false_p = logic_to_loss(expr_false_p, predicates, tnorm=RProductTNorm())
    satisfaction_false_p = logic_loss_false_p.satisfaction(X=x)
    assert torch.allclose(satisfaction_false_p, torch.tensor(1.0), atol=1e-5)

    # P -> true: should be 1 (since 0.7 <= 1 is true)
    expr_p_true = sp.Implies(P(X), sp.true)
    logic_loss_p_true = logic_to_loss(expr_p_true, predicates, tnorm=RProductTNorm())
    satisfaction_p_true = logic_loss_p_true.satisfaction(X=x)
    assert torch.allclose(satisfaction_p_true, torch.tensor(1.0), atol=1e-5)

    # P -> false: R-Product gives 0/0.7 = 0 when evaluated directly
    # But with SymPy constant evaluation, this becomes NOT(P) = 0.3
    expr_p_false = sp.Implies(P(X), sp.false)
    logic_loss_p_false = logic_to_loss(expr_p_false, predicates, tnorm=RProductTNorm())
    satisfaction_p_false = logic_loss_p_false.satisfaction(X=x)
    # SymPy simplifies P -> false to NOT(P), so expect 1 - 0.7 = 0.3
    assert torch.allclose(satisfaction_p_false, torch.tensor(0.3), atol=1e-5)


def test_r_product_transitive_implication() -> None:
    """Test R-Product with transitive implications.

    Tests: (P->Q ∧ Q->R) -> (P->R).
    """
    # pylint: disable=invalid-name
    X = Variable("X")

    P, Q, R = Symbol("P Q R")
    # Transitivity: ((P->Q) AND (Q->R)) -> (P->R)
    expr = sp.Implies(
        sp.And(sp.Implies(P(X), Q(X)), sp.Implies(Q(X), R(X))), sp.Implies(P(X), R(X))
    )

    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        "R": Predicate(lambda x: torch.ones(x.shape[0]) * 0.4),
    }

    logic_loss = logic_to_loss(expr, predicates, tnorm=RProductTNorm())
    x = torch.randn(1, 5)
    satisfaction = logic_loss.satisfaction(X=x)

    # Transitivity should have high satisfaction
    assert (satisfaction >= 0.8).all()


def test_r_product_contrapositive() -> None:
    """Test R-Product with contrapositive: (P->Q) -> (~Q->~P)."""
    X = Variable("X")
    # pylint: disable=invalid-name
    P, Q = Symbol("P Q")
    # Contrapositive: (P->Q) -> (NOT Q -> NOT P)
    expr = sp.Implies(sp.Implies(P(X), Q(X)), sp.Implies(sp.Not(Q(X)), sp.Not(P(X))))

    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7),
        "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5),
    }

    logic_loss = logic_to_loss(expr, predicates, tnorm=RProductTNorm())
    x = torch.randn(1, 5)
    satisfaction = logic_loss.satisfaction(X=x)

    # Contrapositive should have reasonably high satisfaction
    assert (satisfaction >= 0.7).all()


def test_r_product_equivalent_decomposition() -> None:
    """Test R-Product EQUIVALENT decomposes correctly.

    Tests: (P<->Q) = (P->Q)∧(Q->P).
    """
    X = Variable("X")
    # pylint: disable=invalid-name
    P, Q = Symbol("P Q")

    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    x = torch.randn(1, 5)

    # Test P <-> Q
    expr_equiv = sp.Equivalent(P(X), Q(X))
    logic_loss_equiv = logic_to_loss(expr_equiv, predicates, tnorm=RProductTNorm())
    satisfaction_equiv = logic_loss_equiv.satisfaction(X=x)

    # Test (P->Q) AND (Q->P)
    expr_decomposed = sp.And(sp.Implies(P(X), Q(X)), sp.Implies(Q(X), P(X)))
    logic_loss_decomposed = logic_to_loss(
        expr_decomposed, predicates, tnorm=RProductTNorm()
    )
    satisfaction_decomposed = logic_loss_decomposed.satisfaction(X=x)

    # Should be equal
    assert torch.allclose(satisfaction_equiv, satisfaction_decomposed, atol=1e-5)
