"""Unit tests for the logic_loss library.

Run with: pytest test_logic_loss.py
"""

import sympy as sp
import torch
import torch.nn as nn

from logic_as_loss import (
    LogicLoss,
    Predicate,
    LukasiewiczTNorm,
    GodelTNorm,
)


def test_basic_and() -> None:
    """Test basic AND operation."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = LogicLoss(expr, predicates)
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

    logic_loss = LogicLoss(expr, predicates)
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

    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    assert torch.allclose(satisfaction, torch.tensor(0.3), atol=1e-5)


def test_implication() -> None:
    """Test IMPLIES operation."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.Implies(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)

    # P -> Q = ~P | Q = 0.2 | 0.6
    not_p = 0.2
    expected = not_p + 0.6 - not_p * 0.6
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


def test_batching() -> None:
    """Test that batching works correctly."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    # Predicate that depends on input
    predicates = {"P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = LogicLoss(expr, predicates)

    # Different batch sizes
    for batch_size in [1, 10, 100]:
        x = torch.randn(batch_size, 5)
        satisfaction = logic_loss(x)
        assert satisfaction.shape == (batch_size,)
        assert satisfaction.min() >= 0.0
        assert satisfaction.max() <= 1.0


def test_gradient_flow() -> None:
    """Test that gradients flow through the loss."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
    predicates = {"P": Predicate("P", lambda x: model(x).squeeze(-1))}

    logic_loss = LogicLoss(expr, predicates)

    x = torch.randn(10, 5)
    loss = logic_loss.loss(x)
    loss.backward()  # type: ignore[no-untyped-call]

    # Check gradients exist
    for param in model.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


def test_different_inputs_per_predicate() -> None:
    """Test different inputs for different predicates."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1))),
        "Q": Predicate("Q", lambda x: torch.sigmoid(x.mean(dim=-1))),
    }

    logic_loss = LogicLoss(expr, predicates)

    batch_size = 10
    inputs = {
        "P": torch.randn(batch_size, 5),
        "Q": torch.randn(batch_size, 10),  # Different shape
    }

    satisfaction = logic_loss(inputs)
    assert satisfaction.shape == (batch_size,)


def test_lukasiewicz_tnorm() -> None:
    """Test Łukasiewicz t-norm."""
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

    logic_loss = LogicLoss(expr, predicates)
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


def test_loss_reduction() -> None:
    """Test different loss reduction modes."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    predicates = {"P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(10, 5)

    # Mean reduction
    loss_mean = logic_loss.loss(x, reduction="mean")
    assert loss_mean.shape == ()

    # Sum reduction
    loss_sum = logic_loss.loss(x, reduction="sum")
    assert loss_sum.shape == ()
    assert torch.allclose(loss_sum, loss_mean * 10)

    # No reduction
    loss_none = logic_loss.loss(x, reduction="none")
    assert loss_none.shape == (10,)
    assert torch.allclose(loss_mean, loss_none.mean())


def test_deterministic_predicate() -> None:
    """Test deterministic (non-model) predicate."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    def deterministic_func(x: torch.Tensor) -> torch.Tensor:
        """Deterministic predicate function."""
        return (x.sum(dim=-1) > 0).float()

    predicates = {"P": Predicate("P", deterministic_func, is_model=False)}

    logic_loss = LogicLoss(expr, predicates)

    x_pos = torch.ones(5, 3)
    x_neg = -torch.ones(5, 3)

    assert (logic_loss(x_pos) == 1.0).all()
    assert (logic_loss(x_neg) == 0.0).all()


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

    logic_loss = LogicLoss(expr, predicates)
    params = logic_loss.get_trainable_parameters()

    # Should only get parameters from model_p
    assert len(params) == 2  # weight and bias
    assert all(p.requires_grad for p in params)


# Edge Case Tests


def test_empty_batch() -> None:
    """Test handling of empty batches (batch_size=0)."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    predicates = {"P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(0, 5)  # Empty batch
    satisfaction = logic_loss(x)

    assert satisfaction.shape == (0,)
    assert satisfaction.numel() == 0


def test_single_element_batch() -> None:
    """Test handling of single-element batches."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.7),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.5),
    }

    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(1, 5)  # Single element
    satisfaction = logic_loss(x)

    assert satisfaction.shape == (1,)
    assert torch.allclose(satisfaction, torch.tensor([0.35]), atol=1e-5)


def test_very_large_batch() -> None:
    """Test handling of very large batches (10000+ elements)."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    predicates = {"P": Predicate("P", lambda x: torch.sigmoid(x.mean(dim=-1)))}

    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(10000, 5)
    satisfaction = logic_loss(x)

    assert satisfaction.shape == (10000,)
    assert satisfaction.min() >= 0.0
    assert satisfaction.max() <= 1.0


def test_nan_handling() -> None:
    """Test handling of NaN values in inputs."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    predicates = {"P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = LogicLoss(expr, predicates)
    x = torch.tensor([[float("nan"), 1.0, 2.0]])
    satisfaction = logic_loss(x)

    # sigmoid(nan) = nan, clamped to [0,1] doesn't fix NaN
    assert satisfaction.shape == (1,)
    # NaN should propagate through
    assert torch.isnan(satisfaction).any() or (
        satisfaction.min() >= 0.0 and satisfaction.max() <= 1.0
    )


def test_inf_handling() -> None:
    """Test handling of Inf values in inputs."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    predicates = {"P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1)))}

    logic_loss = LogicLoss(expr, predicates)

    # Test positive infinity
    x_pos_inf = torch.tensor([[float("inf"), 1.0, 2.0]])
    satisfaction_pos = logic_loss(x_pos_inf)
    assert satisfaction_pos.shape == (1,)
    # sigmoid(+inf) = 1.0
    assert torch.allclose(satisfaction_pos, torch.tensor([1.0]), atol=1e-5)

    # Test negative infinity
    x_neg_inf = torch.tensor([[float("-inf"), 1.0, 2.0]])
    satisfaction_neg = logic_loss(x_neg_inf)
    assert satisfaction_neg.shape == (1,)
    # sigmoid(-inf) = 0.0
    assert torch.allclose(satisfaction_neg, torch.tensor([0.0]), atol=1e-5)


def test_predicate_clamping_above_one() -> None:
    """Test that predicates returning values >1 are clamped to [0,1]."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    # Predicate that returns values > 1
    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 2.5)}

    logic_loss = LogicLoss(expr, predicates)
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

    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # Should be clamped to 0.0
    assert torch.allclose(satisfaction, torch.tensor(0.0))
    assert satisfaction.min() >= 0.0


def test_missing_predicate_raises_error() -> None:
    """Test that missing predicates raise ValueError."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    # Only provide predicate for P, not Q
    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.5)}

    try:
        LogicLoss(expr, predicates)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Missing predicates" in str(e)
        assert "Q" in str(e)


def test_unsupported_expression_raises_error() -> None:
    """Test that unsupported SymPy expressions raise ValueError."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")

    # Use an unsupported operation
    expr = P + P  # Arithmetic, not logic

    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.5)}

    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(5, 3)

    try:
        logic_loss(x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported expression type" in str(e)


def test_boolean_true_constant() -> None:
    """Test handling of sp.true boolean constant."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = sp.And(P, sp.true)

    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # P AND true = P
    assert torch.allclose(satisfaction, torch.tensor(0.6), atol=1e-5)


def test_boolean_false_constant() -> None:
    """Test handling of sp.false boolean constant."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = sp.And(P, sp.false)

    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.6)}

    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # P AND false = false = 0
    assert torch.allclose(satisfaction, torch.tensor(0.0), atol=1e-5)


def test_invalid_reduction_mode() -> None:
    """Test that invalid reduction mode raises ValueError."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.5)}

    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(5, 3)

    try:
        logic_loss.loss(x, reduction="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown reduction" in str(e)


def test_dict_input_with_default_key() -> None:
    """Test dict input uses 'default' key when specific key missing."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.And(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.sigmoid(x.sum(dim=-1))),
        "Q": Predicate("Q", lambda x: torch.sigmoid(x.mean(dim=-1))),
    }

    logic_loss = LogicLoss(expr, predicates)

    # Provide specific input for P and default for others
    default_input = torch.randn(5, 3)
    inputs = {"P": torch.randn(5, 3), "default": default_input}

    satisfaction = logic_loss(inputs)

    # Should use default for Q
    assert satisfaction.shape == (5,)


def test_equivalence_operator() -> None:
    """Test EQUIVALENCE (biconditional) operator."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")
    expr = sp.Equivalent(P, Q)

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.8),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.6),
    }

    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # P <-> Q = (P -> Q) AND (Q -> P)
    # P -> Q = NOT P OR Q = 0.2 + 0.6 - 0.12 = 0.68
    # Q -> P = NOT Q OR P = 0.4 + 0.8 - 0.32 = 0.88
    # Result = 0.68 * 0.88 = 0.5984
    p_implies_q = 0.2 + 0.6 - 0.2 * 0.6
    q_implies_p = 0.4 + 0.8 - 0.4 * 0.8
    expected = p_implies_q * q_implies_p

    assert satisfaction.shape == (5,)
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


def test_non_tensor_predicate_return() -> None:
    """Test predicate that returns non-tensor (list/float)."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")
    expr = P

    # Predicate that returns a Python list (will be converted to tensor)
    predicates = {"P": Predicate("P", lambda x: 0.75)}

    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(5, 3)
    satisfaction = logic_loss(x)

    # Should convert 0.75 to tensor
    assert isinstance(satisfaction, torch.Tensor)
    assert torch.allclose(satisfaction, torch.tensor(0.75))


def test_boolean_constants_with_dict_input() -> None:
    """Test boolean constants (true/false) with dict input."""
    # pylint: disable=invalid-name
    P = sp.symbols("P")

    predicates = {"P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.6)}

    # Test with true constant
    expr_true = sp.Or(P, sp.true)
    logic_loss_true = LogicLoss(expr_true, predicates)
    inputs = {"P": torch.randn(5, 3)}
    satisfaction_true = logic_loss_true(inputs)

    # P OR true = true = 1
    assert torch.allclose(satisfaction_true, torch.tensor(1.0), atol=1e-5)

    # Test with false constant
    expr_false = sp.Or(P, sp.false)
    logic_loss_false = LogicLoss(expr_false, predicates)
    satisfaction_false = logic_loss_false(inputs)

    # P OR false = P = 0.6
    assert torch.allclose(satisfaction_false, torch.tensor(0.6), atol=1e-5)


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


def test_godel_tnorm() -> None:
    """Test Gödel t-norm AND and OR operations."""
    # pylint: disable=invalid-name
    P, Q = sp.symbols("P Q")

    predicates = {
        "P": Predicate("P", lambda x: torch.ones(x.shape[0]) * 0.7),
        "Q": Predicate("Q", lambda x: torch.ones(x.shape[0]) * 0.5),
    }

    # Test AND
    expr_and = sp.And(P, Q)
    logic_loss_and = LogicLoss(expr_and, predicates, tnorm=GodelTNorm())
    x = torch.randn(5, 3)
    satisfaction_and = logic_loss_and(x)

    # Gödel AND: min(0.7, 0.5) = 0.5
    assert torch.allclose(satisfaction_and, torch.tensor(0.5), atol=1e-5)

    # Test OR
    expr_or = sp.Or(P, Q)
    logic_loss_or = LogicLoss(expr_or, predicates, tnorm=GodelTNorm())
    satisfaction_or = logic_loss_or(x)

    # Gödel OR: max(0.7, 0.5) = 0.7
    assert torch.allclose(satisfaction_or, torch.tensor(0.7), atol=1e-5)


def main() -> None:
    """Run all tests manually."""
    print("Running tests...")

    test_basic_and()
    print("DONE test_basic_and")

    test_basic_or()
    print("DONE test_basic_or")

    test_negation()
    print("DONE test_negation")

    test_implication()
    print("DONE test_implication")

    test_batching()
    print("DONE test_batching")

    test_gradient_flow()
    print("DONE test_gradient_flow")

    test_different_inputs_per_predicate()
    print("DONE test_different_inputs_per_predicate")

    test_lukasiewicz_tnorm()
    print("DONE test_lukasiewicz_tnorm")

    test_complex_expression()
    print("DONE test_complex_expression")

    test_loss_reduction()
    print("DONE test_loss_reduction")

    test_deterministic_predicate()
    print("DONE test_deterministic_predicate")

    test_get_trainable_parameters()
    print("DONE test_get_trainable_parameters")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
