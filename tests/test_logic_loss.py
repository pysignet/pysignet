"""
Unit tests for the logic_loss library.
Run with: pytest test_logic_loss.py
"""

import torch
import torch.nn as nn
import sympy as sp
from logic_loss import LogicLoss, Predicate, ProductTNorm, LukasiewiczTNorm


def test_basic_and():
    """Test basic AND operation."""
    P, Q = sp.symbols('P Q')
    expr = sp.And(P, Q)
    
    predicates = {
        'P': Predicate('P', lambda x: torch.ones(x.shape[0]) * 0.8),
        'Q': Predicate('Q', lambda x: torch.ones(x.shape[0]) * 0.6)
    }
    
    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)
    
    # Product t-norm: 0.8 * 0.6 = 0.48
    assert satisfaction.shape == (10,)
    assert torch.allclose(satisfaction, torch.tensor(0.48), atol=1e-5)


def test_basic_or():
    """Test basic OR operation."""
    P, Q = sp.symbols('P Q')
    expr = sp.Or(P, Q)
    
    predicates = {
        'P': Predicate('P', lambda x: torch.ones(x.shape[0]) * 0.8),
        'Q': Predicate('Q', lambda x: torch.ones(x.shape[0]) * 0.6)
    }
    
    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)
    
    # Product t-conorm: 0.8 + 0.6 - 0.8*0.6 = 0.92
    expected = 0.8 + 0.6 - 0.8 * 0.6
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


def test_negation():
    """Test NOT operation."""
    P = sp.symbols('P')
    expr = sp.Not(P)
    
    predicates = {
        'P': Predicate('P', lambda x: torch.ones(x.shape[0]) * 0.7)
    }
    
    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)
    
    assert torch.allclose(satisfaction, torch.tensor(0.3), atol=1e-5)


def test_implication():
    """Test IMPLIES operation."""
    P, Q = sp.symbols('P Q')
    expr = sp.Implies(P, Q)
    
    predicates = {
        'P': Predicate('P', lambda x: torch.ones(x.shape[0]) * 0.8),
        'Q': Predicate('Q', lambda x: torch.ones(x.shape[0]) * 0.6)
    }
    
    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)
    
    # P -> Q = ~P | Q = 0.2 | 0.6
    not_p = 0.2
    expected = not_p + 0.6 - not_p * 0.6
    assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)


def test_batching():
    """Test that batching works correctly."""
    P = sp.symbols('P')
    expr = P
    
    # Predicate that depends on input
    predicates = {
        'P': Predicate('P', lambda x: torch.sigmoid(x.sum(dim=-1)))
    }
    
    logic_loss = LogicLoss(expr, predicates)
    
    # Different batch sizes
    for batch_size in [1, 10, 100]:
        x = torch.randn(batch_size, 5)
        satisfaction = logic_loss(x)
        assert satisfaction.shape == (batch_size,)
        assert satisfaction.min() >= 0.0
        assert satisfaction.max() <= 1.0


def test_gradient_flow():
    """Test that gradients flow through the loss."""
    P = sp.symbols('P')
    expr = P
    
    model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
    predicates = {
        'P': Predicate('P', lambda x: model(x).squeeze(-1))
    }
    
    logic_loss = LogicLoss(expr, predicates)
    
    x = torch.randn(10, 5)
    loss = logic_loss.loss(x)
    loss.backward()
    
    # Check gradients exist
    for param in model.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


def test_different_inputs_per_predicate():
    """Test different inputs for different predicates."""
    P, Q = sp.symbols('P Q')
    expr = sp.And(P, Q)
    
    predicates = {
        'P': Predicate('P', lambda x: torch.sigmoid(x.sum(dim=-1))),
        'Q': Predicate('Q', lambda x: torch.sigmoid(x.mean(dim=-1)))
    }
    
    logic_loss = LogicLoss(expr, predicates)
    
    batch_size = 10
    inputs = {
        'P': torch.randn(batch_size, 5),
        'Q': torch.randn(batch_size, 10)  # Different shape
    }
    
    satisfaction = logic_loss(inputs)
    assert satisfaction.shape == (batch_size,)


def test_lukasiewicz_tnorm():
    """Test Łukasiewicz t-norm."""
    P, Q = sp.symbols('P Q')
    expr = sp.And(P, Q)
    
    predicates = {
        'P': Predicate('P', lambda x: torch.ones(x.shape[0]) * 0.8),
        'Q': Predicate('Q', lambda x: torch.ones(x.shape[0]) * 0.6)
    }
    
    logic_loss = LogicLoss(expr, predicates, tnorm=LukasiewiczTNorm())
    x = torch.randn(10, 5)
    satisfaction = logic_loss(x)
    
    # Łukasiewicz AND: max(0, 0.8 + 0.6 - 1) = 0.4
    assert torch.allclose(satisfaction, torch.tensor(0.4), atol=1e-5)


def test_complex_expression():
    """Test complex nested expression."""
    P, Q, R = sp.symbols('P Q R')
    expr = sp.And(sp.Or(P, Q), sp.Not(R))
    
    predicates = {
        'P': Predicate('P', lambda x: torch.ones(x.shape[0]) * 0.5),
        'Q': Predicate('Q', lambda x: torch.ones(x.shape[0]) * 0.5),
        'R': Predicate('R', lambda x: torch.ones(x.shape[0]) * 0.3)
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


def test_loss_reduction():
    """Test different loss reduction modes."""
    P = sp.symbols('P')
    expr = P
    
    predicates = {
        'P': Predicate('P', lambda x: torch.sigmoid(x.sum(dim=-1)))
    }
    
    logic_loss = LogicLoss(expr, predicates)
    x = torch.randn(10, 5)
    
    # Mean reduction
    loss_mean = logic_loss.loss(x, reduction='mean')
    assert loss_mean.shape == ()
    
    # Sum reduction
    loss_sum = logic_loss.loss(x, reduction='sum')
    assert loss_sum.shape == ()
    assert torch.allclose(loss_sum, loss_mean * 10)
    
    # No reduction
    loss_none = logic_loss.loss(x, reduction='none')
    assert loss_none.shape == (10,)
    assert torch.allclose(loss_mean, loss_none.mean())


def test_deterministic_predicate():
    """Test deterministic (non-model) predicate."""
    P = sp.symbols('P')
    expr = P
    
    def deterministic_func(x):
        return (x.sum(dim=-1) > 0).float()
    
    predicates = {
        'P': Predicate('P', deterministic_func, is_model=False)
    }
    
    logic_loss = LogicLoss(expr, predicates)
    
    x_pos = torch.ones(5, 3)
    x_neg = -torch.ones(5, 3)
    
    assert (logic_loss(x_pos) == 1.0).all()
    assert (logic_loss(x_neg) == 0.0).all()


def test_get_trainable_parameters():
    """Test getting trainable parameters from models."""
    P, Q = sp.symbols('P Q')
    expr = sp.And(P, Q)
    
    model_P = nn.Linear(5, 1)
    
    predicates = {
        'P': Predicate('P', model_P),
        'Q': Predicate('Q', lambda x: (x > 0).float().mean(dim=-1))
    }
    
    logic_loss = LogicLoss(expr, predicates)
    params = logic_loss.get_trainable_parameters()
    
    # Should only get parameters from model_P
    assert len(params) == 2  # weight and bias
    assert all(p.requires_grad for p in params)


if __name__ == "__main__":
    print("Running tests...")
    
    test_basic_and()
    print("✓ test_basic_and")
    
    test_basic_or()
    print("✓ test_basic_or")
    
    test_negation()
    print("✓ test_negation")
    
    test_implication()
    print("✓ test_implication")
    
    test_batching()
    print("✓ test_batching")
    
    test_gradient_flow()
    print("✓ test_gradient_flow")
    
    test_different_inputs_per_predicate()
    print("✓ test_different_inputs_per_predicate")
    
    test_lukasiewicz_tnorm()
    print("✓ test_lukasiewicz_tnorm")
    
    test_complex_expression()
    print("✓ test_complex_expression")
    
    test_loss_reduction()
    print("✓ test_loss_reduction")
    
    test_deterministic_predicate()
    print("✓ test_deterministic_predicate")
    
    test_get_trainable_parameters()
    print("✓ test_get_trainable_parameters")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
