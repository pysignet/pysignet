"""Tests for LinearThresholdUnitCompiler.

This module tests the LTU-based compilation strategy, which represents
logical operations as linear threshold units.
"""

import pytest
import sympy as sp
import torch
import torch.nn as nn

from pysignet import LogicLoss, Symbol, Variable
from pysignet.compilation import LinearThresholdUnitCompiler


class TestBasicOperations:
    """Tests for basic logical operations using LTU compiler."""

    def test_simple_conjunction_soft(self):
        """Test AND operation with soft mode (sigmoid)."""
        X = Variable("X")
        P, Q = Symbol("P Q")

        expr = sp.And(P(X), Q(X))

        # Fixed values for testing
        p_model = lambda x: torch.full((x.shape[0],), 0.8)
        q_model = lambda x: torch.full((x.shape[0],), 0.9)

        compiler = LinearThresholdUnitCompiler(mode="soft")
        compiled = compiler.compile(expr, {"P": p_model, "Q": q_model})

        batch_size = 4
        x = torch.randn(batch_size, 5)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (batch_size,)
        # With sigmoid threshold, high values should give high output
        assert torch.all(result > 0.5)

    def test_simple_disjunction_soft(self):
        """Test OR operation with soft mode."""
        X = Variable("X")
        P, Q = Symbol("P Q")

        expr = sp.Or(P(X), Q(X))

        p_model = lambda x: torch.full((x.shape[0],), 0.1)
        q_model = lambda x: torch.full((x.shape[0],), 0.2)

        compiler = LinearThresholdUnitCompiler(mode="soft")
        compiled = compiler.compile(expr, {"P": p_model, "Q": q_model})

        batch_size = 3
        x = torch.randn(batch_size, 5)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (batch_size,)
        # Sum is 0.3, threshold is 0.5, so should be < 0.5
        assert torch.all(result < 0.5)

    def test_negation(self):
        """Test NOT operation."""
        X = Variable("X")
        P = Symbol("P")

        expr = sp.Not(P(X))

        p_model = lambda x: torch.full((x.shape[0],), 0.7)

        compiler = LinearThresholdUnitCompiler()
        compiled = compiler.compile(expr, {"P": p_model})

        x = torch.randn(1, 5)
        result = compiled(X=x)

        # 1 - 0.7 = 0.3
        assert torch.allclose(result, torch.full((2,), 0.3))

    def test_hard_mode_conjunction(self):
        """Test AND with hard mode (step function)."""
        X = Variable("X")
        P, Q = Symbol("P Q")

        expr = sp.And(P(X), Q(X))

        p_model = lambda x: torch.full((x.shape[0],), 0.8)
        q_model = lambda x: torch.full((x.shape[0],), 0.9)

        compiler = LinearThresholdUnitCompiler(mode="hard")
        compiled = compiler.compile(expr, {"P": p_model, "Q": q_model})

        x = torch.randn(1, 5)
        result = compiled(X=x)

        # Sum is 1.7, threshold is 1.5, so output is 1.0
        assert torch.allclose(result, torch.ones(3))

    def test_hard_mode_disjunction(self):
        """Test OR with hard mode."""
        X = Variable("X")
        P, Q = Symbol("P Q")

        expr = sp.Or(P(X), Q(X))

        p_model = lambda x: torch.full((x.shape[0],), 0.2)
        q_model = lambda x: torch.full((x.shape[0],), 0.1)

        compiler = LinearThresholdUnitCompiler(mode="hard")
        compiled = compiler.compile(expr, {"P": p_model, "Q": q_model})

        x = torch.randn(1, 5)
        result = compiled(X=x)

        # Sum is 0.3, threshold is 0.5, so output is 0.0
        assert torch.allclose(result, torch.zeros(3))


class TestImplicationAndEquivalence:
    """Tests for implication and equivalence."""

    def test_implication(self):
        """Test L => R compiles as (NOT L) OR R."""
        X = Variable("X")
        P, Q = Symbol("P Q")

        expr = sp.Implies(P(X), Q(X))

        p_model = lambda x: torch.full((x.shape[0],), 0.3)
        q_model = lambda x: torch.full((x.shape[0],), 0.8)

        compiler = LinearThresholdUnitCompiler()
        compiled = compiler.compile(expr, {"P": p_model, "Q": q_model})

        x = torch.randn(1, 5)
        result = compiled(X=x)

        # (NOT 0.3) OR 0.8 = 0.7 OR 0.8
        # Sum is 1.5, threshold is 0.5, so sigmoid(1.0*(1.5-0.5)) = sigmoid(1.0) = 0.7311
        assert torch.allclose(result, torch.full((2,), 0.7311), atol=1e-4)

    def test_equivalence(self):
        """Test L <=> R compiles as (L => R) AND (R => L)."""
        X = Variable("X")
        P, Q = Symbol("P Q")

        expr = sp.Equivalent(P(X), Q(X))

        p_model = lambda x: torch.full((x.shape[0],), 0.6)
        q_model = lambda x: torch.full((x.shape[0],), 0.6)

        compiler = LinearThresholdUnitCompiler()
        compiled = compiler.compile(expr, {"P": p_model, "Q": q_model})

        batch_size = 4
        x = torch.randn(batch_size, 5)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (batch_size,)
        # (P => Q) AND (Q => P)
        # Each implication: (NOT 0.6) OR 0.6 = 0.4 OR 0.6 -> sigmoid(0.5) = 0.6225
        # AND of two 0.6225 values: sum=1.245, threshold=1.5 -> sigmoid(-0.255) = 0.4366
        assert torch.allclose(result, torch.full((batch_size,), 0.4366), atol=1e-4)


class TestWithVariables:
    """Tests for expressions with FOL variables."""

    def test_simple_variable_expression(self):
        """Test expression with one variable."""
        X = Variable("X")
        P = Symbol("P")

        expr = P(X)

        p_model = lambda x: torch.sigmoid(x.sum(dim=-1))

        compiler = LinearThresholdUnitCompiler()
        compiled = compiler.compile(expr, {"P": p_model})

        batch_size = 4
        x = torch.randn(batch_size, 5)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (batch_size,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_two_variables_different_predicates(self):
        """Test expression with two variables."""
        X, Y = Variable("X Y")
        P, Q = Symbol("P Q")

        expr = sp.And(P(X), Q(Y))

        p_model = lambda x: torch.sigmoid(x.sum(dim=-1))
        q_model = lambda y: torch.sigmoid(y.sum(dim=-1))

        compiler = LinearThresholdUnitCompiler()
        compiled = compiler.compile(expr, {"P": p_model, "Q": q_model})

        batch_size = 3
        x = torch.randn(batch_size, 5)
        y = torch.randn(batch_size, 4)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, Y=y, quantify='none')

        assert result.shape == (batch_size,)


class TestQuantifiers:
    """Tests for quantifier expansion."""

    def test_forall_expansion(self):
        """Test ForAll quantifier expands to conjunction."""
        from pysignet.logic import ForAll

        X, Y = Variable("X Y")
        Digit = Symbol("Digit")

        expr = ForAll(Y, [0, 1, 2], Digit(X, Y))

        model = nn.Sequential(nn.Linear(5, 10), nn.Softmax(dim=-1))

        compiler = LinearThresholdUnitCompiler()
        compiled = compiler.compile(expr, {"Digit": model})

        batch_size = 3
        x = torch.randn(batch_size, 5)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (batch_size,)

    def test_exists_expansion(self):
        """Test Exists quantifier expands to disjunction."""
        from pysignet.logic import Exists

        X, Y = Variable("X Y")
        Digit = Symbol("Digit")

        expr = Exists(Y, [0, 1], Digit(X, Y))

        model = nn.Sequential(nn.Linear(5, 10), nn.Softmax(dim=-1))

        compiler = LinearThresholdUnitCompiler()
        compiled = compiler.compile(expr, {"Digit": model})

        batch_size = 4
        x = torch.randn(batch_size, 5)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (batch_size,)


class TestGradientFlow:
    """Tests for gradient flow (soft mode only)."""

    def test_gradients_flow_soft_mode(self):
        """Test gradients flow in soft mode."""
        X = Variable("X")
        P = Symbol("P")

        expr = sp.Not(P(X))

        model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())

        compiler = LinearThresholdUnitCompiler(mode="soft")
        compiled_fn = compiler.compile(expr, {"P": model})

        # Wrap in LogicLoss for loss computation
        logic_loss = LogicLoss(compiled_fn)

        x = torch.randn(1, 5)
        loss = logic_loss.loss(X=x)

        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_hard_mode_not_differentiable(self):
        """Test that hard mode uses non-differentiable operations."""
        X = Variable("X")
        P, Q = Symbol("P Q")

        expr = sp.And(P(X), Q(X))

        # Use requires_grad to test differentiability
        p_model = lambda x: torch.full((x.shape[0],), 0.8, requires_grad=True)
        q_model = lambda x: torch.full((x.shape[0],), 0.9, requires_grad=True)

        compiler = LinearThresholdUnitCompiler(mode="hard")
        compiled = compiler.compile(expr, {"P": p_model, "Q": q_model})

        x = torch.randn(1, 5)
        result = compiled(X=x)

        # Result uses step function, so no grad_fn
        assert result.grad_fn is None


class TestEdgeCases:
    """Tests for edge cases."""

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            LinearThresholdUnitCompiler(mode="invalid")

    def test_boolean_constants(self):
        """Test boolean constants work correctly."""
        X = Variable("X")
        P = Symbol("P")

        expr = sp.And(P(X), sp.true)

        p_model = lambda x: torch.full((x.shape[0],), 0.7)

        compiler = LinearThresholdUnitCompiler()
        compiled = compiler.compile(expr, {"P": p_model})

        x = torch.randn(1, 5)
        result = compiled(X=x)

        # P AND TRUE should be high (sum 1.7, threshold 1.5)
        assert torch.all(result > 0.5)

    def test_multiple_disjuncts(self):
        """Test disjunction with many terms."""
        X = Variable("X")
        P1, P2, P3, P4 = Symbol("P1 P2 P3 P4")

        expr = sp.Or(P1(X), P2(X), P3(X), P4(X))

        p1_model = lambda x: torch.full((x.shape[0],), 0.1)
        p2_model = lambda x: torch.full((x.shape[0],), 0.1)
        p3_model = lambda x: torch.full((x.shape[0],), 0.1)
        p4_model = lambda x: torch.full((x.shape[0],), 0.1)

        compiler = LinearThresholdUnitCompiler()
        compiled = compiler.compile(
            expr, {"P1": p1_model, "P2": p2_model, "P3": p3_model, "P4": p4_model}
        )

        x = torch.randn(1, 5)
        result = compiled(X=x)

        # Sum is 0.4, threshold is 0.5, so output < 0.5
        assert torch.all(result < 0.5)
