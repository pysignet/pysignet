"""Tests for automatic batch quantification with domain quantifiers.

This module tests that free variables are automatically universally quantified
over the batch dimension after domain quantifier expansion.
"""

import pytest
import torch
import torch.nn as nn
import sympy as sp

from pysignet import Symbol, compile_logic
from pysignet.logic import Variable, ForAll, Exists, expand_quantifier


class TestBasicAutoBatching:
    """Tests for basic automatic batch quantification."""

    @pytest.mark.skip(reason="Phase 2.6+: Batch reduction to scalar not implemented. "
                             "Returns (batch_size,), expects ().")
    def test_free_variable_auto_batched_after_expansion(self):
        """Free variable is auto-batched after domain expansion."""
        X, Y = Variable("X Y")
        Digit = Symbol("Digit")

        # ForAll(Y, [0, 1, 2], Digit(X, Y))
        # After expansion: Digit(X, 0) ∧ Digit(X, 1) ∧ Digit(X, 2)
        # X should be auto-batched
        expr = ForAll(Y, [0, 1, 2], Digit(X, Y))

        # Expand the quantifier
        expanded = expand_quantifier(expr)

        # Create a simple multi-class model
        model = nn.Sequential(nn.Linear(10, 10), nn.Softmax(dim=-1))

        # Compile
        predicates = {"Digit": model}
        compiled = compile_logic(expanded, predicates)

        # Evaluate with batch
        batch_size = 5
        x = torch.randn(batch_size, 10)

        # Should work - X is auto-batched
        result = compiled({"X": x})

        # Result should be a scalar (batch reduction via AND)
        assert result.shape == ()
        assert 0 <= result.item() <= 1

    @pytest.mark.skip(reason="Phase 2.6+: Batch reduction not implemented.")
    def test_single_free_variable_batched(self):
        """Single free variable is batched."""
        X = Variable("X")
        P = Symbol("P")

        # Simple expression with one free variable
        expr = P(X)

        # Model
        model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())

        # Compile
        predicates = {"P": model}
        compiled = compile_logic(expr, predicates)

        # Evaluate
        batch_size = 8
        x = torch.randn(batch_size, 5)
        result = compiled({"X": x})

        # Scalar result (batch reduction)
        assert result.shape == ()

    @pytest.mark.skip(reason="Phase 2.6+: Batch reduction not implemented.")
    def test_multiple_free_variables_batched(self):
        """Multiple free variables are batched together."""
        X, Y = Variable("X Y")
        P = Symbol("P")

        # P(X, Y) - both free
        expr = P(X, Y)

        # Binary predicate model
        model = lambda x, y: torch.sigmoid(x.sum(dim=-1) + y)

        # Compile
        predicates = {"P": model}
        compiled = compile_logic(expr, predicates)

        # Evaluate
        batch_size = 4
        x = torch.randn(batch_size, 3)
        y = torch.randint(0, 10, (batch_size,))

        result = compiled({"X": x, "Y": y})

        # Scalar result
        assert result.shape == ()


class TestMixedQuantification:
    """Tests for mixed domain and batch quantification."""

    @pytest.mark.skip(reason="Phase 2.6+: Batch reduction not implemented.")
    def test_domain_quantified_then_batch_quantified(self):
        """Domain quantifier expanded, then free var batch-quantified."""
        X, Y = Variable("X Y")
        Digit = Symbol("Digit")

        # ForAll(Y, [0, 1], Digit(X, Y))
        # Y is domain-quantified, X is batch-quantified
        expr = ForAll(Y, [0, 1], Digit(X, Y))

        # Expand
        expanded = expand_quantifier(expr)

        # Model
        model = nn.Sequential(nn.Linear(5, 10), nn.Softmax(dim=-1))

        # Compile
        predicates = {"Digit": model}
        compiled = compile_logic(expanded, predicates)

        # Evaluate
        batch_size = 3
        x = torch.randn(batch_size, 5)

        result = compiled({"X": x})

        # Should be scalar
        assert result.shape == ()
        assert 0 <= result.item() <= 1

    @pytest.mark.skip(reason="Phase 2.6+: Batch reduction not implemented.")
    def test_exists_with_free_variable(self):
        """Exists quantifier with free variable."""
        X, Y = Variable("X Y")
        Digit = Symbol("Digit")

        # Exists(Y, [0, 1, 2], Digit(X, Y))
        # "X is classified as digit 0, 1, or 2"
        expr = Exists(Y, [0, 1, 2], Digit(X, Y))

        # Expand
        expanded = expand_quantifier(expr)

        # Model
        model = nn.Sequential(nn.Linear(5, 10), nn.Softmax(dim=-1))

        # Compile
        predicates = {"Digit": model}
        compiled = compile_logic(expanded, predicates)

        # Evaluate
        batch_size = 4
        x = torch.randn(batch_size, 5)

        result = compiled({"X": x})

        # Scalar result
        assert result.shape == ()

    @pytest.mark.skip(reason="Phase 2.6+: Batch reduction not implemented.")
    def test_nested_quantifiers_with_free_variable(self):
        """Nested quantifiers with remaining free variable."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        # ForAll(Y, [0, 1], Exists(Z, [2, 3], P(X, Y, Z)))
        # X is free, Y and Z are domain-quantified
        inner = Exists(Z, [2, 3], P(X, Y, Z))
        expr = ForAll(Y, [0, 1], inner)

        # Expand
        expanded = expand_quantifier(expr)

        # Model (ternary predicate)
        def ternary_model(x, y, z):
            # Simple function for testing
            return torch.sigmoid(x.sum(dim=-1) + y + z)

        # Compile
        predicates = {"P": ternary_model}
        compiled = compile_logic(expanded, predicates)

        # Evaluate
        batch_size = 2
        x = torch.randn(batch_size, 4)

        result = compiled({"X": x})

        # Scalar
        assert result.shape == ()


class TestBatchReduction:
    """Tests for batch reduction behavior."""

    @pytest.mark.skip(reason="Phase 2.6+: Batch reduction not implemented.")
    def test_batch_reduction_uses_and(self):
        """Batch reduction uses AND (universal quantification)."""
        X = Variable("X")
        P = Symbol("P")

        expr = P(X)

        # Model that returns different values
        class TestModel(nn.Module):
            def forward(self, x):
                # Return values based on batch index
                return x[:, 0]  # First feature

        model = TestModel()

        # Compile
        predicates = {"P": model}
        compiled = compile_logic(expr, predicates)

        # Create batch with specific values
        batch_size = 3
        x = torch.tensor([[0.9], [0.8], [0.7]])

        result = compiled({"X": x})

        # Result should be minimum (AND semantics with R-Product)
        # With R-Product: AND = product
        expected = 0.9 * 0.8 * 0.7
        assert torch.isclose(result, torch.tensor(expected), atol=1e-5)

    @pytest.mark.skip(reason="Phase 2.6+: Batch reduction not implemented.")
    def test_empty_batch_handling(self):
        """Empty batch is handled correctly."""
        X = Variable("X")
        P = Symbol("P")

        expr = P(X)

        model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())

        predicates = {"P": model}
        compiled = compile_logic(expr, predicates)

        # Empty batch
        x = torch.randn(0, 5)

        result = compiled({"X": x})

        # Should return 1.0 (vacuously true)
        assert result.item() == 1.0


class TestGradientFlow:
    """Tests for gradient flow through batch quantification."""

    def test_gradients_flow_through_batch_quantification(self):
        """Gradients flow from loss through batch quantification."""
        X, Y = Variable("X Y")
        Digit = Symbol("Digit")

        # ForAll(Y, [0, 1], Digit(X, Y))
        expr = ForAll(Y, [0, 1], Digit(X, Y))

        # Expand
        expanded = expand_quantifier(expr)

        # Model with parameters
        model = nn.Sequential(
            nn.Linear(5, 10),
            nn.Softmax(dim=-1)
        )

        # Compile
        predicates = {"Digit": model}
        compiled = compile_logic(expanded, predicates)

        # Evaluate
        batch_size = 3
        x = torch.randn(batch_size, 5)

        # Compute loss
        loss = compiled.loss({"X": x})

        # Backward
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_gradients_with_mixed_quantification(self):
        """Gradients flow with both domain and batch quantification."""
        X, Y = Variable("X Y")
        P = Symbol("P")

        # Exists(Y, [0, 1, 2], P(X, Y))
        expr = Exists(Y, [0, 1, 2], P(X, Y))

        # Expand
        expanded = expand_quantifier(expr)

        # Model
        model = nn.Sequential(nn.Linear(4, 5), nn.Softmax(dim=-1))

        # Compile
        predicates = {"P": model}
        compiled = compile_logic(expanded, predicates)

        # Evaluate
        batch_size = 2
        x = torch.randn(batch_size, 4)

        # Loss
        loss = compiled.loss({"X": x})

        # Backward
        loss.backward()

        # Gradients should exist
        for param in model.parameters():
            assert param.grad is not None


class TestRealWorldPatterns:
    """Tests for real-world usage patterns."""

    @pytest.mark.skip(reason="Phase 2.6+: Batch reduction not implemented.")
    def test_one_hot_constraint_with_batching(self):
        """One-hot constraint works with batch quantification."""
        X, Y = Variable("X Y")
        Digit = Symbol("Digit")

        # Exists(Y, range(10), Digit(X, Y))
        # "Each X in batch is some digit"
        expr = Exists(Y, range(10), Digit(X, Y))

        # Expand
        expanded = expand_quantifier(expr)

        # MNIST-like classifier
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=-1)
        )

        # Compile
        predicates = {"Digit": model}
        compiled = compile_logic(expanded, predicates)

        # Batch of "images"
        batch_size = 4
        x = torch.randn(batch_size, 784)

        # Evaluate
        result = compiled({"X": x})

        # Should be scalar in [0, 1]
        assert result.shape == ()
        assert 0 <= result.item() <= 1

        # Should be able to compute loss
        loss = compiled.loss({"X": x})
        assert loss.shape == ()

    @pytest.mark.skip(reason="Phase 2.6+: Batch reduction not implemented.")
    def test_even_digits_constraint_with_batching(self):
        """Even digits constraint works with batching."""
        X, Y = Variable("X Y")
        Digit, Even = Symbol("Digit Even")

        # ForAll(Y, [0, 2, 4, 6, 8], Digit(X, Y) → Even(X))
        body = sp.Implies(Digit(X, Y), Even(X))
        expr = ForAll(Y, [0, 2, 4, 6, 8], body)

        # Expand
        expanded = expand_quantifier(expr)

        # Models
        digit_model = nn.Sequential(nn.Linear(10, 10), nn.Softmax(dim=-1))
        even_model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

        # Compile
        predicates = {
            "Digit": digit_model,
            "Even": even_model
        }
        compiled = compile_logic(expanded, predicates)

        # Batch
        batch_size = 3
        x = torch.randn(batch_size, 10)

        # Evaluate
        result = compiled({"X": x})

        assert result.shape == ()
        assert 0 <= result.item() <= 1


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.skip(reason="No free variables - needs support for compiled().")
    def test_no_free_variables_after_expansion(self):
        """Expression with no free variables after expansion."""
        Y = Variable("Y")
        P = Symbol("P")

        # ForAll(Y, [0, 1, 2], P(Y))
        # No free variables - Y is bound
        expr = ForAll(Y, [0, 1, 2], P(Y))

        # Expand
        expanded = expand_quantifier(expr)

        # Model (unary predicate on constants)
        def const_model(y):
            # Return fixed value based on y
            return torch.tensor(0.5 if y == 0 else 0.8)

        # Compile
        predicates = {"P": const_model}
        compiled = compile_logic(expanded, predicates)

        # Evaluate - no batch needed!
        result = compiled()

        assert result.shape == ()

    @pytest.mark.skip(reason="No free variables - needs support for compiled().")
    def test_all_variables_domain_quantified(self):
        """All variables are domain-quantified."""
        X, Y = Variable("X Y")
        P = Symbol("P")

        # ForAll(X, [0, 1], ForAll(Y, [2, 3], P(X, Y)))
        inner = ForAll(Y, [2, 3], P(X, Y))
        expr = ForAll(X, [0, 1], inner)

        # Expand
        expanded = expand_quantifier(expr)

        # Model
        def binary_model(x, y):
            return torch.tensor(0.5 if x + y < 4 else 0.7)

        # Compile
        predicates = {"P": binary_model}
        compiled = compile_logic(expanded, predicates)

        # Evaluate - no batch!
        result = compiled()

        assert result.shape == ()
