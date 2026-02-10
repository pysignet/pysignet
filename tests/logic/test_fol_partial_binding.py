"""Tests for partial binding of variables in FOL expressions.

This module tests the ability to incrementally bind variables in logical
expressions, allowing for efficient computation when some inputs are known
before others.
"""

import pytest
import torch
import torch.nn as nn
import sympy as sp

from pysignet import Symbol, compile_logic, logic_to_loss
from pysignet.logic import Variable


class TestBasicPartialBinding:
    """Test basic partial binding functionality."""

    def test_partial_bind_single_variable(self):
        """Test binding a single variable partially."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        # Expression with two variables
        expr = sp.And(P(X), Q(Y))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.sum(dim=-1))
        }

        compiled = compile_logic(expr, predicates)

        # Partial bind X
        batch_size = 4
        x = torch.randn(batch_size, 10)
        partial = compiled.partial(X=x)

        # Now bind Y with same batch size
        y = torch.randn(batch_size, 10)
        # CompiledExpression returns per-batch results by default
        result = partial(Y=y)

        assert result.shape == torch.Size([batch_size])
        assert torch.all((result >= 0) & (result <= 1))

    def test_partial_bind_returns_callable(self):
        """Test that partial returns a callable."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        expr = sp.And(P(X), Q(Y))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.sum(dim=-1))
        }

        compiled = compile_logic(expr, predicates)
        x = torch.randn(1, 10)
        partial = compiled.partial(X=x)

        assert callable(partial)

    def test_partial_bind_order_independent(self):
        """Test that binding order doesn't affect result."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        expr = sp.And(P(X), Q(Y))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.sum(dim=-1))
        }

        compiled = compile_logic(expr, predicates)

        x = torch.randn(1, 10)
        y = torch.randn(1, 10)

        # Bind X then Y
        result1 = compiled.partial(X=x)(Y=y)

        # Bind Y then X
        result2 = compiled.partial(Y=y)(X=x)

        assert torch.allclose(result1, result2)

    def test_partial_same_as_full_binding(self):
        """Test that partial binding gives same result as full binding."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        expr = sp.And(P(X), Q(Y))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.sum(dim=-1))
        }

        compiled = compile_logic(expr, predicates)

        x = torch.randn(1, 10)
        y = torch.randn(1, 10)

        # Full binding
        result_full = compiled(X=x, Y=y)

        # Partial binding
        result_partial = compiled.partial(X=x)(Y=y)

        assert torch.allclose(result_full, result_partial)


class TestMultiplePartialBindings:
    """Test multiple sequential partial bindings."""

    def test_chain_multiple_partial_bindings(self):
        """Test chaining multiple partial bindings."""
        P, Q, R = Symbol("P Q R")
        X, Y, Z = Variable("X Y Z")

        # Expression with three variables
        expr = sp.And(P(X), sp.And(Q(Y), R(Z)))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.sum(dim=-1)),
            "R": lambda z: torch.sigmoid(z.sum(dim=-1))
        }

        compiled = compile_logic(expr, predicates)

        # Chain partial bindings
        x = torch.randn(1, 10)
        y = torch.randn(1, 10)
        z = torch.randn(1, 10)

        # CompiledExpression returns per-batch results by default
        result = compiled.partial(X=x).partial(Y=y)(Z=z)

        assert result.shape == torch.Size([1])
        assert torch.all((result >= 0) & (result <= 1))

    def test_bind_multiple_at_once(self):
        """Test binding multiple variables at once in partial."""
        P, Q, R = Symbol("P Q R")
        X, Y, Z = Variable("X Y Z")

        expr = sp.And(P(X), sp.And(Q(Y), R(Z)))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.sum(dim=-1)),
            "R": lambda z: torch.sigmoid(z.sum(dim=-1))
        }

        compiled = compile_logic(expr, predicates)

        x = torch.randn(1, 10)
        y = torch.randn(1, 10)
        z = torch.randn(1, 10)

        # Bind two variables at once
        result = compiled.partial(X=x, Y=y)(Z=z)

        # Should be same as binding all three
        result_full = compiled(X=x, Y=y, Z=z)

        assert torch.allclose(result, result_full)

    def test_bind_in_different_orders(self):
        """Test that different binding orders give same result."""
        P, Q, R = Symbol("P Q R")
        X, Y, Z = Variable("X Y Z")

        expr = sp.And(P(X), sp.And(Q(Y), R(Z)))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.sum(dim=-1)),
            "R": lambda z: torch.sigmoid(z.sum(dim=-1))
        }

        compiled = compile_logic(expr, predicates)

        x = torch.randn(1, 10)
        y = torch.randn(1, 10)
        z = torch.randn(1, 10)

        # Different orders
        result1 = compiled.partial(X=x).partial(Y=y)(Z=z)
        result2 = compiled.partial(Y=y).partial(X=x)(Z=z)
        result3 = compiled.partial(Z=z).partial(X=x)(Y=y)

        assert torch.allclose(result1, result2)
        assert torch.allclose(result2, result3)


class TestErrorHandling:
    """Test error handling in partial binding."""

    def test_error_on_duplicate_binding(self):
        """Test that binding the same variable twice raises error."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        expr = sp.And(P(X), Q(Y))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.sum(dim=-1))
        }

        compiled = compile_logic(expr, predicates)

        x1 = torch.randn(1, 10)
        x2 = torch.randn(1, 10)

        # Bind X once
        partial = compiled.partial(X=x1)

        # Try to bind X again should raise error
        with pytest.raises(ValueError, match="already bound"):
            partial.partial(X=x2)

    def test_error_on_unbinding_nonexistent_variable(self):
        """Test that binding non-existent variable raises error at partial time."""
        P = Symbol("P")
        X = Variable("X")

        expr = P(X)

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1))
        }

        compiled = compile_logic(expr, predicates)

        # Try to bind a variable that doesn't exist
        # NEW BEHAVIOR: CompiledExpression validates at partial() time
        with pytest.raises(ValueError, match="not in expression"):
            compiled.partial(Z=torch.randn(1, 10))

    def test_error_on_missing_final_bindings(self):
        """Test that calling with missing variables raises error."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        expr = sp.And(P(X), Q(Y))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.sum(dim=-1))
        }

        compiled = compile_logic(expr, predicates)

        # Partial bind X only
        partial = compiled.partial(X=torch.randn(1, 10))

        # Try to call without Y should raise error
        with pytest.raises(ValueError, match="Missing input"):
            partial()


class TestPartialBindingWithDifferentExpressions:
    """Test partial binding with various expression types."""

    def test_partial_with_implication(self):
        """Test partial binding with implication."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        expr = sp.Implies(P(X), Q(Y))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.sum(dim=-1))
        }

        compiled = compile_logic(expr, predicates)

        x = torch.randn(1, 10)
        y = torch.randn(1, 10)

        result_partial = compiled.partial(X=x)(Y=y)
        result_full = compiled(X=x, Y=y)

        assert torch.allclose(result_partial, result_full)

    def test_partial_with_negation(self):
        """Test partial binding with negation."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        expr = sp.And(sp.Not(P(X)), Q(Y))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.sum(dim=-1))
        }

        compiled = compile_logic(expr, predicates)

        x = torch.randn(1, 10)
        y = torch.randn(1, 10)

        result_partial = compiled.partial(X=x)(Y=y)
        result_full = compiled(X=x, Y=y)

        assert torch.allclose(result_partial, result_full)

    def test_partial_with_constants(self):
        """Test partial binding with constants in predicates."""
        Digit = Symbol("Digit")
        X = Variable("X")

        expr = Digit(X, 5)

        model = nn.Sequential(nn.Linear(10, 10), nn.Softmax(dim=-1))

        predicates = {
            "Digit": lambda x, label: model(x)[:, label]
        }

        compiled = compile_logic(expr, predicates)

        x = torch.randn(1, 10)

        # Partial bind X (constant 5 is already in the expression)
        partial = compiled.partial(X=x)

        # Call with no more arguments needed
        # CompiledExpression returns per-batch results by default
        result = partial()

        assert result.shape == torch.Size([1])


class TestPartialBindingLoss:
    """Test loss computation with partial binding."""

    def test_loss_after_partial_binding(self):
        """Test computing loss after partial binding."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        expr = sp.And(P(X), Q(Y))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.sum(dim=-1))
        }

        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(1, 10)
        y = torch.randn(1, 10)

        # Partial bind and compute loss
        partial = logic_loss.partial(X=x)
        loss = partial.loss(Y=y)

        # Compare to full binding
        loss_full = logic_loss.loss(X=x, Y=y)

        assert torch.allclose(loss, loss_full)

    def test_gradient_flow_through_partial_binding(self):
        """Test that gradients flow correctly through partial binding."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        expr = sp.And(P(X), Q(Y))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.sum(dim=-1))
        }

        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(1, 10, requires_grad=True)
        y = torch.randn(1, 10, requires_grad=True)

        # Partial bind and compute loss
        partial = logic_loss.partial(X=x)
        loss = partial.loss(Y=y)

        # Backward pass
        loss.backward()

        # Check gradients exist for both x and y
        assert x.grad is not None
        assert y.grad is not None
        assert not torch.all(x.grad == 0)
        assert not torch.all(y.grad == 0)


class TestRealWorldUseCases:
    """Test real-world use cases for partial binding."""

    def test_partial_binding_for_search(self):
        """Test using partial binding for searching over one variable."""
        Similar = Symbol("Similar")
        X, Y = Variable("X Y")

        # Constraint: Similar(X, Y) - images X and Y should be similar
        expr = Similar(X, Y)

        def similarity(x, y):
            # Cosine similarity rescaled to [0, 1]
            x_norm = x / x.norm(dim=-1, keepdim=True)
            y_norm = y / y.norm(dim=-1, keepdim=True)
            cos_sim = (x_norm * y_norm).sum(dim=-1)
            return (cos_sim + 1.0) / 2.0  # Map [-1, 1] to [0, 1]

        predicates = {"Similar": similarity}
        compiled = compile_logic(expr, predicates)

        # Fix query image X
        query = torch.randn(1, 100)
        partial = compiled.partial(X=query)

        # Search over different candidate images Y
        candidates = [
            torch.randn(1, 100),
            torch.randn(1, 100),
            torch.randn(1, 100)
        ]

        similarities = []
        for candidate in candidates:
            sim = partial(Y=candidate)
            similarities.append(sim.item())

        # All similarities should be computed
        assert len(similarities) == 3
        assert all(0 <= s <= 1 for s in similarities)

    def test_partial_binding_for_constraint_checking(self):
        """Test using partial binding for checking constraints on fixed input."""
        Digit, Even, Odd = Symbol("Digit Even Odd")
        X = Variable("X")

        # Constraint: Digit(X, 5) → Odd(X)
        expr = sp.Implies(Digit(X, 5), Odd(X))

        digit_model = nn.Sequential(nn.Linear(784, 10), nn.Softmax(dim=-1))
        odd_model = nn.Sequential(nn.Linear(784, 1), nn.Sigmoid())

        predicates = {
            "Digit": lambda x, label: digit_model(x)[:, label],
            "Odd": lambda x: odd_model(x).squeeze(-1)
        }

        compiled = compile_logic(expr, predicates)

        # Fix input image
        image = torch.randn(1, 784)
        partial = compiled.partial(X=image)

        # Check constraint multiple times (e.g., during training)
        for _ in range(5):
            # Use quantify='none' to get per-batch results
            satisfaction = partial(quantify='none')
            assert satisfaction.shape == torch.Size([1])
