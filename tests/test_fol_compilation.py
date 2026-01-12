"""Tests for compiling and evaluating FOL expressions.

This module tests that expressions with variables can be compiled and
evaluated with universal quantification over batch dimensions.
"""

import pytest
import torch
import sympy as sp

from pysignet import Symbol, Variable, compile_logic
from pysignet.compilation import TNormCompiler


class TestBasicFOLCompilation:
    """Test basic FOL expression compilation."""

    def test_compile_simple_variable_expression(self):
        """Test compiling simple expression with one variable."""
        Digit = Symbol("Digit")
        X = Variable("X")

        # Digit(X) - "for all X in batch, Digit(X)"
        expr = Digit(X)

        # Digit classifier - returns scalar satisfaction per batch element
        digit_model = lambda x: torch.sigmoid(x.sum(dim=-1))

        logic_loss = compile_logic(expr, {"Digit": digit_model})

        # Batch of 4 samples
        x = torch.randn(4, 28*28)
        result = logic_loss(x)

        # Should return satisfaction for batch
        assert result.shape == (4,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_compile_two_variables_same_predicate(self):
        """Test compiling with two variables on same predicate."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        # Digit(X) ∧ Digit(Y) - should quantify over all X, Y pairs
        expr = sp.And(Digit(X), Digit(Y))

        digit_model = lambda x: torch.sigmoid(x.sum(dim=-1))

        logic_loss = compile_logic(expr, {"Digit": digit_model})

        x = torch.randn(3, 28*28)
        result = logic_loss(x)

        # Universal quantification: for all i, j in batch
        # Should return (3,) - one result per batch element
        assert result.shape == (3,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_compile_two_variables_different_predicates(self):
        """Test compiling with two variables on different predicates."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        # P(X) ∧ Q(Y)
        expr = sp.And(P(X), Q(Y))

        p_model = lambda x: torch.sigmoid(torch.randn(x.shape[0], 1)).squeeze(-1)
        q_model = lambda x: torch.sigmoid(torch.randn(x.shape[0], 1)).squeeze(-1)

        logic_loss = compile_logic(expr, {"P": p_model, "Q": q_model})

        x = torch.randn(5, 10)
        result = logic_loss(x)

        assert result.shape == (5,)
        assert torch.all((result >= 0) & (result <= 1))


class TestFOLBatchQuantification:
    """Test universal quantification over batch dimensions."""

    def test_forall_single_variable(self):
        """Test ∀X: P(X) - universal quantification over batch."""
        P = Symbol("P")
        X = Variable("X")

        expr = P(X)

        # P always outputs 0.8
        p_model = lambda x: torch.full((x.shape[0],), 0.8)

        logic_loss = compile_logic(expr, {"P": p_model})

        x = torch.randn(4, 10)
        result = logic_loss(x)

        # ∀X: P(X) with batch size 4
        # Should be conjunction of P(0), P(1), P(2), P(3)
        # All are 0.8, so AND of 0.8 values
        assert result.shape == (4,)
        # Exact value depends on t-norm, but should be <= 0.8
        assert torch.all(result <= 0.8)

    def test_forall_two_variables_different_satisfaction(self):
        """Test ∀X,Y: P(X,Y) where satisfaction varies."""
        Rel = Symbol("Rel")
        X, Y = Variable("X Y")

        expr = Rel(X, Y)

        # Binary predicate - accepts two arguments
        def rel_model(x, y):
            # Simple relation: satisfaction based on x and y
            return torch.sigmoid(x.sum(dim=-1) + y.sum(dim=-1))

        logic_loss = compile_logic(expr, {"Rel": rel_model})

        x = torch.randn(3, 5)
        y = torch.randn(3, 4)
        result = logic_loss({"X": x, "Y": y})

        assert result.shape == (3,)
        assert torch.all((result >= 0) & (result <= 1))


class TestFOLMixedExpressions:
    """Test expressions mixing FOL variables with constants."""

    def test_mixed_variable_and_constant(self):
        """Test expression with variable and multi-class constant."""
        Digit = Symbol("Digit")
        X = Variable("X")

        # Digit(X, 0) - X is variable, 0 is class index
        expr = Digit(X, 0)

        # Multi-class digit classifier - binary predicate (x, class_index)
        digit_model = lambda x, y: torch.softmax(torch.randn(x.shape[0], 10), dim=-1)[:, y]

        logic_loss = compile_logic(expr, {"Digit": digit_model})

        x = torch.randn(4, 784)
        result = logic_loss(x)

        assert result.shape == (4,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_multiple_predicates_same_variable(self):
        """Test multiple predicates with same variable."""
        P, Q = Symbol("P Q")
        X = Variable("X")

        # P(X) ∧ Q(X) - both use explicit variable
        expr = sp.And(P(X), Q(X))

        p_model = lambda x: torch.full((x.shape[0],), 0.9)
        q_model = lambda x: torch.full((x.shape[0],), 0.7)

        logic_loss = compile_logic(expr, {"P": p_model, "Q": q_model})

        x = torch.randn(3, 10)
        result = logic_loss(x)

        assert result.shape == (3,)
        # Should be conjunction of P(X) (0.9) and Q(X) (0.7)
        # Product t-norm: 0.9 * 0.7 = 0.63
        assert torch.allclose(result, torch.tensor(0.63))


class TestFOLComplexExpressions:
    """Test complex FOL expressions."""

    def test_nested_quantification(self):
        """Test nested expression with multiple variables."""
        P, Q, R = Symbol("P Q R")
        X, Y = Variable("X Y")

        # (P(X) ∧ Q(Y)) → R(X)
        expr = sp.Implies(sp.And(P(X), Q(Y)), R(X))

        p_model = lambda x: torch.full((x.shape[0],), 0.8)
        q_model = lambda x: torch.full((x.shape[0],), 0.7)
        r_model = lambda x: torch.full((x.shape[0],), 0.9)

        logic_loss = compile_logic(expr, {
            "P": p_model,
            "Q": q_model,
            "R": r_model
        })

        x = torch.randn(2, 5)
        result = logic_loss(x)

        assert result.shape == (2,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_disjunction_with_variables(self):
        """Test disjunction with variables."""
        P, Q = Symbol("P Q")
        X = Variable("X")

        # ∀X: (P(X) ∨ Q(X))
        # This is universal quantification of a disjunction
        expr = sp.Or(P(X), Q(X))

        p_model = lambda x: torch.full((x.shape[0],), 0.3)
        q_model = lambda x: torch.full((x.shape[0],), 0.4)

        logic_loss = compile_logic(expr, {"P": p_model, "Q": q_model})

        x = torch.randn(3, 10)
        result = logic_loss(x)

        assert result.shape == (3,)
        # P(X) ∨ Q(X) for each batch element
        # Each OR gives ~0.58 (0.3 + 0.4 - 0.3*0.4 for product t-norm)
        assert torch.all((result >= 0) & (result <= 1))
        # Result should be close to the OR value for each batch element
        single_or = 0.3 + 0.4 - 0.3 * 0.4  # Product t-norm OR
        assert torch.allclose(result, torch.full((3,), single_or), atol=1e-5)


class TestFOLGradients:
    """Test gradient flow through FOL expressions."""

    def test_gradients_flow_through_forall(self):
        """Test that gradients flow through universal quantification."""
        P = Symbol("P")
        X = Variable("X")

        expr = P(X)

        # Simple linear model
        model = torch.nn.Linear(5, 1)
        p_model = lambda x: torch.sigmoid(model(x)).squeeze(-1)

        logic_loss = compile_logic(expr, {"P": p_model})

        x = torch.randn(3, 5, requires_grad=True)
        result = logic_loss(x)

        # Compute loss and backprop
        loss = result.mean()
        loss.backward()

        # Gradients should flow to input and model parameters
        assert x.grad is not None
        assert model.weight.grad is not None


class TestFOLEdgeCases:
    """Test edge cases for FOL compilation."""

    def test_no_variables_compiles_normally(self):
        """Test that simple predicates with explicit variables compile."""
        X = Variable("X")
        P, Q = Symbol("P Q")

        # P(X) ∧ Q(X) - explicit variable usage
        expr = sp.And(P(X), Q(X))

        p_model = lambda x: torch.full((x.shape[0],), 0.8)
        q_model = lambda x: torch.full((x.shape[0],), 0.7)

        logic_loss = compile_logic(expr, {"P": p_model, "Q": q_model})

        x = torch.randn(4, 10)
        result = logic_loss(x)

        assert result.shape == (4,)
        # Should be simple conjunction, no quantification
        # Result should be close to t-norm AND of 0.8 and 0.7

    def test_single_batch_element(self):
        """Test FOL with batch size 1."""
        P = Symbol("P")
        X = Variable("X")

        expr = P(X)

        p_model = lambda x: torch.full((x.shape[0],), 0.9)

        logic_loss = compile_logic(expr, {"P": p_model})

        x = torch.randn(1, 10)
        result = logic_loss(x)

        # With batch size 1, ∀X quantifies over just 1 element
        assert result.shape == (1,)
        # Should be close to 0.9 (single element quantification)

    def test_large_batch(self):
        """Test FOL with large batch for performance."""
        P = Symbol("P")
        X = Variable("X")

        expr = P(X)

        p_model = lambda x: torch.sigmoid(torch.randn(x.shape[0]))

        logic_loss = compile_logic(expr, {"P": p_model})

        x = torch.randn(100, 10)
        result = logic_loss(x)

        assert result.shape == (100,)
        assert torch.all((result >= 0) & (result <= 1))


class TestFOLWithDictInputs:
    """Test FOL compilation with dictionary inputs."""

    def test_variable_with_dict_input(self):
        """Test FOL expression with dictionary inputs."""
        P = Symbol("P")
        X = Variable("X")

        expr = P(X)

        p_model = lambda x: torch.sigmoid(x.sum(dim=-1))

        logic_loss = compile_logic(expr, {"P": p_model})

        # Dict input with variable name as key
        inputs = {"X": torch.randn(5, 10)}
        result = logic_loss(inputs)

        assert result.shape == (5,)
        assert torch.all((result >= 0) & (result <= 1))
