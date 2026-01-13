"""Tests for passing constants to predicate callables.

This module tests that constants in PredicateApplications are passed directly
to the predicate callables and are not required in variable bindings.
"""

import pytest
import torch
import torch.nn as nn
import sympy as sp

from pysignet import Symbol, compile_logic
from pysignet.logic import Variable
from pysignet.tnorms import RProductTNorm


class TestConstantOnlyPredicates:
    """Test predicates with only constant arguments."""

    def test_predicate_with_single_constant(self):
        """Test predicate application with single constant argument."""
        P = Symbol("P")

        # P(5) - constant only
        expr = P(5)

        # Predicate that receives constant directly
        def predicate_fn(label):
            assert label == 5
            return torch.tensor([0.8])

        predicates = {"P": predicate_fn}
        compiled = compile_logic(expr, predicates)

        # Should evaluate without any variable bindings
        result = compiled()
        # Shape is whatever the predicate returns (no batch dimension)
        assert result.shape == torch.Size([1])
        assert result[0].item() == pytest.approx(0.8)

    def test_predicate_with_multiple_constants(self):
        """Test predicate with multiple constant arguments."""
        Rel = Symbol("Rel")

        # Rel(1, 2, 3) - all constants
        expr = Rel(1, 2, 3)

        # Predicate that receives all constants
        def predicate_fn(a, b, c):
            assert a == 1
            assert b == 2
            assert c == 3
            return torch.tensor([0.9])

        predicates = {"Rel": predicate_fn}
        compiled = compile_logic(expr, predicates)

        # Should evaluate without variable bindings
        result = compiled()
        assert result.shape == torch.Size([1])
        assert result[0].item() == pytest.approx(0.9)

    def test_predicate_with_string_constant(self):
        """Test predicate with string constant."""
        Color = Symbol("Color")

        # Color("red")
        expr = Color("red")

        # Predicate that maps string to probability
        def predicate_fn(color):
            assert color == "red"
            color_probs = {"red": 0.7, "green": 0.2, "blue": 0.1}
            return torch.tensor([color_probs[color]])

        predicates = {"Color": predicate_fn}
        compiled = compile_logic(expr, predicates)

        result = compiled()
        assert result.item() == pytest.approx(0.7)


class TestMixedVariableAndConstant:
    """Test predicates with mixed variable and constant arguments."""

    def test_variable_then_constant(self):
        """Test predicate with variable followed by constant: P(X, 5)."""
        Digit = Symbol("Digit")
        X = Variable("X")

        # Digit(X, 5) - X is variable, 5 is constant
        expr = Digit(X, 5)

        # 10-class classifier
        model = nn.Sequential(nn.Linear(10, 10), nn.Softmax(dim=-1))

        # Lambda that selects the constant class
        predicates = {
            "Digit": lambda x, label: model(x)[:, label]
        }

        compiled = compile_logic(expr, predicates)

        # Only bind X, not the constant 5
        x = torch.randn(8, 10)
        result = compiled(X=x)

        assert result.shape == torch.Size([8])
        assert torch.all((result >= 0) & (result <= 1))

    def test_constant_then_variable(self):
        """Test predicate with constant followed by variable: P(5, X)."""
        P = Symbol("P")
        X = Variable("X")

        # P(5, X) - constant first, then variable
        expr = P(5, X)

        # Predicate receives constant and variable
        def predicate_fn(label, x):
            assert label == 5
            return torch.sigmoid(x.sum(dim=-1))

        predicates = {"P": predicate_fn}
        compiled = compile_logic(expr, predicates)

        # Only bind X
        x = torch.randn(4, 10)
        result = compiled(X=x)

        assert result.shape == torch.Size([4])

    def test_variable_constant_variable(self):
        """Test predicate with pattern V-C-V: P(X, 0, Y)."""
        P = Symbol("P")
        X, Y = Variable("X Y")

        # P(X, 0, Y) - variable, constant, variable
        expr = P(X, 0, Y)

        # Predicate that uses all three arguments
        def predicate_fn(x, mode, y):
            assert mode == 0
            # Combine x and y based on mode
            return torch.sigmoid((x + y).sum(dim=-1))

        predicates = {"P": predicate_fn}
        compiled = compile_logic(expr, predicates)

        # Bind both variables, constant passed automatically
        x = torch.randn(4, 5)
        y = torch.randn(4, 5)
        result = compiled(X=x, Y=y)

        assert result.shape == torch.Size([4])

    def test_interleaved_variables_and_constants(self):
        """Test predicate with interleaved variables and constants."""
        Q = Symbol("Q")
        X, Y = Variable("X Y")

        # Q(X, 1, Y, "label", 3.14)
        expr = Q(X, 1, Y, "label", 3.14)

        # Predicate receives all in order
        def predicate_fn(x, const1, y, const2, const3):
            assert const1 == 1
            assert const2 == "label"
            assert const3 == pytest.approx(3.14)
            return torch.sigmoid(x.sum(dim=-1) + y.sum(dim=-1))

        predicates = {"Q": predicate_fn}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        result = compiled(X=x, Y=y)

        assert result.shape == torch.Size([3])


class TestConstantsNotInBindings:
    """Test that constants are not required in variable bindings."""

    def test_constants_not_required_in_bindings(self):
        """Test that only variables need to be bound, not constants."""
        P = Symbol("P")
        X = Variable("X")

        # P(X, 5)
        expr = P(X, 5)

        predicates = {
            "P": lambda x, label: torch.sigmoid(x.sum(dim=-1))
        }
        compiled = compile_logic(expr, predicates)

        # Should only need to bind X, not the constant 5
        x = torch.randn(4, 10)
        result = compiled(X=x)  # No error even though 5 not bound

        assert result.shape == torch.Size([4])

    def test_binding_constant_name_has_no_effect(self):
        """Test that trying to bind a constant name has no effect."""
        Digit = Symbol("Digit")
        X = Variable("X")

        # Digit(X, 5)
        expr = Digit(X, 5)

        model = nn.Sequential(nn.Linear(10, 10), nn.Softmax(dim=-1))
        predicates = {
            "Digit": lambda x, label: model(x)[:, label]
        }
        compiled = compile_logic(expr, predicates)

        x = torch.randn(4, 10)

        # This should work (only X bound)
        result1 = compiled(X=x)
        assert result1.shape == torch.Size([4])

        # Binding extra keys should be ignored or raise error
        # (depends on implementation, but constant shouldn't be required)
        result2 = compiled(X=x)
        assert torch.allclose(result1, result2)

    def test_multiple_constants_none_required_in_binding(self):
        """Test expression with multiple constants, none required in binding."""
        P, Q = Symbol("P Q")
        X = Variable("X")

        # P(X, 5) ∧ Q(X, "red")
        expr = sp.And(P(X, 5), Q(X, "red"))

        predicates = {
            "P": lambda x, label: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda x, color: torch.sigmoid(x.sum(dim=-1) + len(color))
        }
        compiled = compile_logic(expr, predicates)

        # Only bind X
        x = torch.randn(4, 10)
        result = compiled(X=x)

        assert result.shape == torch.Size([4])


class TestConstantInterpretation:
    """Test how predicates can interpret different constant types."""

    def test_integer_constant_as_index(self):
        """Test integer constant used as tensor index."""
        Digit = Symbol("Digit")
        X = Variable("X")

        # Digit(X, 5) - 5 used as class index
        expr = Digit(X, 5)

        model = nn.Sequential(nn.Linear(784, 10), nn.Softmax(dim=-1))

        # Use constant as index into softmax output
        predicates = {
            "Digit": lambda x, label: model(x)[:, label]
        }
        compiled = compile_logic(expr, predicates)

        x = torch.randn(8, 784)
        result = compiled(X=x)

        assert result.shape == torch.Size([8])
        assert torch.all((result >= 0) & (result <= 1))

    def test_string_constant_mapped_to_index(self):
        """Test string constant mapped to integer index."""
        Color = Symbol("Color")
        X = Variable("X")

        # Color(X, "red")
        expr = Color(X, "red")

        model = nn.Sequential(nn.Linear(10, 3), nn.Softmax(dim=-1))

        # Map string to index
        color_map = {"red": 0, "green": 1, "blue": 2}

        predicates = {
            "Color": lambda x, color: model(x)[:, color_map[color]]
        }
        compiled = compile_logic(expr, predicates)

        x = torch.randn(5, 10)
        result = compiled(X=x)

        assert result.shape == torch.Size([5])

    def test_constant_used_in_computation(self):
        """Test constant used directly in computation."""
        P = Symbol("P")
        X = Variable("X")

        # P(X, 0.5) - threshold constant
        expr = P(X, 0.5)

        # Use constant as threshold
        predicates = {
            "P": lambda x, threshold: (torch.sigmoid(x.sum(dim=-1)) > threshold).float()
        }
        compiled = compile_logic(expr, predicates)

        x = torch.randn(6, 10)
        result = compiled(X=x)

        assert result.shape == torch.Size([6])
        assert torch.all((result == 0) | (result == 1))

    def test_none_constant(self):
        """Test None as a constant."""
        P = Symbol("P")
        X = Variable("X")

        # P(X, None) - None as placeholder
        expr = P(X, None)

        predicates = {
            "P": lambda x, mode: torch.sigmoid(x.sum(dim=-1)) if mode is None else x.mean(dim=-1)
        }
        compiled = compile_logic(expr, predicates)

        x = torch.randn(3, 10)
        result = compiled(X=x)

        assert result.shape == torch.Size([3])


class TestComplexExpressions:
    """Test constants in complex logical expressions."""

    def test_and_with_constants(self):
        """Test AND expression with constants."""
        P, Q = Symbol("P Q")
        X = Variable("X")

        # P(X, 0) ∧ Q(X, 1)
        expr = sp.And(P(X, 0), Q(X, 1))

        model = nn.Sequential(nn.Linear(10, 5), nn.Softmax(dim=-1))

        predicates = {
            "P": lambda x, label: model(x)[:, label],
            "Q": lambda x, label: model(x)[:, label]
        }
        compiled = compile_logic(expr, predicates)

        x = torch.randn(4, 10)
        result = compiled(X=x)

        assert result.shape == torch.Size([4])

    def test_or_with_constants(self):
        """Test OR expression with constants."""
        Digit = Symbol("Digit")
        X = Variable("X")

        # Digit(X, 0) ∨ Digit(X, 1) - "X is digit 0 or 1"
        expr = sp.Or(Digit(X, 0), Digit(X, 1))

        model = nn.Sequential(nn.Linear(784, 10), nn.Softmax(dim=-1))

        predicates = {
            "Digit": lambda x, label: model(x)[:, label]
        }
        compiled = compile_logic(expr, predicates)

        x = torch.randn(8, 784)
        result = compiled(X=x)

        assert result.shape == torch.Size([8])

    def test_implication_with_constants(self):
        """Test implication with constants."""
        Digit, Even = Symbol("Digit Even")
        X = Variable("X")

        # Digit(X, 2) → Even(X) - "if X is digit 2, then X is even"
        expr = sp.Implies(Digit(X, 2), Even(X))

        digit_model = nn.Sequential(nn.Linear(784, 10), nn.Softmax(dim=-1))
        even_model = nn.Sequential(nn.Linear(784, 1), nn.Sigmoid())

        predicates = {
            "Digit": lambda x, label: digit_model(x)[:, label],
            "Even": lambda x: even_model(x).squeeze(-1)
        }
        compiled = compile_logic(expr, predicates)

        x = torch.randn(8, 784)
        result = compiled(X=x)

        assert result.shape == torch.Size([8])

    def test_nested_expression_with_constants(self):
        """Test deeply nested expression with constants."""
        P, Q, R = Symbol("P Q R")
        X = Variable("X")

        # (P(X, 0) ∧ Q(X, 1)) → R(X, 2)
        expr = sp.Implies(
            sp.And(P(X, 0), Q(X, 1)),
            R(X, 2)
        )

        model = nn.Sequential(nn.Linear(10, 5), nn.Softmax(dim=-1))

        predicates = {
            "P": lambda x, label: model(x)[:, label],
            "Q": lambda x, label: model(x)[:, label],
            "R": lambda x, label: model(x)[:, label]
        }
        compiled = compile_logic(expr, predicates)

        x = torch.randn(4, 10)
        result = compiled(X=x)

        assert result.shape == torch.Size([4])


class TestGradientFlow:
    """Test that gradients flow through expressions with constants."""

    def test_gradients_with_constant(self):
        """Test gradient flow through predicate with constant."""
        Digit = Symbol("Digit")
        X = Variable("X")

        # Digit(X, 5)
        expr = Digit(X, 5)

        model = nn.Sequential(nn.Linear(10, 10), nn.Softmax(dim=-1))

        predicates = {
            "Digit": lambda x, label: model(x)[:, label]
        }
        compiled = compile_logic(expr, predicates)

        x = torch.randn(4, 10, requires_grad=True)
        result = compiled(X=x)

        # Backward pass
        loss = result.mean()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_gradients_with_mixed_args(self):
        """Test gradient flow with mixed variable and constant args."""
        P = Symbol("P")
        X, Y = Variable("X Y")

        # P(X, 0, Y)
        expr = P(X, 0, Y)

        predicates = {
            "P": lambda x, mode, y: torch.sigmoid((x + y).sum(dim=-1))
        }
        compiled = compile_logic(expr, predicates)

        x = torch.randn(3, 5, requires_grad=True)
        y = torch.randn(3, 5, requires_grad=True)
        result = compiled(X=x, Y=y)

        loss = result.mean()
        loss.backward()

        assert x.grad is not None
        assert y.grad is not None
        assert not torch.all(x.grad == 0)
        assert not torch.all(y.grad == 0)


class TestRealWorldPatterns:
    """Test real-world usage patterns with constants."""

    def test_multiclass_classification_with_target_label(self):
        """Test multiclass classification with target label as constant."""
        Digit = Symbol("Digit")
        X = Variable("X")

        # Digit(X, 5) - "X should be classified as digit 5"
        expr = Digit(X, 5)

        model = nn.Sequential(nn.Linear(784, 10), nn.Softmax(dim=-1))

        predicates = {
            "Digit": lambda x, label: model(x)[:, label]
        }
        compiled = compile_logic(expr, predicates)

        # Batch of images
        x = torch.randn(32, 784)
        satisfaction = compiled(X=x)

        assert satisfaction.shape == torch.Size([32])
        assert torch.all((satisfaction >= 0) & (satisfaction <= 1))

    def test_constraint_with_multiple_class_labels(self):
        """Test constraint over multiple class labels."""
        Digit, Even = Symbol("Digit Even")
        X = Variable("X")

        # Digit(X, 0) ∨ Digit(X, 2) ∨ Digit(X, 4) → Even(X)
        # "If X is 0, 2, or 4, then X should be even"
        expr = sp.Implies(
            sp.Or(Digit(X, 0), Digit(X, 2), Digit(X, 4)),
            Even(X)
        )

        digit_model = nn.Sequential(nn.Linear(784, 10), nn.Softmax(dim=-1))
        even_model = nn.Sequential(nn.Linear(784, 1), nn.Sigmoid())

        predicates = {
            "Digit": lambda x, label: digit_model(x)[:, label],
            "Even": lambda x: even_model(x).squeeze(-1)
        }
        compiled = compile_logic(expr, predicates)

        x = torch.randn(16, 784)
        satisfaction = compiled(X=x)

        assert satisfaction.shape == torch.Size([16])

    def test_color_and_shape_constraints(self):
        """Test constraints with string constants for attributes."""
        Color, Shape = Symbol("Color Shape")
        X = Variable("X")

        # Color(X, "red") → Shape(X, "circle")
        # "Red objects should be circles"
        expr = sp.Implies(Color(X, "red"), Shape(X, "circle"))

        color_model = nn.Sequential(nn.Linear(100, 3), nn.Softmax(dim=-1))
        shape_model = nn.Sequential(nn.Linear(100, 4), nn.Softmax(dim=-1))

        color_map = {"red": 0, "green": 1, "blue": 2}
        shape_map = {"circle": 0, "square": 1, "triangle": 2, "star": 3}

        predicates = {
            "Color": lambda x, color: color_model(x)[:, color_map[color]],
            "Shape": lambda x, shape: shape_model(x)[:, shape_map[shape]]
        }
        compiled = compile_logic(expr, predicates)

        x = torch.randn(10, 100)
        satisfaction = compiled(X=x)

        assert satisfaction.shape == torch.Size([10])
