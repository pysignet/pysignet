"""Tests for Symbol() API - binary and multi-class predicates.

This demonstrates that both binary (nullary) and multi-class (unary) predicates
use the same Symbol() declaration.
"""

import pytest
import torch
import torch.nn as nn
import sympy as sp

from pysignet import Symbol, Variable, logic_to_loss


class TestSymbolAPI:
    """Test the Symbol() API for binary and multi-class predicates."""

    def test_create_symbols(self):
        """Test creating both binary and multi-class predicates with Symbol()."""
        # Same declaration for all predicate types
        P, Q, Digit = Symbol("P Q Digit")

        # All are PredicateSymbol (subclass of sp.Symbol)
        assert isinstance(P, sp.Symbol)
        assert isinstance(Q, sp.Symbol)
        assert isinstance(Digit, sp.Symbol)

        # Names are correct
        assert str(P) == "P"
        assert str(Q) == "Q"
        assert str(Digit) == "Digit"

    def test_nullary_usage_binary_predicates(self):
        """Test using symbols as nullary."""
        X = Variable("X")
        P, Q = Symbol("P Q")

        # Used directly without arguments (nullary)
        expr = sp.And(P(X), Q(X))

        assert isinstance(expr, sp.And)
        assert len(expr.args) == 2

    def test_unary_usage_multiclass_predicates(self):
        """Test using symbols with FOL variables and constants."""
        Digit = Symbol("Digit")
        X = Variable("X")

        # FOL interface: Digit(X, 0), Digit(X, 1), Digit(X, 2)
        # X is variable, 0/1/2 are output channel constants
        expr = sp.Or(Digit(X, 0), Digit(X, 1), Digit(X, 2))

        assert isinstance(expr, sp.Or)
        assert len(expr.args) == 3

    def test_mixed_nullary_and_unary(self):
        """Test mixing binary and multi-class predicates in same expression."""
        P, Q, Digit = Symbol("P Q Digit")
        X = Variable("X")

        # P and Q are nullary, Digit uses FOL interface
        expr = sp.And(P(X), sp.Or(Q(X), Digit(X, 0)))

        # Should compile without error
        binary_model_p = lambda x: torch.sigmoid(x.mean(dim=-1))
        binary_model_q = lambda x: torch.sigmoid(x.sum(dim=-1))
        multiclass = nn.Sequential(nn.Linear(10, 3), nn.Softmax(dim=-1))

        predicates = {"P": binary_model_p, "Q": binary_model_q, "Digit": multiclass}

        compiled = logic_to_loss(expr, predicates)

        # Should evaluate successfully
        x = torch.randn(3, 10)
        result = compiled(X=x, quantify="none")

        assert result.shape == (3,)

    def test_validation_rejects_inconsistent_usage(self):
        """Test that validation rejects predicate used both ways."""
        X = Variable("X")
        P, Digit = Symbol("P Digit")

        # P used both as nullary AND unary - INVALID!
        # P(X) has 1 free var, P(0) has 0 free vars - inconsistent!
        expr = sp.And(P(X), P(X, 0))

        predicates = {"P": lambda x: torch.sigmoid(x)}

        # Should raise ValueError at compile time
        with pytest.raises(ValueError, match="Predicate 'P' used inconsistently"):
            logic_to_loss(expr, predicates)

    def test_validation_rejects_multiclass_without_args(self):
        """Test validation when predicate used with different arities."""
        Digit = Symbol("Digit")
        X = Variable("X")

        # Digit used WITH variable (arity 1) and WITHOUT args (arity 0) - INVALID!
        expr = sp.And(Digit(X, 0), Digit(X))

        predicates = {"Digit": nn.Linear(10, 5)}

        # Should raise ValueError
        with pytest.raises(ValueError, match="used inconsistently"):
            logic_to_loss(expr, predicates)

    def test_validation_tracks_arity(self):
        """Test that validation tracks specific arity (not just nullary vs unary)."""
        Rel = Symbol("Rel")

        # Rel used with arity 1 and arity 2 - INVALID!
        # This tests that we track actual arity, not just "has arguments"
        expr = sp.And(Rel(0), Rel(0, 1))

        predicates = {"Rel": nn.Linear(10, 3)}

        # Should raise ValueError about inconsistent arity
        with pytest.raises(ValueError, match="used inconsistently"):
            logic_to_loss(expr, predicates)


class TestSymbolAPIUsagePatterns:
    """Test realistic usage patterns with Symbol API."""

    def test_mnist_10class_example(self):
        """Test 10-class MNIST example with FOL interface."""
        Digit = Symbol("Digit")
        X = Variable("X")

        # Constraint using FOL interface: Digit(X, 0) OR ... OR Digit(X, 9)
        expr = sp.Or(
            Digit(X, 0),
            Digit(X, 1),
            Digit(X, 2),
            Digit(X, 3),
            Digit(X, 4),
            Digit(X, 5),
            Digit(X, 6),
            Digit(X, 7),
            Digit(X, 8),
            Digit(X, 9),
        )

        # Map to classifier
        digit_classifier = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10), nn.Softmax(dim=-1)
        )

        predicates = {"Digit": digit_classifier}
        compiled = logic_to_loss(expr, predicates)

        # Evaluate
        x = torch.randn(3, 784)
        result = compiled(X=x, quantify="forall")

        assert result.shape == ()

    def test_mixed_binary_and_multiclass(self):
        """Test realistic mixing of binary and multi-class predicates."""
        X = Variable("X")
        IsAdult, HasLicense, VehicleType = Symbol("IsAdult HasLicense VehicleType")

        # IsAdult and HasLicense are binary (nullary)
        # VehicleType is multi-class: 0=car, 1=motorcycle, 2=truck

        # Rule: Can drive car (type 0) if adult AND has license
        expr = sp.Implies(VehicleType(X, 0), sp.And(IsAdult(X), HasLicense(X)))

        # Map to models - same dict structure!
        predicates = {
            "IsAdult": lambda x: (x[:, 0] > 18).float(),
            "HasLicense": lambda x: (x[:, 1] > 0.5).float(),
            "VehicleType": nn.Sequential(nn.Linear(10, 3), nn.Softmax(dim=-1)),
        }

        compiled = logic_to_loss(expr, predicates)

        x = torch.randn(10, 10)
        result = compiled(X=x)

        assert result.shape == ()

    def test_all_binary_predicates(self):
        """Test that all-binary case still works perfectly."""
        X = Variable("X")
        P, Q, R = Symbol("P Q R")

        expr = sp.And(P(X), sp.Or(Q(X), sp.Not(R(X))))

        predicates = {
            "P": lambda x: torch.sigmoid(x[:, 0]),
            "Q": lambda x: torch.sigmoid(x[:, 1]),
            "R": lambda x: torch.sigmoid(x[:, 2]),
        }

        compiled = logic_to_loss(expr, predicates)

        x = torch.randn(10, 10)
        result = compiled(X=x)

        assert result.shape == ()

    def test_all_multiclass_predicates(self):
        """Test all-multiclass case."""
        X = Variable("X")
        Color, Shape = Symbol("Color Shape")

        # Color: 0=red, 1=green, 2=blue
        # Shape: 0=circle, 1=square, 2=triangle

        # Red circles or green squares
        expr = sp.Or(sp.And(Color(X, 0), Shape(X, 0)), sp.And(Color(X, 1), Shape(X, 1)))

        predicates = {
            "Color": nn.Sequential(nn.Linear(10, 3), nn.Softmax(dim=-1)),
            "Shape": nn.Sequential(nn.Linear(10, 3), nn.Softmax(dim=-1)),
        }

        compiled = logic_to_loss(expr, predicates)

        x = torch.randn(10, 10)
        result = compiled(X=x)

        assert result.shape == ()
