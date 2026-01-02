"""Tests for Symbol() API - binary and multi-class predicates.

This demonstrates that both binary (nullary) and multi-class (unary) predicates
use the same Symbol() declaration.
"""

import pytest
import torch
import torch.nn as nn
import sympy as sp

from pysignet import Symbol, compile_logic


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
        """Test using symbols as nullary (binary predicates)."""
        P, Q = Symbol("P Q")

        # Used directly without arguments (nullary)
        expr = sp.And(P, Q)

        assert isinstance(expr, sp.And)
        assert len(expr.args) == 2

    def test_unary_usage_multiclass_predicates(self):
        """Test using symbols as unary (multi-class predicates)."""
        Digit = Symbol("Digit")

        # Used with arguments (unary)
        expr = sp.Or(Digit(0), Digit(1), Digit(2))

        assert isinstance(expr, sp.Or)
        assert len(expr.args) == 3

    def test_mixed_nullary_and_unary(self):
        """Test mixing binary and multi-class predicates in same expression."""
        P, Q, Digit = Symbol("P Q Digit")

        # P and Q are nullary, Digit is unary
        expr = sp.And(
            P,
            sp.Or(Q, Digit(0))
        )

        # Should compile without error
        binary_model_p = lambda x: torch.sigmoid(x.mean(dim=-1))
        binary_model_q = lambda x: torch.sigmoid(x.sum(dim=-1))
        multiclass = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )

        predicates = {
            "P": binary_model_p,
            "Q": binary_model_q,
            "Digit": multiclass
        }

        compiled = compile_logic(expr, predicates)

        # Should evaluate successfully
        x = torch.randn(32, 10)
        result = compiled(x)

        assert result.shape == (32,)

    def test_validation_rejects_inconsistent_usage(self):
        """Test that validation rejects predicate used both ways."""
        P, Digit = Symbol("P Digit")

        # P used both as nullary AND unary - INVALID!
        expr = sp.And(P, P(0))

        predicates = {
            "P": nn.Linear(10, 3)
        }

        # Should raise ValueError at compile time
        with pytest.raises(ValueError, match="used inconsistently"):
            compile_logic(expr, predicates)

    def test_validation_rejects_multiclass_without_args(self):
        """Test validation when unary predicate also used without args."""
        Digit = Symbol("Digit")

        # Digit used WITH args and WITHOUT args - INVALID!
        expr = sp.And(Digit(0), Digit)

        predicates = {
            "Digit": nn.Linear(10, 5)
        }

        # Should raise ValueError
        with pytest.raises(ValueError, match="used inconsistently"):
            compile_logic(expr, predicates)

    def test_validation_tracks_arity(self):
        """Test that validation tracks specific arity (not just nullary vs unary)."""
        Rel = Symbol("Rel")

        # Rel used with arity 1 and arity 2 - INVALID!
        # This tests that we track actual arity, not just "has arguments"
        expr = sp.And(Rel(0), Rel(0, 1))

        predicates = {
            "Rel": nn.Linear(10, 3)
        }

        # Should raise ValueError about inconsistent arity
        with pytest.raises(ValueError, match="used inconsistently"):
            compile_logic(expr, predicates)

class TestSymbolAPIUsagePatterns:
    """Test realistic usage patterns with Symbol API."""

    def test_mnist_10class_example(self):
        """Test 10-class MNIST example."""
        Digit = Symbol("Digit")

        # One-hot constraint: exactly one digit
        expr = sp.Or(
            Digit(0), Digit(1), Digit(2), Digit(3), Digit(4),
            Digit(5), Digit(6), Digit(7), Digit(8), Digit(9)
        )

        # Map to classifier
        digit_classifier = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=-1)
        )

        predicates = {"Digit": digit_classifier}
        compiled = compile_logic(expr, predicates)

        # Evaluate
        x = torch.randn(32, 784)
        result = compiled(x)

        assert result.shape == (32,)

    def test_mixed_binary_and_multiclass(self):
        """Test realistic mixing of binary and multi-class predicates."""
        IsAdult, HasLicense, VehicleType = Symbol("IsAdult HasLicense VehicleType")

        # IsAdult and HasLicense are binary (nullary)
        # VehicleType is multi-class (unary): 0=car, 1=motorcycle, 2=truck

        # Rule: Can drive car (type 0) if adult AND has license
        expr = sp.Implies(
            VehicleType(0),
            sp.And(IsAdult, HasLicense)
        )

        # Map to models - same dict structure!
        predicates = {
            "IsAdult": lambda x: (x[:, 0] > 18).float(),
            "HasLicense": lambda x: (x[:, 1] > 0.5).float(),
            "VehicleType": nn.Sequential(
                nn.Linear(10, 3),
                nn.Softmax(dim=-1)
            )
        }

        compiled = compile_logic(expr, predicates)

        x = torch.randn(16, 10)
        result = compiled(x)

        assert result.shape == (16,)

    def test_all_binary_predicates(self):
        """Test that all-binary case still works perfectly."""
        P, Q, R = Symbol("P Q R")

        expr = sp.And(P, sp.Or(Q, sp.Not(R)))

        predicates = {
            "P": lambda x: torch.sigmoid(x[:, 0]),
            "Q": lambda x: torch.sigmoid(x[:, 1]),
            "R": lambda x: torch.sigmoid(x[:, 2])
        }

        compiled = compile_logic(expr, predicates)

        x = torch.randn(32, 10)
        result = compiled(x)

        assert result.shape == (32,)

    def test_all_multiclass_predicates(self):
        """Test all-multiclass case."""
        Color, Shape = Symbol("Color Shape")

        # Color: 0=red, 1=green, 2=blue
        # Shape: 0=circle, 1=square, 2=triangle

        # Red circles or green squares
        expr = sp.Or(
            sp.And(Color(0), Shape(0)),
            sp.And(Color(1), Shape(1))
        )

        predicates = {
            "Color": nn.Sequential(nn.Linear(10, 3), nn.Softmax(dim=-1)),
            "Shape": nn.Sequential(nn.Linear(10, 3), nn.Softmax(dim=-1))
        }

        compiled = compile_logic(expr, predicates)

        x = torch.randn(16, 10)
        result = compiled(x)

        assert result.shape == (16,)
