"""Tests for compiling expressions with domain quantifiers.

This module tests that ForAll and Exists quantifiers are properly expanded
and compiled by the TNormCompiler.
"""

import pytest
import torch
import torch.nn as nn
import sympy as sp

from pysignet import Symbol, compile_logic
from pysignet.logic import Variable, ForAll, Exists


class TestBasicQuantifierCompilation:
    """Tests for basic quantifier compilation."""

    def test_forall_compiles_and_evaluates(self):
        """ForAll quantifier compiles and evaluates correctly."""
        X, Y = Variable("X Y")
        Digit = Symbol("Digit")

        # ForAll(Y, [0, 1, 2], Digit(X, Y))
        # Should expand to: Digit(X, 0) ∧ Digit(X, 1) ∧ Digit(X, 2)
        expr = ForAll(Y, [0, 1, 2], Digit(X, Y))

        # 10-class digit classifier
        model = nn.Sequential(nn.Linear(10, 10), nn.Softmax(dim=-1))

        # Compile
        compiled = compile_logic(expr, {"Digit": model})

        # Evaluate
        x = torch.randn(4, 10)
        result = compiled(x)

        # Should return (4,) - one value per batch element
        assert result.shape == (4,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_exists_compiles_and_evaluates(self):
        """Exists quantifier compiles and evaluates correctly."""
        X, Y = Variable("X Y")
        Digit = Symbol("Digit")

        # Exists(Y, [0, 1, 2], Digit(X, Y))
        # Should expand to: Digit(X, 0) ∨ Digit(X, 1) ∨ Digit(X, 2)
        expr = Exists(Y, [0, 1, 2], Digit(X, Y))

        # Model
        model = nn.Sequential(nn.Linear(10, 10), nn.Softmax(dim=-1))

        # Compile
        compiled = compile_logic(expr, {"Digit": model})

        # Evaluate
        x = torch.randn(4, 10)
        result = compiled(x)

        assert result.shape == (4,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_nested_quantifiers_compile(self):
        """Nested quantifiers compile correctly."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        # ForAll(Y, [0, 1], ForAll(Z, [2, 3], P(X, Y, Z)))
        inner = ForAll(Z, [2, 3], P(X, Y, Z))
        expr = ForAll(Y, [0, 1], inner)

        # Ternary predicate
        model = nn.Sequential(nn.Linear(5, 12), nn.Softmax(dim=-1))

        # Compile
        compiled = compile_logic(expr, {"P": model})

        # Evaluate
        x = torch.randn(3, 5)
        result = compiled(x)

        assert result.shape == (3,)


class TestDomainSizeLimits:
    """Tests for domain size validation."""

    def test_large_domain_warns(self):
        """Large domain (>100) triggers warning."""
        Y = Variable("Y")
        P = Symbol("P")

        expr = ForAll(Y, range(150), P(Y))

        model = nn.Sequential(nn.Linear(5, 200), nn.Softmax(dim=-1))

        # Should warn
        with pytest.warns(UserWarning, match="Large domain"):
            compiled = compile_logic(expr, {"P": model})

    def test_huge_domain_raises_error(self):
        """Huge domain (>1000) raises error."""
        Y = Variable("Y")
        P = Symbol("P")

        expr = ForAll(Y, range(1500), P(Y))

        model = lambda x: torch.softmax(x, dim=-1)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Domain too large"):
            compiled = compile_logic(expr, {"P": model})

    def test_small_domain_no_warning(self):
        """Small domain (<100) works without warning."""
        import warnings

        Y = Variable("Y")
        P = Symbol("P")

        expr = ForAll(Y, range(10), P(Y))

        model = nn.Sequential(nn.Linear(5, 15), nn.Softmax(dim=-1))

        # Should not warn - just compile successfully
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compiled = compile_logic(expr, {"P": model})

            # Check no domain-size warnings were issued
            domain_warnings = [warning for warning in w
                             if "domain" in str(warning.message).lower()]
            assert len(domain_warnings) == 0


class TestGradientFlow:
    """Tests for gradient flow through quantifiers."""

    def test_gradients_flow_through_forall(self):
        """Gradients flow through ForAll quantification."""
        X, Y = Variable("X Y")
        Digit = Symbol("Digit")

        expr = ForAll(Y, [0, 1], Digit(X, Y))

        model = nn.Sequential(nn.Linear(5, 10), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"Digit": model})

        x = torch.randn(3, 5)
        loss = compiled.loss(x)

        loss.backward()

        # Verify gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_gradients_flow_through_exists(self):
        """Gradients flow through Exists quantification."""
        X, Y = Variable("X Y")
        P = Symbol("P")

        expr = Exists(Y, [0, 1, 2], P(X, Y))

        model = nn.Sequential(nn.Linear(4, 5), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        x = torch.randn(2, 4)
        loss = compiled.loss(x)

        loss.backward()

        for param in model.parameters():
            assert param.grad is not None


class TestRealWorldPatterns:
    """Tests for real-world usage patterns."""

    def test_one_hot_constraint(self):
        """One-hot constraint: exactly one class."""
        X, Y = Variable("X Y")
        Digit = Symbol("Digit")

        # Exists(Y, range(10), Digit(X, Y))
        # "Each X is classified as some digit 0-9"
        expr = Exists(Y, range(10), Digit(X, Y))

        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=-1)
        )

        compiled = compile_logic(expr, {"Digit": model})

        # Batch of "images"
        x = torch.randn(4, 784)
        result = compiled(x)

        assert result.shape == (4,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_even_digits_constraint(self):
        """Constraint on subset of classes."""
        X, Y = Variable("X Y")
        Digit, Even = Symbol("Digit Even")

        # ForAll(Y, [0,2,4,6,8], Digit(X, Y) → Even)
        # "For even digits, if X is classified as that digit, then the Even predicate holds"
        # Note: Even is a nullary predicate (no arguments)
        body = sp.Implies(Digit(X, Y), Even)
        expr = ForAll(Y, [0, 2, 4, 6, 8], body)

        digit_model = nn.Sequential(nn.Linear(10, 10), nn.Softmax(dim=-1))
        # Even model takes full input and returns scalar per batch
        even_model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

        # Need to make even_model return squeezed output
        def even_func(inputs):
            return even_model(inputs).squeeze(-1)

        compiled = compile_logic(expr, {
            "Digit": digit_model,
            "Even": even_func
        })

        x = torch.randn(3, 10)
        result = compiled(x)

        assert result.shape == (3,)
        assert torch.all((result >= 0) & (result <= 1))
