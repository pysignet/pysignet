"""Tests for TNormCompiler - compiling logic into differentiable callables.

This module tests the TNormCompiler class which compiles SymPy logic
expressions into callable PyTorch functions using t-norm relaxations.
"""

import sympy as sp
import torch
import torch.nn as nn
import pytest

# Import from current location (will be moved to pysignet later)
from pysignet import Predicate, Symbol, TNormCompiler, Variable
from pysignet.tnorms import (
    RProductTNorm,
    SProductTNorm,
    LukasiewiczTNorm,
    GodelTNorm,
)


class TestTNormCompilerBasics:
    """Test basic compilation functionality."""

    def test_tnorm_compiler_initialization(self) -> None:
        """Test TNormCompiler can be initialized with a t-norm."""
        # Test with explicit t-norm
        compiler = TNormCompiler(tnorm=RProductTNorm())
        assert isinstance(compiler.tnorm, RProductTNorm)

        # Test with default (should be RProductTNorm)
        compiler_default = TNormCompiler()
        assert isinstance(compiler_default.tnorm, RProductTNorm)

    def test_compile_returns_callable(self) -> None:
        """Test that compile() returns a callable."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        # Should return a callable
        assert callable(compiled)

    def test_compiled_logic_returns_satisfaction_tensor(self) -> None:
        """Test compiled logic returns satisfaction values in [0,1]."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5)
        result = compiled(x)

        # Should return tensor
        assert isinstance(result, torch.Tensor)
        # Should be batch-sized
        assert result.shape == (10,)
        # Should be in [0, 1]
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_compiled_logic_preserves_batch_dimension(self) -> None:
        """Test compiled logic handles batches correctly."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        # Test different batch sizes
        for batch_size in [1, 5, 32, 100]:
            x = torch.randn(batch_size, 10)
            result = compiled(x)
            assert result.shape == (batch_size,)


class TestTNormCompilerOperators:
    """Test compilation of different logical operators."""

    def test_compile_and_operator(self) -> None:
        """Test compilation of AND operator."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5)
        result = compiled(x)

        # With Product t-norm: AND = P * Q = 0.8 * 0.6 = 0.48
        assert torch.allclose(result, torch.tensor(0.48), atol=1e-5)

    def test_compile_or_operator(self) -> None:
        """Test compilation of OR operator."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.Or(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5)
        result = compiled(x)

        # With Product t-norm: OR = P + Q - P*Q = 0.8 + 0.6 - 0.48 = 0.92
        assert torch.allclose(result, torch.tensor(0.92), atol=1e-5)

    def test_compile_not_operator(self) -> None:
        """Test compilation of NOT operator."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = sp.Not(P(X))

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.3)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5)
        result = compiled(x)

        # NOT = 1 - P = 1 - 0.3 = 0.7
        assert torch.allclose(result, torch.tensor(0.7), atol=1e-5)

    def test_compile_implies_operator(self) -> None:
        """Test compilation of IMPLIES operator."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.Implies(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        compiler = TNormCompiler(tnorm=RProductTNorm())
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5)
        result = compiled(x)

        # R-Product: IMPLIES = 1 if P <= Q else Q/P = 0.6/0.8 = 0.75
        assert torch.allclose(result, torch.tensor(0.75), atol=1e-5)

    def test_compile_equivalent_operator(self) -> None:
        """Test compilation of EQUIVALENT operator."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.Equivalent(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7),
        }

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5)
        result = compiled(x)

        # When P == Q, equivalence should be high (close to 1)
        assert (result > 0.9).all()

    def test_compile_complex_expression(self) -> None:
        """Test compilation of complex nested expressions."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q, R = Symbol("P Q R")
        expr = sp.And(P(X), sp.Or(Q(X), sp.Not(R(X))))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.9),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5),
            "R": Predicate(lambda x: torch.ones(x.shape[0]) * 0.3),
        }

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5)
        result = compiled(x)

        # Should return valid satisfaction
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10,)
        assert (result >= 0).all()
        assert (result <= 1).all()


class TestTNormCompilerWithDifferentTNorms:
    """Test compilation with different t-norm types."""

    def test_compile_with_r_product_tnorm(self) -> None:
        """Test compilation using R-Product t-norm."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        compiler = TNormCompiler(tnorm=RProductTNorm())
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5)
        result = compiled(x)

        # R-Product: AND = P * Q = 0.8 * 0.6 = 0.48
        assert torch.allclose(result, torch.tensor(0.48), atol=1e-5)

    def test_compile_with_s_product_tnorm(self) -> None:
        """Test compilation using S-Product t-norm."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        compiler = TNormCompiler(tnorm=SProductTNorm())
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5)
        result = compiled(x)

        # S-Product: AND = P * Q = 0.8 * 0.6 = 0.48 (same as R-Product for AND)
        assert torch.allclose(result, torch.tensor(0.48), atol=1e-5)

    def test_compile_with_lukasiewicz_tnorm(self) -> None:
        """Test compilation using Lukasiewicz t-norm."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        compiler = TNormCompiler(tnorm=LukasiewiczTNorm())
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5)
        result = compiled(x)

        # Lukasiewicz: AND = max(0, P + Q - 1) = max(0, 0.8 + 0.6 - 1) = 0.4
        assert torch.allclose(result, torch.tensor(0.4), atol=1e-5)

    def test_compile_with_godel_tnorm(self) -> None:
        """Test compilation using Godel t-norm."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        compiler = TNormCompiler(tnorm=GodelTNorm())
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5)
        result = compiled(x)

        # Godel: AND = min(P, Q) = min(0.8, 0.6) = 0.6
        assert torch.allclose(result, torch.tensor(0.6), atol=1e-5)

    def test_different_tnorms_produce_different_results(self) -> None:
        """Test that different t-norms produce different satisfaction."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        x = torch.randn(10, 5)

        # Compile with different t-norms
        r_product_compiler = TNormCompiler(tnorm=RProductTNorm())
        lukasiewicz_compiler = TNormCompiler(tnorm=LukasiewiczTNorm())
        godel_compiler = TNormCompiler(tnorm=GodelTNorm())

        r_product_result = r_product_compiler.compile(expr, predicates)(x)
        lukasiewicz_result = lukasiewicz_compiler.compile(expr, predicates)(x)
        godel_result = godel_compiler.compile(expr, predicates)(x)

        # They should produce different results
        # R-Product: 0.48, Lukasiewicz: 0.4, Godel: 0.6
        assert not torch.allclose(r_product_result, lukasiewicz_result)
        assert not torch.allclose(r_product_result, godel_result)
        assert not torch.allclose(lukasiewicz_result, godel_result)


class TestTNormCompilerGradients:
    """Test gradient flow through compiled logic."""

    def test_gradients_flow_through_compiled_logic(self) -> None:
        """Test gradients flow from compiled logic to predicates."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        # Create a simple neural network predicate
        model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        predicates = {"P": Predicate(model)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5, requires_grad=True)
        result = compiled(x)
        loss = result.sum()
        loss.backward()

        # Check that gradients flow to model parameters
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))

    def test_gradients_with_and_operator(self) -> None:
        """Test gradients flow through AND operator."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        # Create neural network predicates
        model_p = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        model_q = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        predicates = {"P": Predicate(model_p), "Q": Predicate(model_q)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5, requires_grad=True)
        result = compiled(x)
        loss = result.sum()
        loss.backward()

        # Check gradients flow to both models
        for param in model_p.parameters():
            assert param.grad is not None
        for param in model_q.parameters():
            assert param.grad is not None

    def test_gradients_with_or_operator(self) -> None:
        """Test gradients flow through OR operator."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.Or(P(X), Q(X))

        # Create neural network predicates
        model_p = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        model_q = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        predicates = {"P": Predicate(model_p), "Q": Predicate(model_q)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5, requires_grad=True)
        result = compiled(x)
        loss = result.sum()
        loss.backward()

        # Check gradients flow to both models
        for param in model_p.parameters():
            assert param.grad is not None
        for param in model_q.parameters():
            assert param.grad is not None

    def test_gradients_with_complex_expression(self) -> None:
        """Test gradients flow through complex expressions."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q, R = Symbol("P Q R")
        expr = sp.And(P(X), sp.Or(Q(X), sp.Not(R(X))))

        # Create neural network predicates
        model_p = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        model_q = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        model_r = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        predicates = {
            "P": Predicate(model_p),
            "Q": Predicate(model_q),
            "R": Predicate(model_r),
        }

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5, requires_grad=True)
        result = compiled(x)
        loss = result.sum()
        loss.backward()

        # Check gradients flow to all models
        for param in model_p.parameters():
            assert param.grad is not None
        for param in model_q.parameters():
            assert param.grad is not None
        for param in model_r.parameters():
            assert param.grad is not None

    def test_no_gradient_vanishing(self) -> None:
        """Test gradients don't vanish in deep expressions."""
        X = Variable("X")
        # pylint: disable=invalid-name
        # Create deeply nested expression: ((P AND Q) AND (R AND S))
        P, Q, R, S = Symbol("P Q R S")
        expr = sp.And(sp.And(P(X), Q(X)), sp.And(R(X), S(X)))

        # Create neural network predicates
        model_p = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        model_q = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        model_r = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        model_s = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        predicates = {
            "P": Predicate(model_p),
            "Q": Predicate(model_q),
            "R": Predicate(model_r),
            "S": Predicate(model_s),
        }

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5, requires_grad=True)
        result = compiled(x)
        loss = result.sum()
        loss.backward()

        # Check that gradients are non-zero (not vanished)
        for model in [model_p, model_q, model_r, model_s]:
            for param in model.parameters():
                assert param.grad is not None
                # Gradient should not be all zeros
                assert param.grad.abs().max() > 1e-6


class TestTNormCompilerInputHandling:
    """Test compiled logic handles different input types."""

    def test_compiled_logic_with_single_tensor_input(self) -> None:
        """Test compiled logic with single tensor (shared input)."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        # Single tensor input should be shared across all predicates
        x = torch.randn(10, 5)
        result = compiled(x)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (10,)
        assert torch.allclose(result, torch.tensor(0.48), atol=1e-5)

    def test_compiled_logic_with_dict_input(self) -> None:
        """Test compiled logic with dict of tensors (variable-based routing)."""
        X = Variable("X")
        Y = Variable("Y")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(Y))

        # Predicates receive tensors (compiler routes by variable name)
        predicates = {
            "P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1))),
            "Q": Predicate(lambda y: torch.sigmoid(y.mean(dim=-1))),
        }

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        # Dict input with keys matching variable names
        x_p = torch.randn(10, 5)
        x_q = torch.randn(10, 3)
        inputs = {"X": x_p, "Y": x_q}

        result = compiled(inputs)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (10,)


class TestTNormCompilerReusability:
    """Test that compiled logic is reusable across batches."""

    def test_compiled_logic_reusable_across_batches(self) -> None:
        """Test same compiled logic can be called multiple times."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        # Call multiple times with different inputs
        for _ in range(5):
            x = torch.randn(10, 5)
            result = compiled(x)
            assert torch.allclose(result, torch.tensor(0.48), atol=1e-5)

    def test_compiled_logic_with_different_batch_sizes(self) -> None:
        """Test compiled logic handles varying batch sizes."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        # Test with different batch sizes
        for batch_size in [1, 5, 32, 100]:
            x = torch.randn(batch_size, 10)
            result = compiled(x)
            assert result.shape == (batch_size,)
            assert torch.allclose(result, torch.tensor(0.7), atol=1e-5)

    def test_multiple_compilations_independent(self) -> None:
        """Test multiple compilations don't interfere with each other."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr1 = sp.And(P(X), Q(X))
        expr2 = sp.Or(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        compiler = TNormCompiler()
        compiled1 = compiler.compile(expr1, predicates)
        compiled2 = compiler.compile(expr2, predicates)

        x = torch.randn(10, 5)
        result1 = compiled1(x)
        result2 = compiled2(x)

        # Should produce different results (AND vs OR)
        assert torch.allclose(result1, torch.tensor(0.48), atol=1e-5)  # AND
        assert torch.allclose(result2, torch.tensor(0.92), atol=1e-5)  # OR


class TestTNormCompilerBooleanConstants:
    """Test compilation of boolean constants."""

    def test_compile_with_true_constant(self) -> None:
        """Test compilation with sp.true constant."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = sp.And(P(X), sp.true)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5)
        result = compiled(x)

        # AND with true should return P's value
        assert torch.allclose(result, torch.tensor(0.7), atol=1e-5)

    def test_compile_with_false_constant(self) -> None:
        """Test compilation with sp.false constant."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = sp.And(P(X), sp.false)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(10, 5)
        result = compiled(x)

        # AND with false should return 0
        assert torch.allclose(result, torch.tensor(0.0), atol=1e-5)

    def test_constants_preserve_batch_size(self) -> None:
        """Test constants return correct batch-sized tensors."""
        # Test true constant
        expr_true = sp.true
        compiler = TNormCompiler()
        compiled_true = compiler.compile(expr_true, {})

        # Test with different batch sizes
        for batch_size in [1, 10, 50]:
            x = torch.randn(batch_size, 5)
            result = compiled_true(x)
            assert result.shape == (batch_size,)
            assert torch.allclose(result, torch.ones(batch_size), atol=1e-5)

        # Test false constant
        expr_false = sp.false
        compiled_false = compiler.compile(expr_false, {})

        for batch_size in [1, 10, 50]:
            x = torch.randn(batch_size, 5)
            result = compiled_false(x)
            assert result.shape == (batch_size,)
            assert torch.allclose(result, torch.zeros(batch_size), atol=1e-5)

    def test_constants_with_dict_input(self) -> None:
        """Test constants work correctly with dict inputs."""
        # Test true constant with dict input
        expr_true = sp.true
        compiler = TNormCompiler()
        compiled_true = compiler.compile(expr_true, {})

        inputs = {"input1": torch.randn(10, 5), "input2": torch.randn(10, 3)}
        result = compiled_true(inputs)
        assert result.shape == (10,)
        assert torch.allclose(result, torch.ones(10), atol=1e-5)

        # Test false constant with dict input
        expr_false = sp.false
        compiled_false = compiler.compile(expr_false, {})

        result = compiled_false(inputs)
        assert result.shape == (10,)
        assert torch.allclose(result, torch.zeros(10), atol=1e-5)


class TestTNormCompilerErrorHandling:
    """Test error handling in TNormCompiler."""

    def test_missing_predicate_raises_error(self) -> None:
        """Test error when symbol has no corresponding predicate."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        # Only provide predicate for P, not Q
        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8)}

        compiler = TNormCompiler()

        # Should raise ValueError about missing predicate
        with pytest.raises(ValueError, match="Missing predicates for symbols"):
            compiler.compile(expr, predicates)

    def test_clear_error_messages(self) -> None:
        """Test error messages are informative."""
        X = Variable("X")
        # pylint: disable=invalid-name
        # Test 1: Missing predicate error includes symbol name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))
        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8)}

        compiler = TNormCompiler()

        try:
            compiler.compile(expr, predicates)
        except ValueError as e:
            error_msg = str(e)
            assert "Q" in error_msg  # Missing symbol should be mentioned
            assert "Missing" in error_msg or "missing" in error_msg

        # Test 2: Predicate name mismatch error
        predicates_mismatch = {
            "WrongName": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8)
        }

        try:
            compiler.compile(P(X), predicates_mismatch)
        except ValueError as e:
            error_msg = str(e)
            # Should mention the missing predicate name 'P'
            assert "P" in error_msg
            assert "Missing" in error_msg or "missing" in error_msg
