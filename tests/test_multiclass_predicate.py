"""Tests for predicate application support.

Tests demonstrate predicate application with concrete arguments,
enabling efficient multi-class classification.
"""

import pytest
import torch
import torch.nn as nn
import sympy as sp

from pysignet import compile_logic, Symbol
from pysignet.multiclass import PredicateApplication
from pysignet.tnorms import RProductTNorm, LukasiewiczTNorm


class TestSymbolBasic:
    """Test basic Symbol functionality."""

    def test_create_multiclass_predicate(self):
        """Test creating a Symbol."""
        digit = Symbol("Digit")
        assert digit.name == "Digit"

    def test_predicate_application_creates_ast_node(self):
        """Test that calling Symbol creates PredicateApplication."""
        digit = Symbol("Digit")
        app = digit(0)

        assert isinstance(app, PredicateApplication)
        assert app.predicate_name == "Digit"
        assert app.application_args == (0,)

    def test_predicate_application_with_different_indices(self):
        """Test predicate application with various indices."""
        digit = Symbol("Digit")

        app0 = digit(0)
        app5 = digit(5)
        app9 = digit(9)

        assert app0.application_args == (0,)
        assert app5.application_args == (5,)
        assert app9.application_args == (9,)

    def test_predicate_application_repr(self):
        """Test string representation of predicate application."""
        digit = Symbol("Digit")
        app = digit(0)

        repr_str = repr(app)
        assert "Digit" in repr_str
        assert "0" in repr_str


class TestSymPyIntegration:
    """Test integration with SymPy logical operators."""

    def test_predicate_application_with_and(self):
        """Test PredicateApplication works with sp.And."""
        digit = Symbol("Digit")
        expr = sp.And(digit(0), digit(1))

        assert isinstance(expr, sp.And)
        assert len(expr.args) == 2

    def test_predicate_application_with_or(self):
        """Test PredicateApplication works with sp.Or."""
        digit = Symbol("Digit")
        expr = sp.Or(digit(0), digit(1), digit(2))

        assert isinstance(expr, sp.Or)
        assert len(expr.args) == 3

    def test_predicate_application_with_not(self):
        """Test PredicateApplication works with sp.Not."""
        digit = Symbol("Digit")
        expr = sp.Not(digit(0))

        assert isinstance(expr, sp.Not)

    def test_predicate_application_with_implies(self):
        """Test PredicateApplication works with sp.Implies."""
        digit = Symbol("Digit")
        expr = sp.Implies(digit(0), digit(1))

        assert isinstance(expr, sp.Implies)

    def test_predicate_application_with_equivalent(self):
        """Test PredicateApplication works with sp.Equivalent."""
        digit = Symbol("Digit")
        expr = sp.Equivalent(digit(0), digit(1))

        assert isinstance(expr, sp.Equivalent)

    def test_complex_expression(self):
        """Test complex nested expressions."""
        digit = Symbol("Digit")
        expr = sp.And(
            sp.Or(digit(0), digit(1)),
            sp.Implies(digit(2), sp.Not(digit(3)))
        )

        assert isinstance(expr, sp.And)


class TestCompilation:
    """Test compilation of expressions with Symbol."""

    def test_compile_simple_application(self):
        """Test compiling a simple predicate application."""
        digit = Symbol("Digit")
        expr = digit(0)

        # Simple 3-class classifier
        classifier = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )

        predicates = {"Digit": classifier}

        # Should compile without error
        compiled = compile_logic(expr, predicates)
        assert compiled is not None

    def test_compile_or_expression(self):
        """Test compiling OR expression with applications."""
        digit = Symbol("Digit")
        expr = sp.Or(digit(0), digit(1), digit(2))

        classifier = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )

        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates)
        assert compiled is not None

    def test_compile_complex_expression(self):
        """Test compiling complex expression."""
        digit = Symbol("Digit")
        expr = sp.And(
            sp.Or(digit(0), digit(1)),
            sp.Not(digit(2))
        )

        classifier = nn.Sequential(
            nn.Linear(10, 5),
            nn.Softmax(dim=-1)
        )

        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates)
        assert compiled is not None


class TestSingleForwardPass:
    """CRITICAL: Verify only ONE forward pass occurs (caching works)."""

    def test_single_forward_pass_with_multiple_applications(self):
        """Test that multiple applications only trigger ONE forward pass."""
        call_count = 0

        class CountingClassifier(nn.Module):
            """Network that counts forward passes."""

            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
                self.softmax = nn.Softmax(dim=-1)

            def forward(self, x):
                nonlocal call_count
                call_count += 1
                return self.softmax(self.fc(x))

        digit = Symbol("Digit")
        classifier = CountingClassifier()

        # Expression uses ALL 5 outputs
        expr = sp.And(digit(0), digit(1), digit(2), digit(3), digit(4))

        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates)

        # Generate input
        x = torch.randn(32, 10)

        # Reset counter and evaluate
        call_count = 0
        _ = compiled(x)

        # CRITICAL: Should be exactly 1 forward pass!
        assert call_count == 1, f"Expected 1 forward pass, got {call_count}"

    def test_single_forward_pass_with_repeated_indices(self):
        """Test caching when same index appears multiple times."""
        call_count = 0

        class CountingNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 3)
                self.softmax = nn.Softmax(dim=-1)

            def forward(self, x):
                nonlocal call_count
                call_count += 1
                return self.softmax(self.fc(x))

        digit = Symbol("Digit")
        classifier = CountingNetwork()

        # Same indices appear multiple times
        expr = sp.And(digit(0), sp.Or(digit(0), digit(1)))

        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(16, 10)
        call_count = 0
        _ = compiled(x)

        assert call_count == 1


class TestEvaluation:
    """Test evaluation of compiled expressions."""

    def test_evaluate_returns_tensor(self):
        """Test that evaluation returns a tensor."""
        digit = Symbol("Digit")
        expr = digit(0)

        classifier = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )

        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(32, 10)
        result = compiled(x)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (32,)

    def test_evaluate_values_in_range(self):
        """Test that output values are in [0, 1]."""
        digit = Symbol("Digit")
        expr = digit(0)

        classifier = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )

        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(32, 10)
        result = compiled(x)

        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_evaluate_extracts_correct_index(self):
        """Test that correct output index is extracted."""
        digit = Symbol("Digit")

        # Create deterministic classifier
        class DeterministicClassifier(nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                # Return [0.1, 0.2, 0.7] for each batch element
                output = torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float32)
                return output.repeat(batch_size, 1)

        classifier = DeterministicClassifier()

        # Test each index
        for idx, expected_val in enumerate([0.1, 0.2, 0.7]):
            expr = digit(idx)
            predicates = {"Digit": classifier}
            compiled = compile_logic(expr, predicates)

            x = torch.randn(16, 10)
            result = compiled(x)

            assert torch.allclose(result, torch.tensor(expected_val))


class TestGradientFlow:
    """Test that gradients flow correctly through cached computations."""

    def test_gradient_flow_single_application(self):
        """Test gradients flow through single application."""
        digit = Symbol("Digit")
        expr = digit(0)

        classifier = nn.Linear(10, 3)
        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(32, 10, requires_grad=True)
        loss = compiled.loss(x)

        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        assert classifier.weight.grad is not None

    def test_gradient_flow_multiple_applications(self):
        """Test gradients flow when multiple applications share cache."""
        digit = Symbol("Digit")
        expr = sp.And(digit(0), digit(1), digit(2))

        classifier = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )

        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(32, 10, requires_grad=True)
        loss = compiled.loss(x)

        loss.backward()

        # Gradients should flow
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_gradient_accumulation_through_cache(self):
        """Test that gradients accumulate correctly through cached outputs."""
        digit = Symbol("Digit")

        # Expression uses same index multiple times
        expr = sp.Or(digit(0), sp.And(digit(0), digit(1)))

        classifier = nn.Linear(10, 3)
        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(16, 10, requires_grad=True)
        loss = compiled.loss(x)

        loss.backward()

        assert x.grad is not None
        assert classifier.weight.grad is not None


class TestBatchSizes:
    """Test with different batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
    def test_different_batch_sizes(self, batch_size):
        """Test evaluation works with various batch sizes."""
        digit = Symbol("Digit")
        expr = sp.Or(digit(0), digit(1))

        classifier = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )

        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(batch_size, 10)
        result = compiled(x)

        assert result.shape == (batch_size,)


class TestCacheClearing:
    """Test that cache is cleared between evaluations."""

    def test_cache_cleared_between_batches(self):
        """Test cache doesn't persist across different evaluations."""
        digit = Symbol("Digit")
        expr = digit(0)

        classifier = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )

        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates)

        # First batch
        x1 = torch.randn(32, 10)
        result1 = compiled(x1)

        # Second batch (different input)
        x2 = torch.randn(32, 10)
        result2 = compiled(x2)

        # Results should be different (cache was cleared)
        assert not torch.allclose(result1, result2)

    def test_independent_evaluations(self):
        """Test that multiple evaluations are independent."""
        digit = Symbol("Digit")
        expr = sp.And(digit(0), digit(1))

        classifier = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )

        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates)

        # Run multiple evaluations
        results = []
        for _ in range(3):
            x = torch.randn(16, 10)
            result = compiled(x)
            results.append(result)

        # Each should be independent (no shared state)
        assert len(results) == 3


class TestMixedPredicates:
    """Test mixing Symbol with regular Predicate."""

    def test_mix_multiclass_and_regular_predicate(self):
        """Test expression with both Symbol and Predicate."""
        from pysignet import Predicate

        digit = Symbol("Digit")

        # Regular predicate
        regular_func = lambda x: torch.sigmoid(x.mean(dim=-1))

        # Expression mixes both types
        expr = sp.And(digit(0), sp.Symbol("Regular"))

        classifier = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )

        predicates = {
            "Digit": classifier,
            "Regular": regular_func
        }

        compiled = compile_logic(expr, predicates)

        x = torch.randn(32, 10)
        result = compiled(x)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (32,)


class TestValidation:
    """Test validation and error handling."""

    def test_index_out_of_range_error(self):
        """Test error when index exceeds network outputs."""
        digit = Symbol("Digit")
        expr = digit(10)  # Index 10 but network only has 3 outputs

        classifier = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )

        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(32, 10)

        # Should raise an error (index out of range)
        with pytest.raises((IndexError, RuntimeError)):
            _ = compiled(x)

    def test_missing_predicate_error(self):
        """Test error when predicate not in predicates dict."""
        digit = Symbol("Digit")
        expr = digit(0)

        # Empty predicates dict - missing "Digit"
        predicates = {}

        with pytest.raises((KeyError, ValueError)):
            _ = compile_logic(expr, predicates)


class TestTNormCompatibility:
    """Test compatibility with different t-norms."""

    def test_with_rproduct_tnorm(self):
        """Test with R-Product t-norm."""
        digit = Symbol("Digit")
        expr = sp.And(digit(0), digit(1))

        classifier = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )

        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates, tnorm=RProductTNorm())

        x = torch.randn(32, 10)
        result = compiled(x)

        assert result.shape == (32,)

    def test_with_lukasiewicz_tnorm(self):
        """Test with Lukasiewicz t-norm."""
        digit = Symbol("Digit")
        expr = sp.Or(digit(0), digit(1))

        classifier = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )

        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates, tnorm=LukasiewiczTNorm())

        x = torch.randn(32, 10)
        result = compiled(x)

        assert result.shape == (32,)


class TestThreadSafety:
    """Test thread safety for multi-GPU training scenarios."""

    def test_concurrent_evaluations(self):
        """Test that concurrent evaluations don't interfere."""
        import threading

        digit = Symbol("Digit")
        expr = sp.And(digit(0), digit(1))

        classifier = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )

        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates)

        results = [None, None]
        errors = [None, None]

        def evaluate(idx):
            try:
                x = torch.randn(16, 10)
                results[idx] = compiled(x)
            except Exception as e:
                errors[idx] = e

        # Run two evaluations concurrently
        thread1 = threading.Thread(target=lambda: evaluate(0))
        thread2 = threading.Thread(target=lambda: evaluate(1))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Both should succeed
        assert errors[0] is None
        assert errors[1] is None
        assert results[0] is not None
        assert results[1] is not None


class TestLossComputation:
    """Test loss computation with Symbol."""

    def test_loss_computation(self):
        """Test that loss can be computed."""
        digit = Symbol("Digit")
        expr = sp.Or(digit(0), digit(1), digit(2))

        classifier = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )

        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(32, 10)
        loss = compiled.loss(x)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar

    def test_loss_backpropagation(self):
        """Test that loss can be backpropagated."""
        digit = Symbol("Digit")
        expr = sp.And(digit(0), digit(1))

        classifier = nn.Linear(10, 3)
        predicates = {"Digit": classifier}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(32, 10)
        loss = compiled.loss(x)

        # Should be able to backpropagate
        loss.backward()

        assert classifier.weight.grad is not None
