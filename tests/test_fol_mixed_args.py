"""Tests for PredicateApplication with mixed variable/constant arguments.

This module tests that predicates can have arguments that are a mix of:
- Free variables (batch-quantified)
- Constants (output indices or values)

Examples:
- P(X1, X2, 0) - two variables, one constant
- P(X, 0, X1) - two variables with constant in between
- P(X1, X2, X3, 0, 1) - three variables, two constants
"""

import pytest
import torch
import torch.nn as nn

from pysignet import Symbol, compile_logic
from pysignet.logic import Variable


class TestMixedArgumentArityValidation:
    """Tests for arity validation with mixed arguments."""

    def test_two_variables_one_constant_valid(self):
        """P(X1, X2, 0) with 3-argument callable is valid."""
        X1, X2 = Variable("X1 X2")
        P = Symbol("P")

        expr = P(X1, X2, 0)

        # Callable accepts 3 arguments (X1, X2, and constant 0)
        def ternary_pred(x1, x2, const):
            # Simple function for testing
            return torch.sigmoid(x1.sum(dim=-1) + x2.sum(dim=-1) + const)

        # Should compile successfully
        predicates = {"P": ternary_pred}
        compiled = compile_logic(expr, predicates)

        # Evaluate
        batch_size = 3
        x1 = torch.randn(batch_size, 5)
        x2 = torch.randn(batch_size, 4)

        # Use quantify='none' to get per-batch results
        result = compiled({"X1": x1, "X2": x2}, quantify='none')

        # Should return satisfaction for batch
        assert result.shape == (batch_size,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_two_variables_one_constant_invalid_arity(self):
        """P(X1, X2, 0) with 2-argument callable raises error."""
        X1, X2 = Variable("X1 X2")
        P = Symbol("P")

        expr = P(X1, X2, 0)

        # Callable only accepts 2 arguments (WRONG - should accept 3)
        def binary_pred(x1, x2):
            return torch.sigmoid(x1.sum(dim=-1) + x2.sum(dim=-1))

        predicates = {"P": binary_pred}

        # Should raise ValueError about arity mismatch
        with pytest.raises(ValueError, match="arity"):
            compile_logic(expr, predicates)

    def test_one_variable_one_constant_valid(self):
        """P(X, 0) with 2-argument callable is valid."""
        X = Variable("X")
        P = Symbol("P")

        expr = P(X, 0)

        # Callable accepts 2 arguments (X and constant)
        def binary_pred(x, const):
            return torch.sigmoid(x.sum(dim=-1) + const)

        predicates = {"P": binary_pred}
        compiled = compile_logic(expr, predicates)

        # Evaluate
        batch_size = 4
        x = torch.randn(batch_size, 5)

        # Use quantify='none' to get per-batch results
        result = compiled({"X": x}, quantify='none')

        assert result.shape == (batch_size,)

    def test_three_variables_two_constants_valid(self):
        """P(X1, X2, X3, 0, 1) with 5-argument callable is valid."""
        X1, X2, X3 = Variable("X1 X2 X3")
        P = Symbol("P")

        expr = P(X1, X2, X3, 0, 1)

        # Callable accepts 5 arguments (3 variables + 2 constants)
        def five_arg_pred(x1, x2, x3, c1, c2):
            return torch.sigmoid(
                x1.sum(dim=-1) + x2.sum(dim=-1) + x3.sum(dim=-1) + c1 + c2
            )

        predicates = {"P": five_arg_pred}
        compiled = compile_logic(expr, predicates)

        # Evaluate
        batch_size = 2
        x1 = torch.randn(batch_size, 3)
        x2 = torch.randn(batch_size, 4)
        x3 = torch.randn(batch_size, 2)

        # Use quantify='none' to get per-batch results
        result = compiled({"X1": x1, "X2": x2, "X3": x3}, quantify='none')

        assert result.shape == (batch_size,)

    def test_variables_interleaved_with_constants(self):
        """P(X1, 0, X2, 1) with 4-argument callable is valid."""
        X1, X2 = Variable("X1 X2")
        P = Symbol("P")

        expr = P(X1, 0, X2, 1)

        # Callable accepts 4 arguments (X1, constant 0, X2, constant 1)
        def quaternary_pred(x1, c1, x2, c2):
            return torch.sigmoid(x1.sum(dim=-1) + x2.sum(dim=-1) + c1 + c2)

        predicates = {"P": quaternary_pred}
        compiled = compile_logic(expr, predicates)

        # Evaluate
        batch_size = 3
        x1 = torch.randn(batch_size, 5)
        x2 = torch.randn(batch_size, 4)

        # Use quantify='none' to get per-batch results
        result = compiled({"X1": x1, "X2": x2}, quantify='none')

        assert result.shape == (batch_size,)


class TestMixedArgumentEvaluation:
    """Tests for evaluation with mixed arguments."""

    def test_constant_used_as_output_index(self):
        """Constant in P(X, 0) selects output channel 0."""
        X = Variable("X")
        Digit = Symbol("Digit")

        expr = Digit(X, 0)

        # Multi-output model (10 classes)
        model = nn.Sequential(nn.Linear(5, 10), nn.Softmax(dim=-1))

        predicates = {"Digit": model}
        compiled = compile_logic(expr, predicates)

        # Evaluate
        batch_size = 4
        x = torch.randn(batch_size, 5)

        # Use quantify='none' to get per-batch results
        result = compiled({"X": x}, quantify='none')

        # Should return satisfaction for batch
        # Value should be in [0, 1] (probabilities for class 0)
        assert result.shape == (batch_size,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_multiple_constants_select_output_channels(self):
        """Multiple constants in P(X, 0, 1) select multiple channels."""
        X = Variable("X")
        P = Symbol("P")

        expr = P(X, 0, 1)

        # Model with multiple outputs
        model = nn.Sequential(nn.Linear(5, 10), nn.Softmax(dim=-1))

        predicates = {"P": model}
        compiled = compile_logic(expr, predicates)

        # Evaluate
        batch_size = 3
        x = torch.randn(batch_size, 5)

        # Use quantify='none' to get per-batch results
        result = compiled({"X": x}, quantify='none')

        assert result.shape == (batch_size,)

    def test_nn_module_as_multiarg_predicate(self):
        """nn.Module can be used as multi-argument predicate."""
        X1, X2 = Variable("X1 X2")
        P = Symbol("P")

        expr = P(X1, X2, 0)

        # Custom module that accepts multiple inputs
        class MultiInputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(5, 10)
                self.linear2 = nn.Linear(4, 10)

            def forward(self, x1, x2):
                # Combine inputs
                out1 = self.linear1(x1)
                out2 = self.linear2(x2)
                combined = out1 + out2
                return torch.softmax(combined, dim=-1)

        model = MultiInputModule()

        predicates = {"P": model}
        compiled = compile_logic(expr, predicates)

        # Evaluate
        batch_size = 3
        x1 = torch.randn(batch_size, 5)
        x2 = torch.randn(batch_size, 4)

        # Use quantify='none' to get per-batch results
        result = compiled({"X1": x1, "X2": x2}, quantify='none')

        assert result.shape == (batch_size,)


class TestEdgeCases:
    """Tests for edge cases with mixed arguments."""

    def test_no_variables_only_constants(self):
        """P(0, 1, 2) with no variables is valid."""
        P = Symbol("P")

        expr = P(0, 1, 2)

        # Callable accepts 3 arguments (the 3 constants)
        def ternary_pred(c1, c2, c3):
            # Returns value in [0, 1] range based on constants
            return torch.tensor(0.1 * (c1 + c2 + c3))

        predicates = {"P": ternary_pred}
        compiled = compile_logic(expr, predicates)

        # Evaluate without free variable inputs
        result = compiled({})

        # No free variables -> scalar output
        assert result.shape == ()
        # Should be 0.1 * (0 + 1 + 2) = 0.3
        assert torch.isclose(result, torch.tensor(0.3))

    def test_duplicate_variables_counted_once(self):
        """P(X, X, 0) with duplicate variable requires 3-arg callable."""
        X = Variable("X")
        P = Symbol("P")

        expr = P(X, X, 0)

        # Callable accepts 3 arguments (X, X, and constant 0)
        # X appears twice so gets passed twice
        def ternary_pred(x1, x2, const):
            return torch.sigmoid(x1.sum(dim=-1) + x2.sum(dim=-1) + const)

        predicates = {"P": ternary_pred}
        compiled = compile_logic(expr, predicates)

        # Evaluate
        batch_size = 2
        x = torch.randn(batch_size, 5)

        # Use quantify='none' to get per-batch results
        result = compiled({"X": x}, quantify='none')

        assert result.shape == (batch_size,)

    def test_single_variable_no_constants(self):
        """P(X) with single variable requires 1-arg callable."""
        X = Variable("X")
        P = Symbol("P")

        expr = P(X)

        # Callable accepts 1 argument
        model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())

        predicates = {"P": model}
        compiled = compile_logic(expr, predicates)

        # Evaluate
        batch_size = 3
        x = torch.randn(batch_size, 5)

        # Use quantify='none' to get per-batch results
        result = compiled({"X": x}, quantify='none')

        assert result.shape == (batch_size,)


class TestGradientFlow:
    """Tests for gradient flow with mixed arguments."""

    def test_gradients_flow_with_multiple_variables(self):
        """Gradients flow through predicates with multiple variables."""
        X1, X2 = Variable("X1 X2")
        P = Symbol("P")

        expr = P(X1, X2, 0)

        # Module with parameters
        class BinaryPredicate(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(5, 8)
                self.linear2 = nn.Linear(4, 8)
                self.combine = nn.Linear(16, 10)

            def forward(self, x1, x2):
                out1 = self.linear1(x1)
                out2 = self.linear2(x2)
                combined = torch.cat([out1, out2], dim=-1)
                return torch.softmax(self.combine(combined), dim=-1)

        model = BinaryPredicate()

        predicates = {"P": model}
        compiled = compile_logic(expr, predicates)

        # Evaluate
        batch_size = 2
        x1 = torch.randn(batch_size, 5)
        x2 = torch.randn(batch_size, 4)

        # Compute loss
        loss = compiled.loss({"X1": x1, "X2": x2})

        # Backward
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestArityValidationErrors:
    """Tests for arity validation error paths."""

    def test_inconsistent_arity_nullary_vs_nary(self):
        """Error when predicate used as both nullary and n-ary."""
        X = Variable("X")
        P = Symbol("P")

        # P used both as nullary (P) and unary (P(X)) in same expression
        import sympy as sp
        expr = sp.And(P, P(X))  # INCONSISTENT!

        predicates = {"P": lambda x: torch.sigmoid(x.sum(dim=-1))}

        with pytest.raises(ValueError, match="used inconsistently"):
            compile_logic(expr, predicates)

    def test_nullary_arity_mismatch_too_many_args(self):
        """Error when predicate is used without arguments (nullary not allowed)."""
        P = Symbol("P")

        # P used as nullary (no arguments) - this is disallowed
        expr = P

        # Callable takes 2 arguments
        predicates = {"P": lambda x, y: torch.tensor(0.5)}

        with pytest.raises(ValueError, match="used without arguments"):
            compile_logic(expr, predicates)

    def test_arity_validation_with_nn_module(self):
        """nn.Module predicates with auto-detectable arity."""
        X = Variable("X")
        P = Symbol("P")

        expr = P(X)

        # Use Sequential with recognizable structure (unary)
        model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        predicates = {"P": model}

        # Should compile successfully
        compiled = compile_logic(expr, predicates)

        batch_size = 3
        x = torch.randn(batch_size, 5)
        # Use quantify='none' to get per-batch results
        result = compiled({"X": x}, quantify='none')

        assert result.shape == (batch_size,)

    # TODO: Bound method arity validation needs work - inspect.ismethod
    # detection is unreliable. Defer until arity validation refactoring.
