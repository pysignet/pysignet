"""Tests for compile_logic() factory function - convenience API.

This module tests the compile_logic() factory function which provides a
one-liner for common usage patterns.
"""

import sympy as sp
import torch
import torch.nn as nn
import pytest

# Import from current location (will be moved to neural_logic later)
from pysignet import LogicLoss, Predicate, Symbol, Variable, compile_logic
from pysignet.tnorms import (
    RProductTNorm,
    SProductTNorm,
    LukasiewiczTNorm,
)


class TestCompileLogicBasics:
    """Test basic compile_logic() functionality."""

    def test_compile_logic_with_tnorm_mode(self) -> None:
        """Test compile_logic with mode='tnorm'."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        logic_loss = compile_logic(expr, predicates, mode="tnorm")
        x = torch.randn(10, 5)
        satisfaction = logic_loss(x)

        # Should return satisfaction values
        assert torch.allclose(satisfaction, torch.tensor(0.7), atol=1e-5)

    def test_compile_logic_returns_logic_loss(self) -> None:
        """Test compile_logic returns LogicLoss instance."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)}

        logic_loss = compile_logic(expr, predicates)

        # Should return LogicLoss instance
        assert isinstance(logic_loss, LogicLoss)

    def test_compile_logic_default_mode(self) -> None:
        """Test compile_logic uses tnorm as default mode."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

        # Should work without specifying mode (defaults to 'tnorm')
        logic_loss = compile_logic(expr, predicates)
        x = torch.randn(10, 5)
        satisfaction = logic_loss(x)

        assert torch.allclose(satisfaction, torch.tensor(0.6), atol=1e-5)


class TestCompileLogicTNormMode:
    """Test compile_logic with different t-norms."""

    def test_compile_logic_with_r_product(self) -> None:
        """Test with R-Product t-norm."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        logic_loss = compile_logic(expr, predicates, tnorm=RProductTNorm())
        x = torch.randn(10, 5)
        satisfaction = logic_loss(x)

        # R-Product AND: 0.8 * 0.6 = 0.48
        assert torch.allclose(satisfaction, torch.tensor(0.48), atol=1e-5)

    def test_compile_logic_with_s_product(self) -> None:
        """Test with S-Product t-norm."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        logic_loss = compile_logic(expr, predicates, tnorm=SProductTNorm())
        x = torch.randn(10, 5)
        satisfaction = logic_loss(x)

        # S-Product AND: 0.8 * 0.6 = 0.48 (same as R-Product for AND)
        assert torch.allclose(satisfaction, torch.tensor(0.48), atol=1e-5)

    def test_compile_logic_with_lukasiewicz(self) -> None:
        """Test with Lukasiewicz t-norm."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        logic_loss = compile_logic(expr, predicates, tnorm=LukasiewiczTNorm())
        x = torch.randn(10, 5)
        satisfaction = logic_loss(x)

        # Lukasiewicz AND: max(0, 0.8 + 0.6 - 1) = 0.4
        expected = max(0.0, 0.8 + 0.6 - 1.0)
        assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)

    def test_compile_logic_default_tnorm(self) -> None:
        """Test default t-norm (should be R-Product)."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        # Don't specify tnorm - should default to R-Product
        logic_loss = compile_logic(expr, predicates)
        x = torch.randn(10, 5)
        satisfaction = logic_loss(x)

        # Default (R-Product) AND: 0.8 * 0.6 = 0.48
        assert torch.allclose(satisfaction, torch.tensor(0.48), atol=1e-5)


class TestCompileLogicPostProcessing:
    """Test compile_logic with post-processing options."""

    def test_compile_logic_with_auto_postprocessing(self) -> None:
        """Test default post-processing uses t-norm's recommendation."""
        X = Variable("X")
        # Default behavior: use t-norm's recommended post-processing
        # RProductTNorm (default) recommends 'log'
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        logic_loss = compile_logic(expr, predicates)
        x = torch.randn(10, 5)
        loss = logic_loss.loss(x)

        # Default uses RProductTNorm with 'log': -log(satisfaction)
        expected_loss = -torch.log(torch.tensor(0.7))
        assert torch.allclose(loss, expected_loss, atol=1e-5)

    def test_compile_logic_with_log_postprocessing(self) -> None:
        """Test post_processing='log'."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)}

        logic_loss = compile_logic(expr, predicates, post_processing="log")
        x = torch.randn(10, 5)
        loss = logic_loss.loss(x)

        # Log post-processing: -log(satisfaction) = -log(0.5)
        expected_loss = -torch.log(torch.tensor(0.5))
        assert torch.allclose(loss, expected_loss, atol=1e-5)

    def test_compile_logic_with_linear_postprocessing(self) -> None:
        """Test post_processing='linear'."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

        logic_loss = compile_logic(expr, predicates, post_processing="linear")
        x = torch.randn(10, 5)
        loss = logic_loss.loss(x)

        # Linear post-processing: 1 - satisfaction = 1 - 0.6 = 0.4
        assert torch.allclose(loss, torch.tensor(0.4), atol=1e-5)

    def test_compile_logic_with_custom_postprocessing(self) -> None:
        """Test custom post-processing callable."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

        # Custom post-processing: square the violation
        def custom_postprocessing(satisfaction):
            return (1 - satisfaction) ** 2

        logic_loss = compile_logic(
            expr, predicates, post_processing=custom_postprocessing
        )
        x = torch.randn(10, 5)
        loss = logic_loss.loss(x)

        # Custom: (1 - 0.6)^2 = 0.4^2 = 0.16
        expected_loss = (1.0 - 0.6) ** 2
        assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-5)


class TestCompileLogicUsability:
    """Test compile_logic is easy to use."""

    def test_one_liner_usage(self) -> None:
        """Test compile_logic enables one-liner usage."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        # One-liner API
        logic_loss = compile_logic(
            expr,
            {
                "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
                "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
            },
        )

        x = torch.randn(10, 5)
        loss = logic_loss.loss(x)

        # Should compute loss correctly
        # Product AND: 0.8 * 0.6 = 0.48
        # RProductTNorm recommends 'log': -log(0.48)
        satisfaction = 0.8 * 0.6
        expected_loss = -torch.log(torch.tensor(satisfaction))
        assert torch.allclose(loss, expected_loss, atol=1e-5)

    def test_compiled_logic_reusable(self) -> None:
        """Test compiled logic can be reused across batches."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        logic_loss = compile_logic(expr, predicates)

        # Use with different batch sizes
        x1 = torch.randn(5, 3)
        x2 = torch.randn(20, 3)
        x3 = torch.randn(1, 3)

        satisfaction1 = logic_loss(x1)
        satisfaction2 = logic_loss(x2)
        satisfaction3 = logic_loss(x3)

        # All should return correct satisfaction
        assert satisfaction1.shape == (5,)
        assert satisfaction2.shape == (20,)
        assert satisfaction3.shape == (1,)
        assert torch.allclose(satisfaction1, torch.tensor(0.7), atol=1e-5)
        assert torch.allclose(satisfaction2, torch.tensor(0.7), atol=1e-5)
        assert torch.allclose(satisfaction3, torch.tensor(0.7), atol=1e-5)

    def test_can_compute_satisfaction(self) -> None:
        """Test can compute satisfaction values."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.Or(P(X), Q(X))

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.3),
        }

        logic_loss = compile_logic(expr, predicates)
        x = torch.randn(10, 5)
        satisfaction = logic_loss(x)

        # Product OR: 0.5 + 0.3 - 0.5*0.3 = 0.65
        expected = 0.5 + 0.3 - 0.5 * 0.3
        assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)

    def test_can_compute_loss(self) -> None:
        """Test can compute loss values."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = sp.Not(P(X))

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.4)}

        logic_loss = compile_logic(expr, predicates)
        x = torch.randn(10, 5)
        loss = logic_loss.loss(x)

        # NOT: 1 - 0.4 = 0.6
        # RProductTNorm recommends 'log': -log(0.6)
        satisfaction = 1.0 - 0.4
        expected_loss = -torch.log(torch.tensor(satisfaction))
        assert torch.allclose(loss, expected_loss, atol=1e-5)


class TestCompileLogicErrorHandling:
    """Test error handling in compile_logic()."""

    def test_invalid_mode_raises_error(self) -> None:
        """Test invalid mode raises ValueError."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)}

        # Invalid mode should raise ValueError
        with pytest.raises(ValueError, match="Unknown mode"):
            compile_logic(expr, predicates, mode="invalid_mode")

    def test_missing_predicates_raises_error(self) -> None:
        """Test missing predicates raises ValueError."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        # Missing predicate 'Q'
        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)}

        # Should raise error when trying to compile with missing predicate
        with pytest.raises(ValueError, match="Missing predicate"):
            compile_logic(expr, predicates)

    def test_clear_error_messages(self) -> None:
        """Test error messages are helpful."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)}

        # Test mode error message
        try:
            compile_logic(expr, predicates, mode="semantic")
        except ValueError as e:
            error_msg = str(e)
            # Should mention the invalid mode and expected values
            assert "semantic" in error_msg
            assert "tnorm" in error_msg


class TestCompileLogicWithComplexExpressions:
    """Test compile_logic with complex expressions."""

    def test_complex_nested_expression(self) -> None:
        """Test with deeply nested expression."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q, R, S = Symbol("P Q R S")
        # (P ∧ Q) → ((R ∨ ¬S) ↔ P)
        expr = sp.Implies(
            sp.And(P(X), Q(X)), sp.Equivalent(sp.Or(R(X), sp.Not(S(X))), P(X))
        )

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7),
            "R": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6),
            "S": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5),
        }

        logic_loss = compile_logic(expr, predicates)
        x = torch.randn(10, 5)
        satisfaction = logic_loss(x)

        # Should compute without error
        assert isinstance(satisfaction, torch.Tensor)
        assert satisfaction.shape == (10,)
        assert (satisfaction >= 0).all()
        assert (satisfaction <= 1).all()

    def test_multiple_predicates(self) -> None:
        """Test with many predicates."""
        # pylint: disable=invalid-name
        X = Variable("X")
        # Create 5 predicates
        P1, P2, P3, P4, P5 = Symbol("P1 P2 P3 P4 P5")
        # P1 ∧ P2 ∧ P3 ∧ P4 ∧ P5
        expr = sp.And(P1(X), P2(X), P3(X), P4(X), P5(X))

        # Fix lambda closure issue by using list comprehension with immediate evaluation
        pred_values = [0.9 - i * 0.1 for i in range(5)]
        predicates = {
            f"P{i+1}": Predicate(
                (lambda val: lambda x: torch.ones(x.shape[0]) * val)(pred_values[i])
            )
            for i in range(5)
        }

        logic_loss = compile_logic(expr, predicates)
        x = torch.randn(10, 5)
        satisfaction = logic_loss(x)

        # Product of all: 0.9 * 0.8 * 0.7 * 0.6 * 0.5
        expected = 0.9 * 0.8 * 0.7 * 0.6 * 0.5
        assert torch.allclose(satisfaction, torch.tensor(expected), atol=1e-5)

    def test_boolean_constants(self) -> None:
        """Test with sp.true and sp.false constants."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")

        # Test with sp.true
        expr_true = sp.And(P(X), sp.true)
        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        logic_loss_true = compile_logic(expr_true, predicates)
        x = torch.randn(10, 5)
        satisfaction_true = logic_loss_true(x)

        # P ∧ true = P
        assert torch.allclose(satisfaction_true, torch.tensor(0.7), atol=1e-5)

        # Test with sp.false
        expr_false = sp.Or(P(X), sp.false)
        logic_loss_false = compile_logic(expr_false, predicates)
        satisfaction_false = logic_loss_false(x)

        # P ∨ false = P
        assert torch.allclose(satisfaction_false, torch.tensor(0.7), atol=1e-5)


class TestCompileLogicInputHandling:
    """Test compile_logic handles different inputs."""

    def test_with_single_tensor_input(self) -> None:
        """Test with single tensor (shared across predicates)."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        # Both predicates use the same input tensor
        predicates = {
            "P": Predicate(lambda x: (x[:, 0] > 0).float()),
            "Q": Predicate(lambda x: (x[:, 1] > 0).float()),
        }

        logic_loss = compile_logic(expr, predicates)

        # Single tensor input shared across all predicates
        x = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]])
        satisfaction = logic_loss(x)

        # P and Q for each row:
        # Row 0: P=1, Q=1 -> AND = 1
        # Row 1: P=1, Q=0 -> AND = 0
        # Row 2: P=0, Q=1 -> AND = 0
        expected = torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(satisfaction, expected, atol=1e-5)

    def test_with_dict_input(self) -> None:
        """Test with dict of tensors (variable-based routing)."""
        X, Y = Variable("X Y")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(Y))

        # Predicates receive tensors (compiler extracts from dict by variable name)
        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda y: torch.ones(y.shape[0]) * 0.6),
        }

        logic_loss = compile_logic(expr, predicates)

        # Dict input with keys matching variable names
        x = {
            "X": torch.randn(10, 3),
            "Y": torch.randn(10, 5),
        }
        satisfaction = logic_loss(x)

        # Should handle dict inputs correctly
        # Product AND: 0.8 * 0.6 = 0.48
        assert torch.allclose(satisfaction, torch.tensor(0.48), atol=1e-5)

    def test_with_mixed_input_types(self) -> None:
        """Test can use both single tensor and dict inputs."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q, R = Symbol("P Q R")
        expr = sp.And(P(X), sp.And(Q(X), R(X)))

        # Predicates accept tensors (compiler handles input extraction)
        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.9),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "R": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7),
        }

        logic_loss = compile_logic(expr, predicates)

        # Can use single tensor (all predicates use same variable X)
        x_tensor = torch.randn(5, 3)
        satisfaction_tensor = logic_loss(x_tensor)

        # Or dict with variable name as key
        x_dict = {"X": torch.randn(5, 3)}
        satisfaction_dict = logic_loss(x_dict)

        # Both should compute correctly
        # Product: 0.9 * 0.8 * 0.7
        expected = 0.9 * 0.8 * 0.7
        assert torch.allclose(satisfaction_tensor, torch.tensor(expected), atol=1e-5)
        assert torch.allclose(satisfaction_dict, torch.tensor(expected), atol=1e-5)


class TestCompileLogicGradients:
    """Test gradient flow through compile_logic results."""

    def test_gradients_flow_to_models(self) -> None:
        """Test gradients reach model parameters."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        # Create a simple neural network predicate
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 1)

            def forward(self, x):
                return torch.sigmoid(self.linear(x).squeeze(-1))

        model = SimpleModel()
        predicates = {"P": Predicate(model, is_model=True)}

        logic_loss = compile_logic(expr, predicates)

        # Forward pass
        x = torch.randn(10, 5)
        loss = logic_loss.loss(x)

        # Backward pass
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)

    def test_can_train_with_compiled_logic(self) -> None:
        """Test can train models using compiled logic loss."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        # Create a trainable model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([0.5]))

            def forward(self, x):
                return torch.sigmoid(self.weight * x.sum(-1))

        model = SimpleModel()
        predicates = {"P": Predicate(model, is_model=True)}

        logic_loss = compile_logic(expr, predicates)

        # Get initial weight
        initial_weight = model.weight.item()

        # Training step
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.randn(10, 5)

        loss = logic_loss.loss(x)
        loss.backward()
        optimizer.step()

        # Weight should have changed
        final_weight = model.weight.item()
        assert initial_weight != final_weight

    def test_get_trainable_parameters(self) -> None:
        """Test can extract trainable parameters."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        # Create models
        model_p = nn.Linear(5, 1)
        model_q = nn.Linear(3, 1)

        predicates = {
            "P": Predicate(
                lambda x: torch.sigmoid(model_p(x).squeeze(-1)),
                is_model=False,  # Lambda, not a model
            ),
            "Q": Predicate(model_q, is_model=True),
        }

        logic_loss = compile_logic(expr, predicates)

        # Get trainable parameters
        params = logic_loss.get_trainable_parameters()

        # Should only get parameters from model_q (Q is marked as is_model)
        # Note: P uses a lambda so won't be detected as a model
        assert isinstance(params, list)
        # model_q has weight and bias
        assert len(params) == 2

    def test_compile_logic_with_non_callable_predicate(self):
        """compile_logic raises error for non-callable predicate."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        # Pass a non-callable value as predicate
        predicates = {"P": "not_a_callable"}

        with pytest.raises(TypeError, match="must be callable"):
            compile_logic(expr, predicates)
