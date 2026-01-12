"""Tests for nn.Module introspection and wrapping in compilation/module_utils.py.

This module tests smart nn.Module handling that:
1. Infers arity from output dimensionality (dim=1 → unary, dim>1 → binary)
2. Detects existing activations (Sigmoid/Softmax)
3. Wraps modules with appropriate callable signature
4. Auto-adds activation only if not already present
"""

import pytest
import torch
import torch.nn as nn

from pysignet.compilation.module_utils import (
    infer_module_arity,
    has_final_activation,
    wrap_module_as_predicate
)


class TestInferModuleArity:
    """Test arity inference from module output dimensionality."""

    def test_linear_output_1_is_unary(self):
        """Test Linear(*, 1) → arity 1 (unary)."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        assert infer_module_arity(model) == 1

    def test_linear_output_3_is_binary(self):
        """Test Linear(*, 3) → arity 2 (binary)."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )
        assert infer_module_arity(model) == 2

    def test_linear_output_10_is_binary(self):
        """Test Linear(*, 10) → arity 2 (binary)."""
        model = nn.Sequential(
            nn.Linear(10, 10)
        )
        assert infer_module_arity(model) == 2

    def test_sigmoid_is_unary(self):
        """Test Sigmoid() → arity 1."""
        model = nn.Sequential(
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        assert infer_module_arity(model) == 1

    def test_softmax_is_binary(self):
        """Test Softmax() → arity 2."""
        model = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )
        assert infer_module_arity(model) == 2

    def test_single_linear_1(self):
        """Test single Linear(*, 1) layer."""
        model = nn.Linear(10, 1)
        assert infer_module_arity(model) == 1

    def test_single_linear_5(self):
        """Test single Linear(*, 5) layer."""
        model = nn.Linear(10, 5)
        assert infer_module_arity(model) == 2

    def test_relu_output_raises_error(self):
        """Test ReLU as final layer raises error."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU()
        )

        with pytest.raises(ValueError) as exc_info:
            infer_module_arity(model)

        error_msg = str(exc_info.value)
        assert "ReLU" in error_msg
        assert "Linear" in error_msg or "Sigmoid" in error_msg or "Softmax" in error_msg

    def test_tanh_output_raises_error(self):
        """Test Tanh as final layer raises error."""
        model = nn.Sequential(
            nn.Linear(10, 3),
            nn.Tanh()
        )

        with pytest.raises(ValueError) as exc_info:
            infer_module_arity(model)

        assert "Tanh" in str(exc_info.value)


class TestHasFinalActivation:
    """Test detection of final activation layers."""

    def test_linear_sigmoid_has_activation(self):
        """Test Linear → Sigmoid is detected."""
        model = nn.Sequential(
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        assert has_final_activation(model) is True

    def test_linear_softmax_has_activation(self):
        """Test Linear → Softmax is detected."""
        model = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )
        assert has_final_activation(model) is True

    def test_linear_no_activation(self):
        """Test Linear without activation."""
        model = nn.Sequential(
            nn.Linear(10, 1)
        )
        assert has_final_activation(model) is False

    def test_linear_relu_no_activation(self):
        """Test Linear → ReLU is not considered activation."""
        model = nn.Sequential(
            nn.Linear(10, 1),
            nn.ReLU()
        )
        assert has_final_activation(model) is False

    def test_single_sigmoid(self):
        """Test single Sigmoid layer."""
        model = nn.Sigmoid()
        assert has_final_activation(model) is True

    def test_single_softmax(self):
        """Test single Softmax layer."""
        model = nn.Softmax(dim=-1)
        assert has_final_activation(model) is True


class TestWrapModuleUnary:
    """Test wrapping modules as unary predicates."""

    def test_linear_without_activation_adds_sigmoid(self):
        """Test Linear(*, 1) gets sigmoid added."""
        model = nn.Sequential(nn.Linear(10, 1))
        wrapper = wrap_module_as_predicate(model, arity=1)

        x = torch.randn(32, 10)
        output = wrapper(x)

        # Should be activated (all values in [0, 1])
        assert output.shape == (32,)
        assert torch.all((output >= 0) & (output <= 1))

    def test_linear_with_sigmoid_no_double_activation(self):
        """Test Linear → Sigmoid doesn't get double-activated."""
        model = nn.Sequential(
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        wrapper = wrap_module_as_predicate(model, arity=1)

        x = torch.randn(32, 10)

        # Direct module call
        direct_output = model(x).squeeze(-1)

        # Wrapper call
        wrapped_output = wrapper(x)

        # Should be identical (no double sigmoid)
        assert torch.allclose(direct_output, wrapped_output, atol=1e-6)

    def test_unary_wrapper_shape(self):
        """Test unary wrapper returns (batch,) shape."""
        model = nn.Linear(10, 1)
        wrapper = wrap_module_as_predicate(model, arity=1)

        x = torch.randn(32, 10)
        output = wrapper(x)

        assert output.shape == (32,)
        assert output.dim() == 1

    def test_unary_wrapper_gradient_flow(self):
        """Test gradients flow through unary wrapper."""
        model = nn.Linear(10, 1)
        wrapper = wrap_module_as_predicate(model, arity=1)

        x = torch.randn(32, 10, requires_grad=True)
        output = wrapper(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert model.weight.grad is not None


class TestWrapModuleBinary:
    """Test wrapping modules as binary predicates."""

    def test_linear_without_activation_adds_softmax(self):
        """Test Linear(*, N) gets softmax added."""
        model = nn.Sequential(nn.Linear(10, 3))
        wrapper = wrap_module_as_predicate(model, arity=2)

        x = torch.randn(32, 10)
        y = 1  # Select class 1
        output = wrapper(x, y)

        # Should be activated
        assert output.shape == (32,)
        assert torch.all((output >= 0) & (output <= 1))

    def test_linear_with_softmax_no_double_activation(self):
        """Test Linear → Softmax doesn't get double-activated."""
        model = nn.Sequential(
            nn.Linear(10, 3),
            nn.Softmax(dim=-1)
        )
        wrapper = wrap_module_as_predicate(model, arity=2)

        x = torch.randn(32, 10)
        y = 1

        # Direct module call
        direct_output = model(x)[:, y]

        # Wrapper call
        wrapped_output = wrapper(x, y)

        # Should be identical (no double softmax)
        assert torch.allclose(direct_output, wrapped_output, atol=1e-6)

    def test_binary_wrapper_all_classes(self):
        """Test binary wrapper works for all classes."""
        model = nn.Linear(10, 5)
        wrapper = wrap_module_as_predicate(model, arity=2)

        x = torch.randn(32, 10)

        for class_idx in range(5):
            output = wrapper(x, class_idx)
            assert output.shape == (32,)
            assert torch.all((output >= 0) & (output <= 1))

    def test_binary_wrapper_gradient_flow(self):
        """Test gradients flow through binary wrapper."""
        model = nn.Linear(10, 3)
        wrapper = wrap_module_as_predicate(model, arity=2)

        x = torch.randn(32, 10, requires_grad=True)
        output = wrapper(x, 1)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert model.weight.grad is not None


class TestCustomModules:
    """Test with custom nn.Module subclasses."""

    def test_custom_unary_module(self):
        """Test custom module with single output."""
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 1)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = CustomModel()
        assert infer_module_arity(model) == 1
        assert has_final_activation(model) is False

        wrapper = wrap_module_as_predicate(model, arity=1)
        x = torch.randn(16, 10)
        output = wrapper(x)
        assert output.shape == (16,)

    def test_custom_binary_module(self):
        """Test custom module with multiple outputs."""
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 3)

            def forward(self, x):
                return self.fc(x)

        model = CustomModel()
        assert infer_module_arity(model) == 2
        assert has_final_activation(model) is False

        wrapper = wrap_module_as_predicate(model, arity=2)
        x = torch.randn(16, 10)
        output = wrapper(x, 1)
        assert output.shape == (16,)


class TestNonPollutingWrapper:
    """Test that wrapper doesn't pollute original module."""

    def test_module_outputs_unchanged(self):
        """Test original module outputs are not affected."""
        # Create module used in larger graph
        module1 = nn.Sequential(nn.Linear(10, 3))

        # Wrap for predicate use
        wrapper = wrap_module_as_predicate(module1, arity=2)

        # Original module should still output raw logits
        x = torch.randn(16, 10)
        logits = module1(x)
        assert logits.shape == (16, 3)

        # Values should be unbounded (not probabilities)
        assert torch.any(logits < 0) or torch.any(logits > 1)

        # Wrapper applies softmax only for predicate evaluation
        prob = wrapper(x, 0)
        assert torch.all((prob >= 0) & (prob <= 1))


class TestArityMismatch:
    """Test validation of arity parameter."""

    def test_unary_module_with_binary_arity_error(self):
        """Test Linear(*, 1) used as binary raises error."""
        model = nn.Linear(10, 1)

        with pytest.raises(ValueError) as exc_info:
            wrap_module_as_predicate(model, arity=2)

        error_msg = str(exc_info.value)
        assert "unary" in error_msg.lower() or "1" in error_msg
        assert "binary" in error_msg.lower() or "2" in error_msg

    def test_binary_module_with_unary_arity_error(self):
        """Test Linear(*, 3) used as unary raises error."""
        model = nn.Linear(10, 3)

        with pytest.raises(ValueError) as exc_info:
            wrap_module_as_predicate(model, arity=1)

        error_msg = str(exc_info.value)
        assert "binary" in error_msg.lower() or "2" in error_msg
        assert "unary" in error_msg.lower() or "1" in error_msg


class TestErrorMessages:
    """Test that error messages are helpful."""

    def test_unsupported_layer_suggests_alternatives(self):
        """Test error message suggests supported layer types."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Tanh())

        with pytest.raises(ValueError) as exc_info:
            infer_module_arity(model)

        error_msg = str(exc_info.value)
        assert "Linear" in error_msg
        assert "Sigmoid" in error_msg
        assert "Softmax" in error_msg

    def test_unsupported_layer_suggests_wrapper(self):
        """Test error message suggests explicit wrapper."""
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())

        with pytest.raises(ValueError) as exc_info:
            infer_module_arity(model)

        error_msg = str(exc_info.value)
        assert "lambda" in error_msg.lower() or "wrapper" in error_msg.lower()
