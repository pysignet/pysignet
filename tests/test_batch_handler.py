"""Tests for BatchHandlerMixin.

This module tests the batch handling functionality that consolidates
batch reduction logic into a reusable mixin. The mixin delegates
conjunction and disjunction to the compiler that produced the
expression.

Tests cover:
- Forall quantification (conjunction via compiler)
- Exists quantification (disjunction via compiler)
- None quantification (no reduction)
- Reduction modes (mean/sum/none)
- Empty batch handling
- Gradient flow
- Edge cases
"""

import pytest
import torch

from pysignet.batch_handler import BatchHandlerMixin
from pysignet.compilation import TNormCompiler
from pysignet.tnorms import (
    RProductTNorm,
    LukasiewiczTNorm,
    GodelTNorm,
)


class ConcreteBatchHandler(BatchHandlerMixin):
    """Concrete implementation for testing the mixin."""

    def __init__(self, compiler):
        """Initialize with a compiler."""
        self._compiler = compiler


class TestForallQuantification:
    """Tests for forall (conjunction) quantification."""

    def test_forall_rproduct_returns_product_of_values(self):
        """Forall with RProduct computes product of all values."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([0.9, 0.8, 0.7])
        result = handler._reduce_batch(tensor, quantifier='forall')
        expected = 0.9 * 0.8 * 0.7
        assert torch.isclose(result, torch.tensor(expected), atol=1e-6)

    def test_forall_lukasiewicz_returns_clamped_sum(self):
        """Forall with Lukasiewicz computes max(0, sum - (n-1))."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=LukasiewiczTNorm())
        )
        tensor = torch.tensor([0.9, 0.8, 0.7])
        result = handler._reduce_batch(tensor, quantifier='forall')
        # max(0, 0.9 + 0.8 + 0.7 - 2) = max(0, 0.4) = 0.4
        expected = max(0.0, 0.9 + 0.8 + 0.7 - 2)
        assert torch.isclose(result, torch.tensor(expected), atol=1e-6)

    def test_forall_godel_returns_min_value(self):
        """Forall with Godel computes min of all values."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=GodelTNorm())
        )
        tensor = torch.tensor([0.9, 0.8, 0.7])
        result = handler._reduce_batch(tensor, quantifier='forall')
        expected = 0.7
        assert torch.isclose(result, torch.tensor(expected), atol=1e-6)

    def test_forall_with_single_value(self):
        """Forall with single value returns that value."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([0.75])
        result = handler._reduce_batch(tensor, quantifier='forall')
        assert torch.isclose(result, torch.tensor(0.75), atol=1e-6)

    def test_forall_with_zero(self):
        """Forall with any zero returns zero (product t-norm)."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([0.9, 0.0, 0.7])
        result = handler._reduce_batch(tensor, quantifier='forall')
        assert result == 0.0

    def test_forall_with_all_ones(self):
        """Forall with all ones returns one."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.ones(10)
        result = handler._reduce_batch(tensor, quantifier='forall')
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-6)


class TestExistsQuantification:
    """Tests for exists (disjunction) quantification."""

    def test_exists_rproduct_returns_probabilistic_or(self):
        """Exists with RProduct computes 1 - prod(1 - values)."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([0.3, 0.4])
        result = handler._reduce_batch(tensor, quantifier='exists')
        # 1 - (1 - 0.3)(1 - 0.4) = 1 - 0.7*0.6 = 1 - 0.42 = 0.58
        expected = 1.0 - (1.0 - 0.3) * (1.0 - 0.4)
        assert torch.isclose(result, torch.tensor(expected), atol=1e-6)

    def test_exists_lukasiewicz_returns_clamped_sum(self):
        """Exists with Lukasiewicz computes min(1, sum)."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=LukasiewiczTNorm())
        )
        tensor = torch.tensor([0.3, 0.4])
        result = handler._reduce_batch(tensor, quantifier='exists')
        # min(1, 0.3 + 0.4) = min(1, 0.7) = 0.7
        expected = min(1.0, 0.3 + 0.4)
        assert torch.isclose(result, torch.tensor(expected), atol=1e-6)

    def test_exists_godel_returns_max_value(self):
        """Exists with Godel computes max of all values."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=GodelTNorm())
        )
        tensor = torch.tensor([0.3, 0.4, 0.5])
        result = handler._reduce_batch(tensor, quantifier='exists')
        expected = 0.5
        assert torch.isclose(result, torch.tensor(expected), atol=1e-6)

    def test_exists_with_single_value(self):
        """Exists with single value returns that value."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([0.75])
        result = handler._reduce_batch(tensor, quantifier='exists')
        assert torch.isclose(result, torch.tensor(0.75), atol=1e-6)

    def test_exists_with_one(self):
        """Exists with any one returns one (product t-norm)."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([0.3, 1.0, 0.4])
        result = handler._reduce_batch(tensor, quantifier='exists')
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-6)

    def test_exists_with_all_zeros(self):
        """Exists with all zeros returns zero."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.zeros(10)
        result = handler._reduce_batch(tensor, quantifier='exists')
        assert result == 0.0

    def test_exists_three_values(self):
        """Exists with three values computes iterative OR."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([0.2, 0.3, 0.4])
        result = handler._reduce_batch(tensor, quantifier='exists')
        # 1 - (1-0.2)(1-0.3)(1-0.4) = 1 - 0.8*0.7*0.6 = 1 - 0.336 = 0.664
        expected = 1.0 - (1.0 - 0.2) * (1.0 - 0.3) * (1.0 - 0.4)
        assert torch.isclose(result, torch.tensor(expected), atol=1e-6)


class TestNoneQuantification:
    """Tests for 'none' quantification (no reduction)."""

    def test_none_returns_original_tensor(self):
        """Quantifier 'none' returns the original tensor unchanged."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([0.9, 0.8, 0.7])
        result = handler._reduce_batch(tensor, quantifier='none')
        assert torch.equal(result, tensor)


class TestApplyReduction:
    """Tests for loss reduction (mean/sum/none)."""

    def test_apply_reduction_mean(self):
        """Mean reduction computes mean of tensor."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = handler._apply_reduction(tensor, reduction='mean')
        assert torch.isclose(result, torch.tensor(2.0), atol=1e-6)

    def test_apply_reduction_sum(self):
        """Sum reduction computes sum of tensor."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = handler._apply_reduction(tensor, reduction='sum')
        assert torch.isclose(result, torch.tensor(6.0), atol=1e-6)

    def test_apply_reduction_none(self):
        """None reduction returns tensor unchanged."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = handler._apply_reduction(tensor, reduction='none')
        assert torch.equal(result, tensor)

    def test_apply_reduction_scalar_input(self):
        """Reduction on scalar returns scalar."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor(5.0)
        assert torch.isclose(
            handler._apply_reduction(tensor, 'mean'), torch.tensor(5.0)
        )
        assert torch.isclose(
            handler._apply_reduction(tensor, 'sum'), torch.tensor(5.0)
        )
        assert torch.isclose(
            handler._apply_reduction(tensor, 'none'), torch.tensor(5.0)
        )


class TestEmptyBatchHandling:
    """Tests for empty batch edge cases."""

    def test_forall_empty_batch_returns_one(self):
        """Forall over empty batch returns 1.0 (vacuous truth)."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([])
        result = handler._reduce_batch(tensor, quantifier='forall')
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-6)

    def test_exists_empty_batch_returns_zero(self):
        """Exists over empty batch returns 0.0 (no witnesses)."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([])
        result = handler._reduce_batch(tensor, quantifier='exists')
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)


class TestGradientFlow:
    """Tests for gradient flow through batch reduction."""

    def test_forall_gradient_flows(self):
        """Gradients flow through forall reduction."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([0.9, 0.8, 0.7], requires_grad=True)
        result = handler._reduce_batch(tensor, quantifier='forall')
        result.backward()
        assert tensor.grad is not None
        assert not torch.any(torch.isnan(tensor.grad))

    def test_exists_gradient_flows(self):
        """Gradients flow through exists reduction."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([0.3, 0.4, 0.5], requires_grad=True)
        result = handler._reduce_batch(tensor, quantifier='exists')
        result.backward()
        assert tensor.grad is not None
        assert not torch.any(torch.isnan(tensor.grad))

    def test_forall_lukasiewicz_gradient_flows(self):
        """Gradients flow through Lukasiewicz forall reduction."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=LukasiewiczTNorm())
        )
        tensor = torch.tensor([0.9, 0.8, 0.7], requires_grad=True)
        result = handler._reduce_batch(tensor, quantifier='forall')
        result.backward()
        assert tensor.grad is not None
        assert not torch.any(torch.isnan(tensor.grad))


class TestInvalidInputs:
    """Tests for error handling with invalid inputs."""

    def test_invalid_quantifier_raises_error(self):
        """Invalid quantifier raises ValueError."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([0.9, 0.8])
        with pytest.raises(ValueError, match="quantifier"):
            handler._reduce_batch(tensor, quantifier='invalid')

    def test_invalid_reduction_raises_error(self):
        """Invalid reduction raises ValueError."""
        handler = ConcreteBatchHandler(
            TNormCompiler(tnorm=RProductTNorm())
        )
        tensor = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="reduction"):
            handler._apply_reduction(tensor, reduction='invalid')
