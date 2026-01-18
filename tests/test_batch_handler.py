"""Tests for BatchHandlerMixin.

This module tests the batch handling functionality that consolidates
batch reduction logic into a reusable mixin. The mixin provides t-norm
aware computation spaces for numerical stability.

Tests cover:
- T-norm detection (product vs non-product)
- Forall quantification (linear and log space)
- Exists quantification (linear space)
- Reduction modes (mean/sum/none)
- Numerical stability for large batches
- Empty batch handling
- Edge cases
"""

import pytest
import torch

from pysignet.batch_handler import BatchHandlerMixin
from pysignet.tnorms import (
    RProductTNorm,
    SProductTNorm,
    LukasiewiczTNorm,
    GodelTNorm,
)


class ConcreteBatchHandler(BatchHandlerMixin):
    """Concrete implementation for testing the mixin."""

    def __init__(self, tnorm):
        """Initialize with a t-norm."""
        self._tnorm = tnorm


class TestTNormDetection:
    """Tests for t-norm type detection."""

    def test_detects_rproduct_as_product_tnorm(self):
        """RProductTNorm should be detected as a product t-norm."""
        handler = ConcreteBatchHandler(RProductTNorm())
        assert handler._is_product_tnorm() is True

    def test_detects_sproduct_as_product_tnorm(self):
        """SProductTNorm should be detected as a product t-norm."""
        handler = ConcreteBatchHandler(SProductTNorm())
        assert handler._is_product_tnorm() is True

    def test_detects_lukasiewicz_as_non_product_tnorm(self):
        """LukasiewiczTNorm should not be detected as a product t-norm."""
        handler = ConcreteBatchHandler(LukasiewiczTNorm())
        assert handler._is_product_tnorm() is False

    def test_detects_godel_as_non_product_tnorm(self):
        """GodelTNorm should not be detected as a product t-norm."""
        handler = ConcreteBatchHandler(GodelTNorm())
        assert handler._is_product_tnorm() is False


class TestForallLinearSpace:
    """Tests for forall quantification in linear space."""

    def test_forall_linear_returns_product_of_values(self):
        """Forall in linear space computes product of all values."""
        handler = ConcreteBatchHandler(LukasiewiczTNorm())
        tensor = torch.tensor([0.9, 0.8, 0.7])
        result = handler._reduce_forall_linear(tensor)
        expected = 0.9 * 0.8 * 0.7
        assert torch.isclose(result, torch.tensor(expected), atol=1e-6)

    def test_forall_linear_with_single_value(self):
        """Forall with single value returns that value."""
        handler = ConcreteBatchHandler(LukasiewiczTNorm())
        tensor = torch.tensor([0.75])
        result = handler._reduce_forall_linear(tensor)
        assert torch.isclose(result, torch.tensor(0.75), atol=1e-6)

    def test_forall_linear_with_zero(self):
        """Forall with any zero returns zero."""
        handler = ConcreteBatchHandler(LukasiewiczTNorm())
        tensor = torch.tensor([0.9, 0.0, 0.7])
        result = handler._reduce_forall_linear(tensor)
        assert result == 0.0

    def test_forall_linear_with_all_ones(self):
        """Forall with all ones returns one."""
        handler = ConcreteBatchHandler(LukasiewiczTNorm())
        tensor = torch.ones(10)
        result = handler._reduce_forall_linear(tensor)
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-6)


class TestForallLogSpace:
    """Tests for forall quantification in log space."""

    def test_forall_log_returns_sum_of_log_values(self):
        """Forall in log space computes sum of log values."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([0.9, 0.8, 0.7])
        result = handler._reduce_forall_log(tensor)
        expected = torch.log(torch.tensor(0.9)) + torch.log(
            torch.tensor(0.8)
        ) + torch.log(torch.tensor(0.7))
        assert torch.isclose(result, expected, atol=1e-6)

    def test_forall_log_with_single_value(self):
        """Forall log with single value returns log of that value."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([0.75])
        result = handler._reduce_forall_log(tensor)
        expected = torch.log(torch.tensor(0.75))
        assert torch.isclose(result, expected, atol=1e-6)

    def test_forall_log_with_all_ones(self):
        """Forall log with all ones returns zero (log(1) = 0)."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.ones(10)
        result = handler._reduce_forall_log(tensor)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_forall_log_handles_small_values(self):
        """Forall log handles small values without NaN."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([1e-10, 0.5, 0.9])
        result = handler._reduce_forall_log(tensor)
        assert not torch.isnan(result)
        assert not torch.isinf(result)


class TestExistsLinearSpace:
    """Tests for exists quantification in linear space."""

    def test_exists_linear_returns_probabilistic_or(self):
        """Exists in linear space computes probabilistic OR."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([0.3, 0.4])
        result = handler._reduce_exists_linear(tensor)
        # P(A or B) = P(A) + P(B) - P(A)P(B) = 0.3 + 0.4 - 0.12 = 0.58
        expected = 0.3 + 0.4 - 0.3 * 0.4
        assert torch.isclose(result, torch.tensor(expected), atol=1e-6)

    def test_exists_linear_with_single_value(self):
        """Exists with single value returns that value."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([0.75])
        result = handler._reduce_exists_linear(tensor)
        assert torch.isclose(result, torch.tensor(0.75), atol=1e-6)

    def test_exists_linear_with_one(self):
        """Exists with any one returns one."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([0.3, 1.0, 0.4])
        result = handler._reduce_exists_linear(tensor)
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-6)

    def test_exists_linear_with_all_zeros(self):
        """Exists with all zeros returns zero."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.zeros(10)
        result = handler._reduce_exists_linear(tensor)
        assert result == 0.0

    def test_exists_three_values(self):
        """Exists with three values computes iterative OR."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([0.2, 0.3, 0.4])
        result = handler._reduce_exists_linear(tensor)
        # (0.2 OR 0.3) = 0.2 + 0.3 - 0.06 = 0.44
        # (0.44 OR 0.4) = 0.44 + 0.4 - 0.176 = 0.664
        expected = 0.664
        assert torch.isclose(result, torch.tensor(expected), atol=1e-6)


class TestReduceBatch:
    """Tests for the main _reduce_batch method."""

    def test_reduce_batch_forall_auto_selects_log_for_product_tnorm(self):
        """Auto space selects log for product t-norms with forall."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([0.9, 0.8, 0.7])
        result = handler._reduce_batch(tensor, quantifier='forall', space='auto')
        # Should use log space internally but return in requested format
        expected_log = handler._reduce_forall_log(tensor)
        assert torch.isclose(result, expected_log, atol=1e-6)

    def test_reduce_batch_forall_auto_selects_linear_for_lukasiewicz(self):
        """Auto space selects linear for Lukasiewicz with forall."""
        handler = ConcreteBatchHandler(LukasiewiczTNorm())
        tensor = torch.tensor([0.9, 0.8, 0.7])
        result = handler._reduce_batch(tensor, quantifier='forall', space='auto')
        expected = handler._reduce_forall_linear(tensor)
        assert torch.isclose(result, expected, atol=1e-6)

    def test_reduce_batch_forall_linear_explicit(self):
        """Explicit linear space for forall."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([0.9, 0.8, 0.7])
        result = handler._reduce_batch(tensor, quantifier='forall', space='linear')
        expected = handler._reduce_forall_linear(tensor)
        assert torch.isclose(result, expected, atol=1e-6)

    def test_reduce_batch_forall_log_explicit(self):
        """Explicit log space for forall."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([0.9, 0.8, 0.7])
        result = handler._reduce_batch(tensor, quantifier='forall', space='log')
        expected = handler._reduce_forall_log(tensor)
        assert torch.isclose(result, expected, atol=1e-6)

    def test_reduce_batch_exists(self):
        """Exists quantification uses disjunction."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([0.3, 0.4])
        result = handler._reduce_batch(tensor, quantifier='exists', space='linear')
        expected = handler._reduce_exists_linear(tensor)
        assert torch.isclose(result, expected, atol=1e-6)

    def test_reduce_batch_none_returns_original(self):
        """Quantifier 'none' returns the original tensor unchanged."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([0.9, 0.8, 0.7])
        result = handler._reduce_batch(tensor, quantifier='none', space='linear')
        assert torch.equal(result, tensor)


class TestApplyReduction:
    """Tests for loss reduction (mean/sum/none)."""

    def test_apply_reduction_mean(self):
        """Mean reduction computes mean of tensor."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = handler._apply_reduction(tensor, reduction='mean')
        assert torch.isclose(result, torch.tensor(2.0), atol=1e-6)

    def test_apply_reduction_sum(self):
        """Sum reduction computes sum of tensor."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = handler._apply_reduction(tensor, reduction='sum')
        assert torch.isclose(result, torch.tensor(6.0), atol=1e-6)

    def test_apply_reduction_none(self):
        """None reduction returns tensor unchanged."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = handler._apply_reduction(tensor, reduction='none')
        assert torch.equal(result, tensor)

    def test_apply_reduction_scalar_input(self):
        """Reduction on scalar returns scalar."""
        handler = ConcreteBatchHandler(RProductTNorm())
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
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([])
        result = handler._reduce_batch(tensor, quantifier='forall', space='linear')
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-6)

    def test_exists_empty_batch_returns_zero(self):
        """Exists over empty batch returns 0.0 (no witnesses)."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([])
        result = handler._reduce_batch(tensor, quantifier='exists', space='linear')
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_forall_log_empty_batch_returns_zero(self):
        """Forall log over empty batch returns 0.0 (log(1) = 0)."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([])
        result = handler._reduce_batch(tensor, quantifier='forall', space='log')
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)


class TestNumericalStability:
    """Tests for numerical stability with large batches."""

    def test_forall_linear_underflows_with_large_batch(self):
        """Linear space underflows to zero with large batch of small values."""
        handler = ConcreteBatchHandler(RProductTNorm())
        # 0.9^1000 is extremely small (underflows to 0)
        tensor = torch.full((1000,), 0.9)
        result = handler._reduce_forall_linear(tensor)
        # This should underflow to 0 or very small number
        assert result < 1e-30 or result == 0.0

    def test_forall_log_stable_with_large_batch(self):
        """Log space remains stable with large batch."""
        handler = ConcreteBatchHandler(RProductTNorm())
        # 0.9^1000 in log space: 1000 * log(0.9) ≈ -105.36
        tensor = torch.full((1000,), 0.9)
        result = handler._reduce_forall_log(tensor)
        expected = 1000 * torch.log(torch.tensor(0.9))
        assert not torch.isnan(result)
        assert not torch.isinf(result)
        assert torch.isclose(result, expected, atol=1e-4)

    def test_product_tnorm_auto_uses_log_space(self):
        """Product t-norms should use log space by default for stability."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.full((1000,), 0.9)
        result = handler._reduce_batch(tensor, quantifier='forall', space='auto')
        # Should be in log space, so not underflowed
        expected = 1000 * torch.log(torch.tensor(0.9))
        assert torch.isclose(result, expected, atol=1e-4)


class TestGradientFlow:
    """Tests for gradient flow through batch reduction."""

    def test_forall_linear_gradient_flows(self):
        """Gradients flow through forall linear reduction."""
        handler = ConcreteBatchHandler(LukasiewiczTNorm())
        tensor = torch.tensor([0.9, 0.8, 0.7], requires_grad=True)
        result = handler._reduce_forall_linear(tensor)
        result.backward()
        assert tensor.grad is not None
        assert not torch.any(torch.isnan(tensor.grad))

    def test_forall_log_gradient_flows(self):
        """Gradients flow through forall log reduction."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([0.9, 0.8, 0.7], requires_grad=True)
        result = handler._reduce_forall_log(tensor)
        result.backward()
        assert tensor.grad is not None
        assert not torch.any(torch.isnan(tensor.grad))

    def test_exists_linear_gradient_flows(self):
        """Gradients flow through exists linear reduction."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([0.3, 0.4, 0.5], requires_grad=True)
        result = handler._reduce_exists_linear(tensor)
        result.backward()
        assert tensor.grad is not None
        assert not torch.any(torch.isnan(tensor.grad))


class TestInvalidInputs:
    """Tests for error handling with invalid inputs."""

    def test_invalid_quantifier_raises_error(self):
        """Invalid quantifier raises ValueError."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([0.9, 0.8])
        with pytest.raises(ValueError, match="quantifier"):
            handler._reduce_batch(tensor, quantifier='invalid', space='linear')

    def test_invalid_space_raises_error(self):
        """Invalid space raises ValueError."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([0.9, 0.8])
        with pytest.raises(ValueError, match="space"):
            handler._reduce_batch(tensor, quantifier='forall', space='invalid')

    def test_invalid_reduction_raises_error(self):
        """Invalid reduction raises ValueError."""
        handler = ConcreteBatchHandler(RProductTNorm())
        tensor = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="reduction"):
            handler._apply_reduction(tensor, reduction='invalid')
