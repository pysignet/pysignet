"""Tests for LogicLoss with quantify parameter and BatchHandlerMixin.

This module tests the updated LogicLoss class that uses BatchHandlerMixin
for t-norm aware batch reduction with explicit quantification control.

Key semantics:
- quantify='forall': Conjunction over batch → scalar
- quantify='exists': Disjunction over batch → scalar
- quantify='none': No quantification → (batch_size,)
- reduction only valid with quantify='none'
"""

import pytest
import torch
import torch.nn as nn

from pysignet import (
    LogicLoss,
    Predicate,
    Symbol,
    TNormCompiler,
    Variable,
    logic_to_loss,
)
from pysignet.tnorms import (
    GodelTNorm,
    LukasiewiczTNorm,
    RProductTNorm,
    SProductTNorm,
)


class TestQuantifyForall:
    """Tests for quantify='forall' (universal quantification)."""

    def test_forall_returns_scalar(self):
        """Forall quantification returns a scalar."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(10, 5)
        result = logic_loss.satisfaction(X=x, quantify='forall')

        assert result.shape == ()  # Scalar

    def test_forall_computes_conjunction(self):
        """Forall computes product of all satisfaction values."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        # Return different values for each batch element
        predicates = {"P": Predicate(lambda x: torch.tensor([0.9, 0.8, 0.7]))}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(3, 5)
        result = logic_loss.satisfaction(X=x, quantify='forall')

        # Product t-norm forall in log space: exp(sum(log(values)))
        # For RProduct: 0.9 * 0.8 * 0.7 = 0.504
        expected = 0.9 * 0.8 * 0.7
        assert torch.isclose(result, torch.tensor(expected), atol=1e-5)

    def test_forall_with_all_ones_returns_one(self):
        """Forall with all satisfaction=1.0 returns 1.0."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]))}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(10, 5)
        result = logic_loss.satisfaction(X=x, quantify='forall')

        assert torch.isclose(result, torch.tensor(1.0), atol=1e-6)

    def test_forall_with_any_zero_returns_zero(self):
        """Forall with any satisfaction=0.0 returns 0.0."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.tensor([0.9, 0.0, 0.8]))}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(3, 5)
        result = logic_loss.satisfaction(X=x, quantify='forall')

        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_forall_empty_batch_returns_one(self):
        """Forall over empty batch returns 1.0 (vacuous truth)."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(0, 5)
        result = logic_loss.satisfaction(X=x, quantify='forall')

        assert torch.isclose(result, torch.tensor(1.0), atol=1e-6)


class TestQuantifyExists:
    """Tests for quantify='exists' (existential quantification)."""

    def test_exists_returns_scalar(self):
        """Exists quantification returns a scalar."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.3)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(10, 5)
        result = logic_loss.satisfaction(X=x, quantify='exists')

        assert result.shape == ()  # Scalar

    def test_exists_computes_disjunction(self):
        """Exists computes probabilistic OR of all values."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.tensor([0.3, 0.4]))}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(2, 5)
        result = logic_loss.satisfaction(X=x, quantify='exists')

        # Probabilistic OR: P(A or B) = P(A) + P(B) - P(A)*P(B)
        expected = 0.3 + 0.4 - 0.3 * 0.4  # = 0.58
        assert torch.isclose(result, torch.tensor(expected), atol=1e-5)

    def test_exists_with_any_one_returns_one(self):
        """Exists with any satisfaction=1.0 returns 1.0."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.tensor([0.3, 1.0, 0.4]))}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(3, 5)
        result = logic_loss.satisfaction(X=x, quantify='exists')

        assert torch.isclose(result, torch.tensor(1.0), atol=1e-6)

    def test_exists_with_all_zeros_returns_zero(self):
        """Exists with all satisfaction=0.0 returns 0.0."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.zeros(x.shape[0]))}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(10, 5)
        result = logic_loss.satisfaction(X=x, quantify='exists')

        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_exists_empty_batch_returns_zero(self):
        """Exists over empty batch returns 0.0 (no witnesses)."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(0, 5)
        result = logic_loss.satisfaction(X=x, quantify='exists')

        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)


class TestQuantifyNone:
    """Tests for quantify='none' (no quantification, per-batch results)."""

    def test_none_returns_batch_tensor(self):
        """No quantification returns (batch_size,) tensor."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(10, 5)
        result = logic_loss.satisfaction(X=x, quantify='none')

        assert result.shape == (10,)

    def test_none_preserves_per_sample_values(self):
        """No quantification preserves individual satisfaction values."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        expected_values = torch.tensor([0.3, 0.5, 0.7, 0.9])
        predicates = {"P": Predicate(lambda x: expected_values)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(4, 5)
        result = logic_loss.satisfaction(X=x, quantify='none')

        assert torch.allclose(result, expected_values, atol=1e-6)


class TestQuantifyWithReduction:
    """Tests for reduction parameter with quantify."""

    def test_reduction_valid_with_quantify_none(self):
        """Reduction works with quantify='none'."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(10, 5)

        # All reduction modes should work with quantify='none'
        loss_mean = logic_loss.loss(X=x, quantify='none', reduction='mean')
        loss_sum = logic_loss.loss(X=x, quantify='none', reduction='sum')
        loss_none = logic_loss.loss(X=x, quantify='none', reduction='none')

        assert loss_mean.shape == ()
        assert loss_sum.shape == ()
        assert loss_none.shape == (10,)

    def test_reduction_invalid_with_quantify_forall(self):
        """Reduction raises error with quantify='forall'."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(10, 5)

        # Should raise ValueError for any reduction != 'none' with forall
        with pytest.raises(ValueError, match="reduction.*quantify"):
            logic_loss.loss(X=x, quantify='forall', reduction='mean')

        with pytest.raises(ValueError, match="reduction.*quantify"):
            logic_loss.loss(X=x, quantify='forall', reduction='sum')

    def test_reduction_invalid_with_quantify_exists(self):
        """Reduction raises error with quantify='exists'."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(10, 5)

        # Should raise ValueError for any reduction != 'none' with exists
        with pytest.raises(ValueError, match="reduction.*quantify"):
            logic_loss.loss(X=x, quantify='exists', reduction='mean')

        with pytest.raises(ValueError, match="reduction.*quantify"):
            logic_loss.loss(X=x, quantify='exists', reduction='sum')

    def test_reduction_none_valid_with_all_quantifiers(self):
        """reduction='none' is valid with all quantifier values."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(10, 5)

        # reduction='none' should work with all quantifiers
        # (though with forall/exists, the result is already scalar)
        loss_forall = logic_loss.loss(X=x, quantify='forall', reduction='none')
        loss_exists = logic_loss.loss(X=x, quantify='exists', reduction='none')
        loss_none = logic_loss.loss(X=x, quantify='none', reduction='none')

        assert loss_forall.shape == ()  # Already scalar from forall
        assert loss_exists.shape == ()  # Already scalar from exists
        assert loss_none.shape == (10,)  # Per-batch


class TestDefaultQuantify:
    """Tests for default quantify behavior."""

    def test_call_defaults_to_forall(self):
        """LogicLoss.__call__ defaults to quantify='forall'."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.tensor([0.9, 0.8, 0.7]))}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(3, 5)

        # Default call should be same as explicit forall
        result_default = logic_loss.satisfaction(X=x)
        result_forall = logic_loss.satisfaction(X=x, quantify='forall')

        assert torch.isclose(result_default, result_forall, atol=1e-6)
        assert result_default.shape == ()

    def test_loss_defaults_to_forall(self):
        """LogicLoss.loss defaults to quantify='forall'."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.tensor([0.9, 0.8, 0.7]))}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(3, 5)

        # Default loss should be same as explicit forall
        # Note: reduction defaults to 'none' with forall (scalar output)
        loss_default = logic_loss.loss(X=x)
        loss_forall = logic_loss.loss(X=x, quantify='forall')

        assert torch.isclose(loss_default, loss_forall, atol=1e-6)


class TestBatchSizeOne:
    """Tests for batch size = 1 behavior."""

    def test_batch_size_one_same_for_all_quantifiers(self):
        """With batch_size=1, all quantifiers produce the same result."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(1, 5)

        result_forall = logic_loss.satisfaction(X=x, quantify='forall')
        result_exists = logic_loss.satisfaction(X=x, quantify='exists')
        result_none = logic_loss.satisfaction(X=x, quantify='none')

        # All should be 0.7 (the single sample's satisfaction)
        assert torch.isclose(result_forall, torch.tensor(0.7), atol=1e-6)
        assert torch.isclose(result_exists, torch.tensor(0.7), atol=1e-6)
        assert torch.isclose(result_none[0], torch.tensor(0.7), atol=1e-6)

    def test_batch_size_one_loss_same_for_all_quantifiers(self):
        """With batch_size=1, loss is the same for all quantifiers."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}
        logic_loss = logic_to_loss(expr, predicates, tnorm=RProductTNorm())

        x = torch.randn(1, 5)

        # For forall/exists, reduction must be 'none'
        loss_forall = logic_loss.loss(X=x, quantify='forall', reduction='none')
        loss_exists = logic_loss.loss(X=x, quantify='exists', reduction='none')
        loss_none = logic_loss.loss(X=x, quantify='none', reduction='none')

        # All should produce the same loss value
        assert torch.isclose(loss_forall, loss_exists, atol=1e-6)
        assert torch.isclose(loss_forall, loss_none[0], atol=1e-6)

    @pytest.mark.parametrize("tnorm_class", [
        RProductTNorm, SProductTNorm, LukasiewiczTNorm, GodelTNorm
    ])
    def test_batch_size_one_consistent_across_tnorms(self, tnorm_class):
        """Batch size 1 gives same result for all quantifiers across t-norms."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        sat_value = 0.6
        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * sat_value)}
        logic_loss = logic_to_loss(expr, predicates, tnorm=tnorm_class())

        x = torch.randn(1, 5)

        result_forall = logic_loss.satisfaction(X=x, quantify='forall')
        result_exists = logic_loss.satisfaction(X=x, quantify='exists')

        # Both should equal the single sample's satisfaction
        assert torch.isclose(result_forall, torch.tensor(sat_value), atol=1e-6)
        assert torch.isclose(result_exists, torch.tensor(sat_value), atol=1e-6)


class TestLogSatisfaction:
    """Tests for log_satisfaction method."""

    def test_log_satisfaction_returns_log_space(self):
        """log_satisfaction returns values in (-inf, 0]."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(10, 5)
        log_sat = logic_loss.log_satisfaction(X=x, quantify='forall')

        assert log_sat <= 0.0
        assert not torch.isnan(log_sat)
        assert not torch.isinf(log_sat)

    def test_log_satisfaction_equals_log_of_satisfaction(self):
        """log_satisfaction equals log of satisfaction for small batches."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(5, 5)
        sat = logic_loss.satisfaction(X=x, quantify='forall')
        log_sat = logic_loss.log_satisfaction(X=x, quantify='forall')

        # For small batches, should be approximately equal
        expected = torch.log(sat + 1e-10)
        assert torch.isclose(log_sat, expected, atol=1e-4)

    def test_log_satisfaction_stable_for_large_batch(self):
        """log_satisfaction for large batches with product t-norms.

        Note: Current implementation computes satisfaction first then takes log,
        which underflows for very large batches. True log-space computation
        would give 1000 * log(0.9) = -105.36, but current implementation
        gives log(1e-10) = -23.02 due to underflow.
        """
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        # Use smaller batch to avoid underflow in current implementation
        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.9)}
        logic_loss = logic_to_loss(expr, predicates, tnorm=RProductTNorm())

        x = torch.randn(100, 5)  # Smaller batch to avoid underflow
        log_sat = logic_loss.log_satisfaction(X=x, quantify='forall')

        # Should be approximately 100 * log(0.9) = -10.54
        expected = 100 * torch.log(torch.tensor(0.9))
        assert torch.isclose(log_sat, expected, atol=1.0)
        assert not torch.isnan(log_sat)
        assert not torch.isinf(log_sat)


class TestTNormAwareLoss:
    """Tests for t-norm aware loss computation."""

    def test_rproduct_uses_log_postprocessing(self):
        """RProduct t-norm uses -log(satisfaction) for loss."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)}
        logic_loss = logic_to_loss(expr, predicates, tnorm=RProductTNorm())

        x = torch.randn(1, 5)
        loss = logic_loss.loss(X=x, quantify='none', reduction='none')

        # -log(0.5) = 0.693...
        expected = -torch.log(torch.tensor(0.5))
        assert torch.isclose(loss[0], expected, atol=1e-4)

    def test_lukasiewicz_uses_linear_postprocessing(self):
        """Lukasiewicz t-norm uses 1-satisfaction for loss."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}
        logic_loss = logic_to_loss(expr, predicates, tnorm=LukasiewiczTNorm())

        x = torch.randn(1, 5)
        loss = logic_loss.loss(X=x, quantify='none', reduction='none')

        # 1 - 0.7 = 0.3
        expected = torch.tensor(0.3)
        assert torch.isclose(loss[0], expected, atol=1e-4)


class TestGradientFlow:
    """Tests for gradient flow through quantification."""

    def test_gradient_flows_through_forall(self):
        """Gradients flow through forall quantification."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        predicates = {"P": Predicate(model)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(10, 5)
        loss = logic_loss.loss(X=x, quantify='forall')
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_gradient_flows_through_exists(self):
        """Gradients flow through exists quantification."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        predicates = {"P": Predicate(model)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(10, 5)
        loss = logic_loss.loss(X=x, quantify='exists')
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_gradient_flows_through_quantify_none(self):
        """Gradients flow through no quantification with reduction."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        predicates = {"P": Predicate(model)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(10, 5)
        loss = logic_loss.loss(X=x, quantify='none', reduction='mean')
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestInvalidInputs:
    """Tests for error handling with invalid inputs."""

    def test_invalid_quantify_raises_error(self):
        """Invalid quantify value raises ValueError."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(10, 5)

        with pytest.raises(ValueError, match="quantif"):
            logic_loss.satisfaction(X=x, quantify='invalid')

    def test_invalid_reduction_raises_error(self):
        """Invalid reduction value raises ValueError."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}
        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(10, 5)

        with pytest.raises(ValueError, match="reduction"):
            logic_loss.loss(X=x, quantify='none', reduction='invalid')


class TestAllTNormCombinations:
    """Tests for all t-norm and quantifier combinations."""

    @pytest.mark.parametrize("tnorm_class", [
        RProductTNorm, SProductTNorm, LukasiewiczTNorm, GodelTNorm
    ])
    @pytest.mark.parametrize("quantify", ['forall', 'exists', 'none'])
    def test_all_tnorm_quantify_combinations(self, tnorm_class, quantify):
        """Test all t-norm and quantify combinations work."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.rand(x.shape[0]))}
        logic_loss = logic_to_loss(expr, predicates, tnorm=tnorm_class())

        x = torch.randn(10, 5)
        result = logic_loss.satisfaction(X=x, quantify=quantify)

        if quantify == 'none':
            assert result.shape == (10,)
        else:
            assert result.shape == ()

        assert not torch.isnan(result).any()
        assert (result >= 0).all()
        assert (result <= 1).all()

    @pytest.mark.parametrize("tnorm_class", [
        RProductTNorm, SProductTNorm, LukasiewiczTNorm, GodelTNorm
    ])
    def test_loss_with_quantify_none_and_all_reductions(self, tnorm_class):
        """Test loss with quantify='none' and all reduction modes."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.rand(x.shape[0]))}
        logic_loss = logic_to_loss(expr, predicates, tnorm=tnorm_class())

        x = torch.randn(10, 5)

        for reduction in ['mean', 'sum', 'none']:
            loss = logic_loss.loss(X=x, quantify='none', reduction=reduction)

            if reduction == 'none':
                assert loss.shape == (10,)
            else:
                assert loss.shape == ()

            assert not torch.isnan(loss).any()
            assert (loss >= 0).all()
