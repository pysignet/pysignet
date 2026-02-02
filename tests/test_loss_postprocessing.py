"""Tests for configurable loss post-processing.

This module tests t-norm-specific loss post-processing strategies:
- R-Product/S-Product: -log(satisfaction)
- Lukasiewicz: -satisfaction (linear)
- Gödel: TBD
- Custom post-processing functions
"""

import pytest
import torch
import torch.nn as nn
import sympy as sp

from pysignet import (
    Symbol,
    Variable,
    logic_to_loss,
    Predicate,
    RProductTNorm,
    SProductTNorm,
    LukasiewiczTNorm,
    GodelTNorm,
)


class TestRProductPostProcessing:
    """Test R-Product t-norm uses -log(satisfaction) post-processing."""

    def test_r_product_log_loss(self) -> None:
        """Test R-Product uses -log(satisfaction) for loss."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate( lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate( lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        compiler = logic_to_loss(expr, predicates, tnorm=RProductTNorm())
        x = torch.randn(1, 5)

        # Get satisfaction
        satisfaction = compiler(X=x)
        # Expected: 0.8 * 0.6 = 0.48

        # Get loss
        loss = compiler.loss(X=x, reduction="none")

        # Should be -log(satisfaction)
        expected_loss = -torch.log(satisfaction)
        assert torch.allclose(loss, expected_loss, atol=1e-5)

    def test_r_product_log_loss_with_mean_reduction(self) -> None:
        """Test R-Product log loss with mean reduction."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {
            "P": Predicate( lambda x: torch.tensor([0.5, 0.8, 0.2]))
        }

        compiler = logic_to_loss(expr, predicates, tnorm=RProductTNorm())
        x = torch.randn(1, 5)

        # Use quantify='none' to get per-batch losses, then mean reduction
        loss = compiler.loss(X=x, quantify='none', reduction="mean")
        satisfaction = compiler(X=x, quantify='none')

        expected_loss = -torch.log(satisfaction + 1e-10).mean()
        assert torch.allclose(loss, expected_loss, atol=1e-5)

    def test_r_product_log_loss_numerical_stability(self) -> None:
        """Test R-Product handles near-zero satisfaction without NaN."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        # Very low satisfaction values
        predicates = {
            "P": Predicate( lambda x: torch.tensor([1e-10, 1e-8, 0.01]))
        }

        compiler = logic_to_loss(expr, predicates, tnorm=RProductTNorm())
        x = torch.randn(1, 5)

        loss = compiler.loss(X=x, reduction="none")

        # Should not be NaN or Inf
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()
        # Should be positive (negative log of small number is positive)
        assert (loss > 0).all()


class TestSProductPostProcessing:
    """Test S-Product t-norm uses -log(satisfaction) post-processing."""

    def test_s_product_log_loss(self) -> None:
        """Test S-Product uses -log(satisfaction) for loss."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.Implies(P(X), Q(X))

        predicates = {
            "P": Predicate( lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate( lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        compiler = logic_to_loss(expr, predicates, tnorm=SProductTNorm())
        x = torch.randn(1, 5)

        satisfaction = compiler(X=x)
        loss = compiler.loss(X=x, reduction="none")

        # Should be -log(satisfaction)
        expected_loss = -torch.log(satisfaction)
        assert torch.allclose(loss, expected_loss, atol=1e-5)


class TestLukasiewiczPostProcessing:
    """Test Lukasiewicz t-norm uses -satisfaction (linear) post-processing."""

    def test_lukasiewicz_linear_loss(self) -> None:
        """Test Lukasiewicz uses 1 - satisfaction for loss."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate( lambda x: torch.ones(x.shape[0]) * 0.7),
            "Q": Predicate( lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        compiler = logic_to_loss(
            expr, predicates, tnorm=LukasiewiczTNorm()
        )
        x = torch.randn(1, 5)

        satisfaction = compiler(X=x)
        loss = compiler.loss(X=x, reduction="none")

        # Should be 1 - satisfaction
        expected_loss = 1.0 - satisfaction
        assert torch.allclose(loss, expected_loss, atol=1e-5)

    def test_lukasiewicz_linear_loss_with_sum_reduction(self) -> None:
        """Test Lukasiewicz linear loss with sum reduction."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {
            "P": Predicate( lambda x: torch.tensor([0.3, 0.7, 0.9]))
        }

        compiler = logic_to_loss(
            expr, predicates, tnorm=LukasiewiczTNorm()
        )
        x = torch.randn(1, 5)

        # Use quantify='none' to get per-batch losses, then sum reduction
        loss = compiler.loss(X=x, quantify='none', reduction="sum")
        satisfaction = compiler(X=x, quantify='none')

        expected_loss = (1.0 - satisfaction).sum()
        assert torch.allclose(loss, expected_loss, atol=1e-5)


class TestGodelPostProcessing:
    """Test Gödel t-norm post-processing (semantics TBD)."""

    def test_godel_default_loss(self) -> None:
        """Test Gödel t-norm default loss computation.

        NOTE: The semantics for Gödel post-processing are still TBD.
        This test documents current behavior (1 - satisfaction).
        """
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate( lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate( lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        compiler = logic_to_loss(expr, predicates, tnorm=GodelTNorm())
        x = torch.randn(1, 5)

        satisfaction = compiler(X=x)
        loss = compiler.loss(X=x, reduction="none")

        # Gödel uses linear post-processing: loss = 1 - satisfaction
        expected_loss = 1.0 - satisfaction
        assert torch.allclose(loss, expected_loss, atol=1e-5)


class TestCustomPostProcessing:
    """Test custom post-processing functions."""

    def test_custom_postprocessing_callable(self) -> None:
        """Test custom post-processing function."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate( lambda x: torch.tensor([0.8, 0.6]))}

        # Custom post-processing: square of (1 - satisfaction)
        def custom_loss(satisfaction: torch.Tensor) -> torch.Tensor:
            return (1.0 - satisfaction) ** 2

        compiler = logic_to_loss(expr, predicates)
        x = torch.randn(1, 5)

        # Use custom post-processing with quantify='none' for per-batch losses
        loss = compiler.loss(
            X=x, quantify='none', reduction="none", post_processing=custom_loss
        )

        # Should be (1 - satisfaction)^2
        expected_loss = torch.tensor([0.04, 0.16])
        assert torch.allclose(loss, expected_loss, atol=1e-5)

    def test_custom_postprocessing_with_gradients(self) -> None:
        """Test custom post-processing preserves gradients."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        model = nn.Linear(5, 1)
        predicates = {
            "P": Predicate(
                lambda x: torch.sigmoid(model(x).squeeze())
            )
        }

        # Custom: absolute difference from 0.5
        def custom_loss(satisfaction: torch.Tensor) -> torch.Tensor:
            return torch.abs(satisfaction - 0.5)

        compiler = logic_to_loss(expr, predicates)
        batch_size = 10
        x = torch.randn(batch_size, 5)

        # Use quantify='none' with reduction='mean' for per-batch losses
        loss = compiler.loss(
            X=x, quantify='none', reduction="mean", post_processing=custom_loss
        )
        loss.backward()

        # Gradients should flow
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestPostProcessingDifferentiability:
    """Test that post-processing preserves differentiability."""

    def test_log_postprocessing_gradients_flow(self) -> None:
        """Test gradients flow through -log post-processing."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        model = nn.Linear(5, 1)
        predicates = {
            "P": Predicate(
                lambda x: torch.sigmoid(model(x).squeeze())
            )
        }

        compiler = logic_to_loss(expr, predicates, tnorm=RProductTNorm())
        batch_size = 10
        x = torch.randn(batch_size, 5)

        # Use quantify='none' with reduction='mean' for per-batch losses
        loss = compiler.loss(X=x, quantify='none', reduction="mean")
        loss.backward()

        # Gradients should exist and not be NaN
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_linear_postprocessing_gradients_flow(self) -> None:
        """Test gradients flow through linear (1-x) post-processing."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        model = nn.Linear(5, 1)
        predicates = {
            "P": Predicate(
                lambda x: torch.sigmoid(model(x).squeeze())
            )
        }

        compiler = logic_to_loss(
            expr, predicates, tnorm=LukasiewiczTNorm()
        )
        batch_size = 10
        x = torch.randn(batch_size, 5)

        # Use quantify='none' with reduction='mean' for per-batch losses
        loss = compiler.loss(X=x, quantify='none', reduction="mean")
        loss.backward()

        # Gradients should exist and not be NaN
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestBoundaryValues:
    """Test post-processing with boundary satisfaction values."""

    def test_perfect_satisfaction(self) -> None:
        """Test loss when satisfaction = 1.0."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {
            "P": Predicate( lambda x: torch.ones(x.shape[0]))
        }

        # Test R-Product (log)
        compiler_log = logic_to_loss(expr, predicates, tnorm=RProductTNorm())
        x = torch.randn(1, 3)
        loss_log = compiler_log.loss(X=x, reduction="none")
        # -log(1) = 0
        assert torch.allclose(loss_log, torch.zeros(5), atol=1e-5)

        # Test Lukasiewicz (linear)
        compiler_linear = logic_to_loss(
            expr, predicates, tnorm=LukasiewiczTNorm()
        )
        loss_linear = compiler_linear.loss(X=x, reduction="none")
        # 1 - 1 = 0
        assert torch.allclose(loss_linear, torch.zeros(5), atol=1e-5)

    def test_zero_satisfaction(self) -> None:
        """Test loss when satisfaction = 0.0 (or very close)."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        # Use very small but non-zero to avoid log(0)
        predicates = {
            "P": Predicate( lambda x: torch.ones(x.shape[0]) * 1e-10)
        }

        # Test R-Product (log)
        compiler_log = logic_to_loss(expr, predicates, tnorm=RProductTNorm())
        x = torch.randn(1, 3)
        loss_log = compiler_log.loss(X=x, reduction="none")
        # -log(~0) should be large but finite
        assert (loss_log > 10).all()  # Should be large
        assert not torch.isinf(loss_log).any()

        # Test Lukasiewicz (linear)
        compiler_linear = logic_to_loss(
            expr, predicates, tnorm=LukasiewiczTNorm()
        )
        loss_linear = compiler_linear.loss(X=x, reduction="none")
        # 1 - ~0 ≈ 1
        assert torch.allclose(loss_linear, torch.ones(5), atol=1e-5)

    def test_mid_range_satisfaction(self) -> None:
        """Test loss with satisfaction = 0.5."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {
            "P": Predicate( lambda x: torch.ones(x.shape[0]) * 0.5)
        }

        x = torch.randn(1, 3)

        # Test R-Product (log)
        compiler_log = logic_to_loss(expr, predicates, tnorm=RProductTNorm())
        loss_log = compiler_log.loss(X=x, reduction="none")
        # -log(0.5) ≈ 0.693
        expected_log = -torch.log(torch.tensor(0.5))
        assert torch.allclose(loss_log, expected_log.expand(5), atol=1e-3)

        # Test Lukasiewicz (linear)
        compiler_linear = logic_to_loss(
            expr, predicates, tnorm=LukasiewiczTNorm()
        )
        loss_linear = compiler_linear.loss(X=x, reduction="none")
        # 1 - 0.5 = 0.5
        assert torch.allclose(loss_linear, torch.ones(5) * 0.5, atol=1e-5)


class TestReductionModes:
    """Test different reduction modes with post-processing."""

    @pytest.mark.parametrize(
        "tnorm_class",
        [RProductTNorm, SProductTNorm, LukasiewiczTNorm, GodelTNorm],
    )
    def test_mean_reduction(self, tnorm_class) -> None:
        """Test mean reduction across all t-norms."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {
            "P": Predicate( lambda x: torch.tensor([0.3, 0.6, 0.9]))
        }

        compiler = logic_to_loss(expr, predicates, tnorm=tnorm_class())
        x = torch.randn(1, 5)

        # Use quantify='none' to get per-batch losses, then apply reductions
        loss_mean = compiler.loss(X=x, quantify='none', reduction="mean")
        loss_none = compiler.loss(X=x, quantify='none', reduction="none")

        expected_mean = loss_none.mean()
        assert torch.allclose(loss_mean, expected_mean, atol=1e-5)
        assert loss_mean.shape == ()  # Scalar

    @pytest.mark.parametrize(
        "tnorm_class",
        [RProductTNorm, SProductTNorm, LukasiewiczTNorm, GodelTNorm],
    )
    def test_sum_reduction(self, tnorm_class) -> None:
        """Test sum reduction across all t-norms."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {
            "P": Predicate( lambda x: torch.tensor([0.3, 0.6, 0.9]))
        }

        compiler = logic_to_loss(expr, predicates, tnorm=tnorm_class())
        x = torch.randn(1, 5)

        # Use quantify='none' to get per-batch losses, then apply reductions
        loss_sum = compiler.loss(X=x, quantify='none', reduction="sum")
        loss_none = compiler.loss(X=x, quantify='none', reduction="none")

        expected_sum = loss_none.sum()
        assert torch.allclose(loss_sum, expected_sum, atol=1e-5)
        assert loss_sum.shape == ()  # Scalar

    @pytest.mark.parametrize(
        "tnorm_class",
        [RProductTNorm, SProductTNorm, LukasiewiczTNorm, GodelTNorm],
    )
    def test_none_reduction(self, tnorm_class) -> None:
        """Test no reduction (returns per-sample loss)."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        batch_size = 7
        predicates = {
            "P": Predicate( lambda x: torch.rand(batch_size))
        }

        compiler = logic_to_loss(expr, predicates, tnorm=tnorm_class())
        x = torch.randn(batch_size, 5)

        # Use quantify='none' to get per-batch losses with no reduction
        loss_none = compiler.loss(X=x, quantify='none', reduction="none")

        assert loss_none.shape == (batch_size,)
        assert (loss_none >= 0).all()  # Losses should be non-negative


class TestCombinedPostProcessingAndReduction:
    """Test all combinations of t-norms and reduction modes."""

    @pytest.mark.parametrize(
        "tnorm_class,reduction",
        [
            (RProductTNorm, "mean"),
            (RProductTNorm, "sum"),
            (RProductTNorm, "none"),
            (SProductTNorm, "mean"),
            (SProductTNorm, "sum"),
            (SProductTNorm, "none"),
            (LukasiewiczTNorm, "mean"),
            (LukasiewiczTNorm, "sum"),
            (LukasiewiczTNorm, "none"),
            (GodelTNorm, "mean"),
            (GodelTNorm, "sum"),
            (GodelTNorm, "none"),
        ],
    )
    def test_all_combinations(self, tnorm_class, reduction) -> None:
        """Test all t-norm and reduction combinations work correctly."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        batch_size = 10
        predicates = {
            "P": Predicate( lambda x: torch.rand(x.shape[0])),
            "Q": Predicate( lambda x: torch.rand(x.shape[0])),
        }

        compiler = logic_to_loss(expr, predicates, tnorm=tnorm_class())
        x = torch.randn(batch_size, 5)

        # Use quantify='none' to get per-batch losses, then apply reduction
        loss = compiler.loss(X=x, quantify='none', reduction=reduction)

        # Verify shape
        if reduction == "none":
            assert loss.shape == (batch_size,)
        else:
            assert loss.shape == ()

        # Verify no NaN/Inf
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

        # Verify non-negative
        assert (loss >= 0).all()


class TestPostProcessingParameterOverride:
    """Test that post_processing parameter allows overriding defaults."""

    def test_override_r_product_to_linear(self) -> None:
        """Test overriding R-Product's default log to linear."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {
            "P": Predicate( lambda x: torch.tensor([0.8]))
        }

        compiler = logic_to_loss(expr, predicates, tnorm=RProductTNorm())
        x = torch.randn(1, 5)

        # Default should be log
        loss_default = compiler.loss(X=x, reduction="none")
        satisfaction = compiler(X=x)
        expected_log = -torch.log(satisfaction)
        assert torch.allclose(loss_default, expected_log, atol=1e-5)

        # Override to linear
        loss_linear = compiler.loss(
            X=x, reduction="none", post_processing="linear"
        )
        expected_linear = 1.0 - satisfaction
        assert torch.allclose(loss_linear, expected_linear, atol=1e-5)

    def test_override_lukasiewicz_to_log(self) -> None:
        """Test overriding Lukasiewicz's default linear to log."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {
            "P": Predicate( lambda x: torch.tensor([0.6]))
        }

        compiler = logic_to_loss(
            expr, predicates, tnorm=LukasiewiczTNorm()
        )
        x = torch.randn(1, 5)

        # Default should be linear
        loss_default = compiler.loss(X=x, reduction="none")
        satisfaction = compiler(X=x)
        expected_linear = 1.0 - satisfaction
        assert torch.allclose(loss_default, expected_linear, atol=1e-5)

        # Override to log
        loss_log = compiler.loss(X=x, reduction="none", post_processing="log")
        expected_log = -torch.log(satisfaction)
        assert torch.allclose(loss_log, expected_log, atol=1e-5)

    def test_invalid_postprocessing_raises_error(self) -> None:
        """Test that invalid post_processing raises ValueError."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {
            "P": Predicate( lambda x: torch.tensor([0.5]))
        }

        compiler = logic_to_loss(expr, predicates)
        x = torch.randn(1, 5)

        # Invalid string should raise error
        with pytest.raises(ValueError, match="Unknown post-processing"):
            compiler.loss(X=x, post_processing="invalid_mode")

    def test_none_uses_tnorm_recommendation(self) -> None:
        """Test that post_processing=None uses t-norm's recommendation."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {
            "P": Predicate( lambda x: torch.tensor([0.7]))
        }

        # R-Product recommends log
        compiler_r = logic_to_loss(expr, predicates, tnorm=RProductTNorm())
        x = torch.randn(1, 5)

        loss_none = compiler_r.loss(X=x, reduction="none", post_processing=None)
        loss_default = compiler_r.loss(X=x, reduction="none")

        # Should be the same (both use t-norm's recommendation)
        assert torch.allclose(loss_none, loss_default, atol=1e-5)


class TestComplexExpressions:
    """Test post-processing with complex logical expressions."""

    def test_complex_expression_with_r_product(self) -> None:
        """Test -log post-processing with complex expression."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q, R = Symbol("P Q R")
        expr = sp.And(sp.Or(P(X), Q(X)), sp.Not(R(X)))

        predicates = {
            "P": Predicate( lambda x: torch.ones(x.shape[0]) * 0.6),
            "Q": Predicate( lambda x: torch.ones(x.shape[0]) * 0.4),
            "R": Predicate( lambda x: torch.ones(x.shape[0]) * 0.3),
        }

        compiler = logic_to_loss(expr, predicates, tnorm=RProductTNorm())
        x = torch.randn(1, 3)

        satisfaction = compiler(X=x)
        loss = compiler.loss(X=x, reduction="none")

        # Should be -log(satisfaction)
        expected_loss = -torch.log(satisfaction)
        assert torch.allclose(loss, expected_loss, atol=1e-5)

    def test_complex_expression_with_lukasiewicz(self) -> None:
        """Test linear post-processing with complex expression."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q, R = Symbol("P Q R")
        expr = sp.Implies(sp.And(P(X), Q(X)), R(X))

        predicates = {
            "P": Predicate( lambda x: torch.ones(x.shape[0]) * 0.7),
            "Q": Predicate( lambda x: torch.ones(x.shape[0]) * 0.8),
            "R": Predicate( lambda x: torch.ones(x.shape[0]) * 0.6),
        }

        compiler = logic_to_loss(
            expr, predicates, tnorm=LukasiewiczTNorm()
        )
        x = torch.randn(1, 3)

        satisfaction = compiler(X=x)
        loss = compiler.loss(X=x, reduction="none")

        # Should be 1 - satisfaction
        expected_loss = 1.0 - satisfaction
        assert torch.allclose(loss, expected_loss, atol=1e-5)
