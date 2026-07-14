"""Tests for CombinedLoss - loss-level combination of LogicLoss objects.

CombinedLoss combines independently compiled LogicLoss objects by
weighting and summing their already-computed loss values. It does not
touch compilation or satisfaction -- each LogicLoss keeps its own
predicates, variables, t-norm, and post-processing.
"""

import pytest
import torch
import torch.nn as nn

from pysignet import (
    CombinedLoss,
    LogicLoss,
    Predicate,
    Symbol,
    Variable,
    logic_to_loss,
)
from pysignet.tnorms import LukasiewiczTNorm, RProductTNorm


def _make_loss_a(value: float = 0.7) -> LogicLoss:
    """Build a LogicLoss for P(X) with a constant-valued predicate."""
    X = Variable("X")  # pylint: disable=invalid-name
    P = Symbol("P")  # pylint: disable=invalid-name
    predicates = {
        "P": Predicate(lambda x: torch.ones(x.shape[0]) * value)
    }
    return logic_to_loss(P(X), predicates, post_processing="linear")


def _make_loss_b(value: float = 0.4) -> LogicLoss:
    """Build a LogicLoss for Q(Y) with a constant-valued predicate."""
    Y = Variable("Y")  # pylint: disable=invalid-name
    Q = Symbol("Q")  # pylint: disable=invalid-name
    predicates = {
        "Q": Predicate(lambda y: torch.ones(y.shape[0]) * value)
    }
    return logic_to_loss(Q(Y), predicates, post_processing="linear")


class TestCombinedLossBasics:
    """Test basic weighted-sum combination."""

    def test_equal_weight_matches_manual_sum(self) -> None:
        """Default weights (1.0) match manual loss_a + loss_b."""
        loss_a = _make_loss_a()
        loss_b = _make_loss_b()
        combined = CombinedLoss({"a": loss_a, "b": loss_b})

        x = torch.randn(5, 3)
        y = torch.randn(5, 3)
        result = combined.loss({"a": {"X": x}, "b": {"Y": y}})
        manual = loss_a.loss(X=x) + loss_b.loss(Y=y)

        assert torch.allclose(result, manual, atol=1e-6)

    def test_weighted_combination_matches_manual_weighted_sum(self) -> None:
        """Static weights scale each constraint's loss before summing."""
        loss_a = _make_loss_a()
        loss_b = _make_loss_b()
        combined = CombinedLoss(
            {"a": loss_a, "b": loss_b},
            weights={"a": 2.0, "b": 0.5},
        )

        x = torch.randn(5, 3)
        y = torch.randn(5, 3)
        result = combined.loss({"a": {"X": x}, "b": {"Y": y}})
        manual = 2.0 * loss_a.loss(X=x) + 0.5 * loss_b.loss(Y=y)

        assert torch.allclose(result, manual, atol=1e-6)

    def test_normalize_divides_by_sum_of_weights(self) -> None:
        """normalize=True divides the weighted sum by sum(weights)."""
        loss_a = _make_loss_a()
        loss_b = _make_loss_b()
        combined = CombinedLoss(
            {"a": loss_a, "b": loss_b},
            weights={"a": 3.0, "b": 1.0},
            normalize=True,
        )

        x = torch.randn(5, 3)
        y = torch.randn(5, 3)
        result = combined.loss({"a": {"X": x}, "b": {"Y": y}})
        manual = (3.0 * loss_a.loss(X=x) + 1.0 * loss_b.loss(Y=y)) / 4.0

        assert torch.allclose(result, manual, atol=1e-6)

    def test_repr_smoke(self) -> None:
        """repr() does not raise and mentions the constraint names."""
        combined = CombinedLoss({"a": _make_loss_a(), "b": _make_loss_b()})
        text = repr(combined)

        assert "CombinedLoss" in text
        assert "a" in text
        assert "b" in text


class TestCombinedLossWeights:
    """Test static and learnable weights."""

    def test_learnable_weight_receives_gradient(self) -> None:
        """nn.Parameter weights get a gradient after backward()."""
        loss_a = _make_loss_a()
        loss_b = _make_loss_b()
        weight_a = nn.Parameter(torch.tensor(1.0))
        weight_b = nn.Parameter(torch.tensor(1.0))
        combined = CombinedLoss(
            {"a": loss_a, "b": loss_b},
            weights={"a": weight_a, "b": weight_b},
        )

        x = torch.randn(5, 3)
        y = torch.randn(5, 3)
        result = combined.loss({"a": {"X": x}, "b": {"Y": y}})
        result.backward()

        assert weight_a.grad is not None
        assert weight_b.grad is not None
        assert not torch.allclose(weight_a.grad, torch.tensor(0.0))
        assert not torch.allclose(weight_b.grad, torch.tensor(0.0))


class TestCombinedLossTrainableParameters:
    """Test trainable_parameters aggregation and deduplication."""

    def test_includes_submodel_and_weight_parameters(self) -> None:
        """trainable_parameters concatenates sub-model and weight params."""
        X = Variable("X")  # pylint: disable=invalid-name
        Y = Variable("Y")  # pylint: disable=invalid-name
        P, Q = Symbol("P Q")  # pylint: disable=invalid-name
        model_p = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())
        model_q = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())
        loss_a = logic_to_loss(P(X), {"P": Predicate(model_p)})
        loss_b = logic_to_loss(Q(Y), {"Q": Predicate(model_q)})
        weight_a = nn.Parameter(torch.tensor(1.0))
        combined = CombinedLoss(
            {"a": loss_a, "b": loss_b},
            weights={"a": weight_a, "b": 1.0},
        )

        params = combined.trainable_parameters

        # 2 params per Linear (weight, bias) * 2 models + 1 weight param.
        assert len(params) == 5
        assert any(p is weight_a for p in params)

    def test_deduplicates_shared_predicate_parameters(self) -> None:
        """A model reused across constraints is not double-counted."""
        X = Variable("X")  # pylint: disable=invalid-name
        Y = Variable("Y")  # pylint: disable=invalid-name
        P, Q = Symbol("P Q")  # pylint: disable=invalid-name
        shared_model = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())
        loss_a = logic_to_loss(P(X), {"P": Predicate(shared_model)})
        loss_b = logic_to_loss(Q(Y), {"Q": Predicate(shared_model)})
        combined = CombinedLoss({"a": loss_a, "b": loss_b})

        params = combined.trainable_parameters

        assert len(params) == 2  # weight + bias, counted once


class TestCombinedLossValidation:
    """Test error handling for mismatched or missing inputs."""

    def test_mismatched_bindings_keys_raises(self) -> None:
        """Missing or extra constraint names in bindings raise ValueError."""
        loss_a = _make_loss_a()
        loss_b = _make_loss_b()
        combined = CombinedLoss({"a": loss_a, "b": loss_b})
        x = torch.randn(5, 3)

        with pytest.raises(ValueError, match="bindings"):
            combined.loss({"a": {"X": x}})

        with pytest.raises(ValueError, match="bindings"):
            combined.loss(
                {"a": {"X": x}, "b": {"Y": x}, "c": {"Z": x}}
            )

    def test_mismatched_weights_keys_raises(self) -> None:
        """Missing or extra constraint names in weights raise ValueError."""
        loss_a = _make_loss_a()
        loss_b = _make_loss_b()

        with pytest.raises(ValueError, match="weights"):
            CombinedLoss({"a": loss_a, "b": loss_b}, weights={"a": 1.0})

        with pytest.raises(ValueError, match="weights"):
            CombinedLoss(
                {"a": loss_a, "b": loss_b},
                weights={"a": 1.0, "b": 1.0, "c": 1.0},
            )

    def test_empty_losses_raises(self) -> None:
        """An empty losses dict raises ValueError."""
        with pytest.raises(ValueError):
            CombinedLoss({})

    def test_quantify_none_reduction_none_raises(self) -> None:
        """quantify='none' with reduction='none' is ambiguous."""
        loss_a = _make_loss_a()
        loss_b = _make_loss_b()
        combined = CombinedLoss({"a": loss_a, "b": loss_b})
        x = torch.randn(5, 3)
        y = torch.randn(5, 3)

        with pytest.raises(ValueError):
            combined.loss(
                {"a": {"X": x}, "b": {"Y": y}},
                quantify="none",
                reduction="none",
            )

    def test_unknown_post_processing_key_raises(self) -> None:
        """post_processing naming an unknown constraint raises ValueError."""
        loss_a = _make_loss_a()
        loss_b = _make_loss_b()
        combined = CombinedLoss({"a": loss_a, "b": loss_b})
        x = torch.randn(5, 3)
        y = torch.randn(5, 3)

        with pytest.raises(ValueError, match="post_processing"):
            combined.loss(
                {"a": {"X": x}, "b": {"Y": y}},
                post_processing={"c": "linear"},
            )


class TestCombinedLossQuantifyReduction:
    """Test quantify/reduction pass-through and heterogeneous t-norms."""

    def test_quantify_none_reduction_mean_matches_manual(self) -> None:
        """quantify='none' with reduction='mean' collapses each constraint."""
        loss_a = _make_loss_a()
        loss_b = _make_loss_b()
        combined = CombinedLoss({"a": loss_a, "b": loss_b})
        x = torch.randn(5, 3)
        y = torch.randn(5, 3)

        result = combined.loss(
            {"a": {"X": x}, "b": {"Y": y}},
            quantify="none",
            reduction="mean",
        )
        manual = loss_a.loss(
            X=x, quantify="none", reduction="mean"
        ) + loss_b.loss(Y=y, quantify="none", reduction="mean")

        assert torch.allclose(result, manual, atol=1e-6)

    def test_quantify_exists_matches_manual(self) -> None:
        """quantify='exists' is forwarded to each sub-loss."""
        loss_a = _make_loss_a()
        loss_b = _make_loss_b()
        combined = CombinedLoss({"a": loss_a, "b": loss_b})
        x = torch.randn(5, 3)
        y = torch.randn(5, 3)

        result = combined.loss(
            {"a": {"X": x}, "b": {"Y": y}}, quantify="exists"
        )
        manual = loss_a.loss(X=x, quantify="exists") + loss_b.loss(
            Y=y, quantify="exists"
        )

        assert torch.allclose(result, manual, atol=1e-6)

    def test_heterogeneous_tnorms_combine(self) -> None:
        """Constraints compiled with different t-norms combine correctly."""
        X = Variable("X")  # pylint: disable=invalid-name
        Y = Variable("Y")  # pylint: disable=invalid-name
        P, Q = Symbol("P Q")  # pylint: disable=invalid-name
        loss_a = logic_to_loss(
            P(X),
            {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)},
            tnorm=RProductTNorm(),
            post_processing="linear",
        )
        loss_b = logic_to_loss(
            Q(Y),
            {"Q": Predicate(lambda y: torch.ones(y.shape[0]) * 0.4)},
            tnorm=LukasiewiczTNorm(),
            post_processing="linear",
        )
        combined = CombinedLoss({"a": loss_a, "b": loss_b})
        x = torch.randn(5, 3)
        y = torch.randn(5, 3)

        result = combined.loss({"a": {"X": x}, "b": {"Y": y}})
        manual = loss_a.loss(X=x) + loss_b.loss(Y=y)

        assert torch.allclose(result, manual, atol=1e-6)


class TestCombinedLossPostProcessing:
    """Test per-constraint post-processing overrides."""

    def test_per_constraint_post_processing_override(self) -> None:
        """post_processing dict overrides one constraint's default."""
        X = Variable("X")  # pylint: disable=invalid-name
        Y = Variable("Y")  # pylint: disable=invalid-name
        P, Q = Symbol("P Q")  # pylint: disable=invalid-name
        loss_a = logic_to_loss(
            P(X),
            {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)},
            post_processing="log",
        )
        loss_b = logic_to_loss(
            Q(Y),
            {"Q": Predicate(lambda y: torch.ones(y.shape[0]) * 0.4)},
            post_processing="linear",
        )
        combined = CombinedLoss({"a": loss_a, "b": loss_b})
        x = torch.randn(5, 3)
        y = torch.randn(5, 3)

        result = combined.loss(
            {"a": {"X": x}, "b": {"Y": y}},
            post_processing={"a": "linear"},
        )
        manual = loss_a.loss(X=x, post_processing="linear") + loss_b.loss(
            Y=y
        )

        assert torch.allclose(result, manual, atol=1e-6)
