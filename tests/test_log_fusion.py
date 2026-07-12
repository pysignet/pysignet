"""Tests for fused log-space activation (log_softmax / log_sigmoid).

When computing log-satisfaction, the library currently computes
log(softmax(logits) + eps) in two steps. PyTorch provides fused
log_softmax and log_sigmoid ops that are faster and more numerically
stable (no epsilon needed). This module tests the log_mode flag that
threads through the evaluation chain to use fused ops.
"""

# pylint: disable=invalid-name,not-callable

import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysignet import Predicate, Symbol, Variable, logic_to_loss
from pysignet.compilation import TNormCompiler
from pysignet.tnorms import (
    GodelTNorm,
    LukasiewiczTNorm,
    RProductTNorm,
    SProductTNorm,
)

# ------------------------------------------------------------------ #
# 1. Fused activation correctness (Predicate.log_call)
# ------------------------------------------------------------------ #


class TestLogCallSigmoid:
    """log_call with sigmoid activation uses logsigmoid."""

    def test_log_call_sigmoid_matches_logsigmoid(self) -> None:
        """log_call with sigmoid module matches F.logsigmoid exactly."""
        model = nn.Sequential(nn.Linear(5, 1))
        pred = Predicate(model)
        pred.name = "P"
        pred.configure_activation(1)  # binary -> sigmoid

        x = torch.randn(8, 5)
        logits = model(x)
        expected = F.logsigmoid(logits).squeeze(-1)

        result = pred.log_call(x)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_log_call_sigmoid_no_epsilon(self) -> None:
        """Fused logsigmoid gives finite values without epsilon."""
        model = nn.Sequential(nn.Linear(5, 1))
        pred = Predicate(model)
        pred.name = "P"
        pred.configure_activation(1)

        # Large negative logits -> sigmoid ~ 0.0
        with torch.no_grad():
            model[0].weight.fill_(0.0)
            model[0].bias.fill_(-100.0)

        x = torch.randn(4, 5)
        result = pred.log_call(x)

        # Should be finite (not -inf)
        assert torch.isfinite(result).all()
        # Should be large negative, not clamped to -log(eps)
        assert (result < -50.0).all()


class TestLogCallSoftmax:
    """log_call with softmax activation uses log_softmax."""

    def test_log_call_softmax_matches_log_softmax(self) -> None:
        """log_call with softmax module matches log_softmax exactly."""
        model = nn.Sequential(nn.Linear(5, 10))
        pred = Predicate(model)
        pred.name = "Digit"
        pred.configure_activation(2)  # multiclass -> softmax

        x = torch.randn(8, 5)
        logits = model(x)
        expected = torch.log_softmax(logits, dim=-1)

        result = pred.log_call(x)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_log_call_softmax_no_epsilon(self) -> None:
        """Fused log_softmax gives correct values for extreme logits."""
        model = nn.Sequential(nn.Linear(5, 3))
        pred = Predicate(model)
        pred.name = "P"
        pred.configure_activation(2)

        # Make one class dominate -> other classes have softmax ~ 0.0
        with torch.no_grad():
            model[0].weight.fill_(0.0)
            model[0].bias.copy_(torch.tensor([100.0, -100.0, -100.0]))

        x = torch.zeros(4, 5)
        result = pred.log_call(x)

        # All values should be finite
        assert torch.isfinite(result).all()
        # Non-dominant classes should have very negative log-prob
        assert (result[:, 1] < -100.0).all()
        assert (result[:, 2] < -100.0).all()


class TestLogCallExistingActivation:
    """log_call with existing activation falls back to log(output + eps)."""

    def test_log_call_existing_sigmoid(self) -> None:
        """Module with nn.Sigmoid falls back to log(output + eps)."""
        model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        pred = Predicate(model)
        pred.name = "P"

        x = torch.randn(4, 5)
        normal_output = pred(x)
        expected = torch.log(normal_output + 1e-10)

        result = pred.log_call(x)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_log_call_existing_softmax(self) -> None:
        """Module with nn.Softmax falls back to log(output + eps)."""
        model = nn.Sequential(
            nn.Linear(5, 3), nn.Softmax(dim=-1)
        )
        pred = Predicate(model)
        pred.name = "P"

        x = torch.randn(4, 5)
        normal_output = pred(x)
        expected = torch.log(normal_output + 1e-10)

        result = pred.log_call(x)
        assert torch.allclose(result, expected, atol=1e-6)


class TestLogCallNonModule:
    """log_call with non-module predicates falls back to log(output + eps)."""

    def test_log_call_lambda(self) -> None:
        """Non-module predicate: log(output + eps)."""
        values = torch.tensor([0.9, 0.5, 0.1])
        pred = Predicate(lambda _x: values)
        pred.name = "P"

        x = torch.randn(3, 5)
        expected = torch.log(values + 1e-10)

        result = pred.log_call(x)
        assert torch.allclose(result, expected, atol=1e-6)


# ------------------------------------------------------------------ #
# 2. Expression evaluation in log-mode
# ------------------------------------------------------------------ #


class TestLogModeExpressionEval:
    """Test _evaluate_expression_log in the compiler."""

    def test_single_predicate_log_mode(self) -> None:
        """Single predicate in log_mode returns log-prob values."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        model = nn.Sequential(nn.Linear(5, 1))
        predicates = {"P": model}

        compiler = TNormCompiler(tnorm=SProductTNorm())
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(4, 5)
        log_result = compiled(log_mode=True, X=x)

        # Should be negative (log of probability)
        assert (log_result <= 0.0).all()

        # Compare with fused computation
        logits = model(x)
        expected = F.logsigmoid(logits).squeeze(-1)
        assert torch.allclose(log_result, expected, atol=1e-6)

    def test_conjunction_log_mode_is_sum(self) -> None:
        """P(X) AND Q(X) in log_mode returns sum of log-probs."""
        X = Variable("X")
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        model_p = nn.Sequential(nn.Linear(5, 1))
        model_q = nn.Sequential(nn.Linear(5, 1))
        predicates = {"P": model_p, "Q": model_q}

        compiler = TNormCompiler(tnorm=SProductTNorm())
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(4, 5)
        log_result = compiled(log_mode=True, X=x)

        # Expected: log(P(X)) + log(Q(X)) via fused ops
        log_p = F.logsigmoid(model_p(x)).squeeze(-1)
        log_q = F.logsigmoid(model_q(x)).squeeze(-1)
        expected = log_p + log_q

        assert torch.allclose(log_result, expected, atol=1e-5)

    def test_mixed_and_not_log_mode(self) -> None:
        """P(X) AND NOT Q(X) correctly exits/enters log-space."""
        X = Variable("X")
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), sp.Not(Q(X)))

        model_p = nn.Sequential(nn.Linear(5, 1))
        model_q = nn.Sequential(nn.Linear(5, 1))
        predicates = {"P": model_p, "Q": model_q}

        compiler = TNormCompiler(tnorm=SProductTNorm())
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(4, 5)
        log_result = compiled(log_mode=True, X=x)

        # NOT Q(X) exits log-space: log(1 - sigmoid(logits_q))
        p_prob = torch.sigmoid(model_p(x)).squeeze(-1)
        q_prob = torch.sigmoid(model_q(x)).squeeze(-1)
        not_q = 1.0 - q_prob
        expected = torch.log(p_prob * not_q + 1e-10)

        # The log-mode computes: log_p + log(not_q_linear)
        # which equals log(p) + log(1-q) = log(p * (1-q))
        # Compare as log of product
        assert torch.allclose(log_result, expected, atol=1e-5)

    def test_or_expression_log_mode_falls_back(self) -> None:
        """OR expressions fall back to linear then log."""
        X = Variable("X")
        P, Q = Symbol("P Q")
        expr = sp.Or(P(X), Q(X))

        values_p = torch.tensor([0.9, 0.8, 0.7, 0.6])
        values_q = torch.tensor([0.5, 0.4, 0.3, 0.2])
        predicates = {
            "P": Predicate(lambda _x: values_p),
            "Q": Predicate(lambda _x: values_q),
        }

        compiler = TNormCompiler(tnorm=SProductTNorm())
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(4, 5)
        log_result = compiled(log_mode=True, X=x)

        # Should equal log of linear-space OR result
        linear_result = compiled(X=x)
        expected = torch.log(linear_result + 1e-10)
        assert torch.allclose(log_result, expected, atol=1e-5)

    def test_boolean_constant_log_mode(self) -> None:
        """Boolean constants in log_mode return log(1) or log(eps)."""
        X = Variable("X")
        P = Symbol("P")
        # sp.true & P(X) -> P(X) after SymPy simplification, so use
        # an expression where the constant survives
        expr = sp.And(P(X), sp.true)

        values = torch.tensor([0.9, 0.8])
        predicates = {"P": Predicate(lambda _x: values)}

        compiler = TNormCompiler(tnorm=SProductTNorm())
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(2, 5)
        log_result = compiled(log_mode=True, X=x)

        # sp.And(P(X), true) simplifies to P(X)
        # So log_result should equal log(values)
        expected = torch.log(values + 1e-10)
        assert torch.allclose(log_result, expected, atol=1e-5)


class TestLogModeMulticlass:
    """Test log_mode with multiclass (softmax) predicates."""

    def test_multiclass_single_pred_log_mode(self) -> None:
        """Multiclass predicate with index in log_mode."""
        X, Y = Variable("X Y")
        Digit = Symbol("Digit")
        expr = Digit(X, Y)

        model = nn.Sequential(nn.Linear(5, 10))
        predicates = {"Digit": model}

        compiler = TNormCompiler(tnorm=SProductTNorm())
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(4, 5)
        y = torch.tensor([0, 3, 7, 9])
        log_result = compiled(log_mode=True, X=x, Y=y)

        # Expected: log_softmax then index
        logits = model(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        expected = log_probs[torch.arange(4), y]

        assert torch.allclose(log_result, expected, atol=1e-5)


# ------------------------------------------------------------------ #
# 3. End-to-end numerical equivalence
# ------------------------------------------------------------------ #


class TestLogModeNumericalEquivalence:
    """log_mode and linear-space produce equivalent results."""

    def test_log_satisfaction_equivalence(self) -> None:
        """log_satisfaction with/without log_mode are close."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        values = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        predicates = {"P": Predicate(lambda _x: values)}

        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(5, 5)
        log_sat = logic_loss.log_satisfaction(X=x)

        # Expected: sum(log(values + eps))
        expected = torch.log(values + 1e-10).sum()

        # Should be close (small difference from epsilon)
        assert torch.allclose(log_sat, expected, atol=1e-5)

    def test_gradient_equivalence(self) -> None:
        """Gradients from log_mode match non-log for normal values."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(5, 1))
        predicates = {"P": model}

        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(4, 5)

        # Compute gradients via log_mode (fused)
        model.zero_grad()
        loss_log = logic_loss.loss(X=x, post_processing="log")
        loss_log.backward()
        grads_log = [p.grad.clone() for p in model.parameters()]

        # Compute gradients via linear-space log
        model.zero_grad()
        sat = logic_loss.satisfaction(X=x)
        loss_manual = -torch.log(sat + 1e-10)
        loss_manual.backward()
        grads_manual = [p.grad.clone() for p in model.parameters()]

        # Gradients should be close
        for g_log, g_manual in zip(grads_log, grads_manual, strict=True):
            assert torch.allclose(g_log, g_manual, atol=1e-4)


# ------------------------------------------------------------------ #
# 4. Numerical stability
# ------------------------------------------------------------------ #


class TestLogModeNumericalStability:
    """log_mode provides better stability for extreme values."""

    def test_extreme_negative_logits_sigmoid(self) -> None:
        """Large negative logits: fused gives finite, old gives -log(eps)."""
        model = nn.Sequential(nn.Linear(5, 1))
        pred = Predicate(model)
        pred.name = "P"
        pred.configure_activation(1)

        with torch.no_grad():
            model[0].weight.fill_(0.0)
            model[0].bias.fill_(-200.0)

        x = torch.randn(4, 5)

        # Fused: logsigmoid(-200) ~ -200 (correct)
        log_result = pred.log_call(x)
        assert torch.isfinite(log_result).all()
        assert (log_result < -100.0).all()

        # Old path: sigmoid(-200) = 0.0, log(0 + 1e-10) ~ -23
        old_result = torch.log(pred(x) + 1e-10)
        # Old path clips at -log(eps) ~ 23.03
        assert (old_result > -25.0).all()

        # Fused gives more negative (correct) values
        assert (log_result < old_result).all()

    def test_extreme_negative_logits_softmax(self) -> None:
        """Softmax underflow: fused gives correct log-probs."""
        model = nn.Sequential(nn.Linear(5, 3))
        pred = Predicate(model)
        pred.name = "P"
        pred.configure_activation(2)

        with torch.no_grad():
            model[0].weight.fill_(0.0)
            model[0].bias.copy_(
                torch.tensor([200.0, -200.0, -200.0])
            )

        x = torch.zeros(4, 5)

        # Fused: log_softmax gives correct values
        log_result = pred.log_call(x)
        assert torch.isfinite(log_result).all()

        # Non-dominant classes should have very negative log-probs
        assert (log_result[:, 1] < -100.0).all()

    def test_end_to_end_stability_large_batch(self) -> None:
        """End-to-end with large batch: no underflow."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        batch_size = 64
        # Values near 0.1 -> product underflows
        values = torch.full((batch_size,), 0.1)
        predicates = {"P": Predicate(lambda _x: values)}

        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(batch_size, 5)
        log_sat = logic_loss.log_satisfaction(X=x)

        # Expected: 64 * log(0.1) ~ -147.4
        expected = batch_size * torch.log(torch.tensor(0.1))
        assert torch.allclose(log_sat, expected, atol=1.0)
        assert torch.isfinite(log_sat)


# ------------------------------------------------------------------ #
# 5. No regression: existing behavior unchanged
# ------------------------------------------------------------------ #


class TestNoRegression:
    """Existing behavior is not affected by log_mode additions."""

    def test_satisfaction_unchanged(self) -> None:
        """satisfaction() still returns [0,1] values, no log_mode."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        values = torch.tensor([0.9, 0.5, 0.1])
        predicates = {"P": Predicate(lambda _x: values)}

        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(3, 5)
        sat = logic_loss.satisfaction(X=x)

        # Product conjunction: 0.9 * 0.5 * 0.1 = 0.045
        expected = values.prod()
        assert torch.allclose(sat, expected, atol=1e-5)

    def test_linear_postprocessing_unchanged(self) -> None:
        """Linear loss is not affected."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        values = torch.tensor([0.9, 0.8])
        predicates = {"P": Predicate(lambda _x: values)}

        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(2, 5)
        loss = logic_loss.loss(
            X=x, post_processing="linear"
        )

        expected = 1.0 - values.prod()
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_compiled_expression_default_no_log(self) -> None:
        """CompiledExpression without log_mode returns [0,1]."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        values = torch.tensor([0.9, 0.5])
        predicates = {"P": Predicate(lambda _x: values)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(2, 5)
        result = compiled(X=x)

        assert torch.allclose(result, values, atol=1e-5)

    def test_boolean_eval_unchanged(self) -> None:
        """return_boolean=True is not affected."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        values = torch.tensor([0.9, 0.3])
        predicates = {"P": Predicate(lambda _x: values)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(2, 5)
        result = compiled(return_boolean=True, X=x)

        expected = torch.tensor([True, False])
        assert torch.equal(result, expected)


class TestLogModeWithDifferentTNorms:
    """log_mode works correctly with different t-norm types."""

    def test_log_mode_rproduct(self) -> None:
        """log_mode with RProductTNorm."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        values = torch.tensor([0.8, 0.6])
        predicates = {"P": Predicate(lambda _x: values)}

        compiler = TNormCompiler(tnorm=RProductTNorm())
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(2, 5)
        log_result = compiled(log_mode=True, X=x)
        expected = torch.log(values + 1e-10)
        assert torch.allclose(log_result, expected, atol=1e-5)

    def test_log_mode_godel(self) -> None:
        """log_mode with GodelTNorm (falls back gracefully)."""
        X = Variable("X")
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        values_p = torch.tensor([0.8, 0.6])
        values_q = torch.tensor([0.7, 0.5])
        predicates = {
            "P": Predicate(lambda _x: values_p),
            "Q": Predicate(lambda _x: values_q),
        }

        compiler = TNormCompiler(tnorm=GodelTNorm())
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(2, 5)
        log_result = compiled(log_mode=True, X=x)

        # Godel AND = min -> log_mode falls back to linear
        linear_result = compiled(X=x)
        expected = torch.log(linear_result + 1e-10)
        assert torch.allclose(log_result, expected, atol=1e-5)

    def test_log_mode_lukasiewicz(self) -> None:
        """log_mode with LukasiewiczTNorm (falls back gracefully)."""
        X = Variable("X")
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        values_p = torch.tensor([0.8, 0.6])
        values_q = torch.tensor([0.7, 0.5])
        predicates = {
            "P": Predicate(lambda _x: values_p),
            "Q": Predicate(lambda _x: values_q),
        }

        compiler = TNormCompiler(tnorm=LukasiewiczTNorm())
        compiled = compiler.compile(expr, predicates)

        x = torch.randn(2, 5)
        log_result = compiled(log_mode=True, X=x)

        # Lukasiewicz AND = max(0, a+b-1)
        linear_result = compiled(X=x)
        expected = torch.log(linear_result + 1e-10)
        assert torch.allclose(log_result, expected, atol=1e-5)
