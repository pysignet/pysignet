"""Tests for log-space batch loss computation.

When using product-based conjunction (RProduct/SProduct) with
quantify='forall' and post_processing='log', the loss should be
computed as sum(-log(per_batch_i)) directly, instead of
-log(product(per_batch_i)). These are mathematically equivalent
but the log-space form avoids numerical underflow with large batches.
"""

import torch
import torch.nn as nn

from pysignet import Predicate, Symbol, Variable, logic_to_loss
from pysignet.compilation import TNormCompiler
from pysignet.tnorms import RProductTNorm, SProductTNorm, MixedTNorm


class TestLogSpaceForallLoss:
    """Tests for log-space forall + log loss computation."""

    def test_log_forall_matches_sum_neg_log(self) -> None:
        """Log forall loss equals sum(-log(p_i))."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        values = torch.tensor([0.9, 0.8, 0.7, 0.6])
        predicates = {
            "P": Predicate(lambda _x: values)
        }

        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(4, 3)
        loss = logic_loss.loss(X=x, post_processing="log")

        # Expected: sum(-log(p_i))
        expected = (-torch.log(values + 1e-10)).sum()
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_log_forall_large_batch_no_underflow(self) -> None:
        """Log forall loss does not underflow with large batches."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        batch_size = 32
        # Each value ~0.1 -> product = 1e-32, underflows
        values = torch.full((batch_size,), 0.1)
        predicates = {
            "P": Predicate(lambda _x: values)
        }

        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(batch_size, 3)
        loss = logic_loss.loss(X=x, post_processing="log")

        # Expected: 32 * -log(0.1) = 32 * 2.3026 = 73.68
        expected = batch_size * (-torch.log(torch.tensor(0.1)))
        assert torch.allclose(loss, expected, atol=0.1)

        # Should NOT be the underflow constant -log(eps) ~= 23.03
        underflow_constant = -torch.log(torch.tensor(1e-10))
        assert not torch.allclose(loss, underflow_constant, atol=1.0)

    def test_log_forall_gradients_nonzero(self) -> None:
        """Log forall loss produces non-zero gradients."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        predicates = {"P": model}

        logic_loss = logic_to_loss(expr, predicates)

        batch_size = 32
        x = torch.randn(batch_size, 5)

        loss = logic_loss.loss(X=x, post_processing="log")
        loss.backward()

        # Gradients should exist and not be near-zero
        for param in model.parameters():
            assert param.grad is not None
            assert param.grad.abs().max() > 1e-10

    def test_log_forall_with_sproduct(self) -> None:
        """Log-space also works with SProductTNorm."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        values = torch.tensor([0.5, 0.6, 0.7])
        predicates = {
            "P": Predicate(lambda _x: values)
        }

        logic_loss = logic_to_loss(
            expr, predicates,
            tnorm=SProductTNorm()
        )

        x = torch.randn(3, 3)
        loss = logic_loss.loss(X=x, post_processing="log")

        # Expected: sum(-log(p_i))
        expected = (-torch.log(values + 1e-10)).sum()
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_log_forall_with_mixed_tnorm(self) -> None:
        """Log-space works with MixedTNorm (batch uses RProduct)."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        values = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        predicates = {
            "P": Predicate(lambda _x: values)
        }

        # MixedTNorm - batch compiler should be RProduct
        logic_loss = logic_to_loss(
            expr, predicates,
            tnorm=MixedTNorm()
        )

        x = torch.randn(5, 3)
        loss = logic_loss.loss(X=x, post_processing="log")

        # Expected: sum(-log(p_i))
        expected = (-torch.log(values + 1e-10)).sum()
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_log_forall_single_element(self) -> None:
        """Log-space with single element: -log(p)."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        values = torch.tensor([0.8])
        predicates = {
            "P": Predicate(lambda _x: values)
        }

        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(1, 3)
        loss = logic_loss.loss(X=x, post_processing="log")

        expected = -torch.log(torch.tensor(0.8) + 1e-10)
        assert torch.allclose(loss, expected, atol=1e-5)


class TestNonLogPathUnchanged:
    """Ensure non-log paths are NOT affected by the optimization."""

    def test_linear_forall_unchanged(self) -> None:
        """Linear post-processing with forall is unchanged."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        values = torch.tensor([0.9, 0.8, 0.7])
        predicates = {
            "P": Predicate(lambda _x: values)
        }

        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(3, 3)
        loss = logic_loss.loss(X=x, post_processing="linear")

        # Linear forall: 1 - product(values)
        expected = 1.0 - values.prod()
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_log_quantify_none_unchanged(self) -> None:
        """Log with quantify='none' gives per-batch -log(p_i)."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        values = torch.tensor([0.9, 0.8, 0.7])
        predicates = {
            "P": Predicate(lambda _x: values)
        }

        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(3, 3)
        loss = logic_loss.loss(
            X=x, post_processing="log",
            quantify="none", reduction="none"
        )

        # Per-batch: -log(p_i)
        expected = -torch.log(values + 1e-10)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_log_exists_unchanged(self) -> None:
        """Log with quantify='exists' is unchanged."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        values = torch.tensor([0.9, 0.8, 0.7])
        predicates = {
            "P": Predicate(lambda _x: values)
        }

        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(3, 3)
        loss = logic_loss.loss(
            X=x, post_processing="log", quantify="exists"
        )

        # Exists uses disjunction: 1 - prod(1-p_i)
        exists_sat = 1.0 - (1.0 - values).prod()
        expected = -torch.log(exists_sat + 1e-10)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_callable_postprocessing_forall_unchanged(self) -> None:
        """Callable post-processing with forall uses product path."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        values = torch.tensor([0.9, 0.8, 0.7])
        predicates = {
            "P": Predicate(lambda _x: values)
        }

        logic_loss = logic_to_loss(expr, predicates)

        x = torch.randn(3, 3)
        custom_fn = lambda sat: (1.0 - sat) ** 2
        loss = logic_loss.loss(
            X=x, post_processing=custom_fn, quantify="forall"
        )

        # Custom: (1 - product(values))^2
        product_sat = values.prod()
        expected = (1.0 - product_sat) ** 2
        assert torch.allclose(loss, expected, atol=1e-5)


class TestLogSpaceEndToEnd:
    """End-to-end tests for log-space batch loss."""

    def test_mnist_like_training_step(self) -> None:
        """Simulate one MNIST-like training step with log forall."""
        X, Y = Variable("X Y")
        Digit = Symbol("Digit")
        expr = Digit(X, Y)

        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        predicates = {"Digit": model}
        logic_loss = logic_to_loss(expr, predicates)

        batch_size = 32
        x = torch.randn(batch_size, 784)
        y = torch.randint(0, 10, (batch_size,))

        # Compute loss
        loss = logic_loss.loss(X=x, Y=y)

        # Loss should be finite, not NaN, and not the underflow
        # constant
        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

        # Should produce meaningful gradients
        loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in model.parameters()
            if p.grad is not None
        ]
        assert all(g > 1e-10 for g in grad_norms)

    def test_log_forall_loss_decreases_with_training(self) -> None:
        """Loss decreases over a few training steps."""
        torch.manual_seed(42)

        X, Y = Variable("X Y")
        Digit = Symbol("Digit")
        expr = Digit(X, Y)

        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
        )

        predicates = {"Digit": model}
        logic_loss = logic_to_loss(expr, predicates)

        optimizer = torch.optim.Adam(
            logic_loss.trainable_parameters, lr=0.01
        )

        # Fixed batch
        x = torch.randn(16, 10)
        y = torch.randint(0, 3, (16,))

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            loss = logic_loss.loss(X=x, Y=y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0]
