"""Tests for LogicLoss - wrapper for compiled logic with loss computation.

This module tests the LogicLoss class which wraps compiled logic and provides
loss computation with configurable post-processing and reduction modes.
"""

import pytest
import torch
import torch.nn as nn

# Import from pysignet package
from pysignet import LogicLoss, Predicate, Symbol, TNormCompiler, Variable


class TestLogicLossBasics:
    """Test basic LogicLoss functionality."""

    def test_logic_loss_initialization(self) -> None:
        """Test LogicLoss can wrap compiled logic."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        # Compile and wrap in LogicLoss
        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled)

        assert logic_loss._compiled_expr is compiled
        # Default TNormCompiler uses RProductTNorm which recommends 'log'
        assert logic_loss.default_post_processing == "log"

    def test_logic_loss_call_returns_satisfaction(self) -> None:
        """Test that calling LogicLoss returns satisfaction values."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled)

        x = torch.randn(10, 5)
        # Use quantify='none' to get per-batch satisfaction values
        satisfaction = logic_loss.satisfaction(X=x, quantify='none')

        # Should return satisfaction values in [0, 1]
        assert isinstance(satisfaction, torch.Tensor)
        assert satisfaction.shape == (10,)
        assert (satisfaction >= 0).all()
        assert (satisfaction <= 1).all()
        assert torch.allclose(satisfaction, torch.tensor(0.7), atol=1e-5)

    def test_logic_loss_loss_method_returns_scalar(self) -> None:
        """Test that loss() method returns loss values."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled)

        x = torch.randn(1, 5)
        loss = logic_loss.loss(X=x)

        # Should return scalar loss (mean reduction by default)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        # Default post-processing is 'log': loss = -log(satisfaction) = -log(0.7)
        expected_loss = -torch.log(torch.tensor(0.7))
        assert torch.allclose(loss, expected_loss, atol=1e-5)


class TestLogicLossPostProcessing:
    """Test different post-processing modes."""

    def test_post_processing_log(self) -> None:
        """Test post_processing='log' applies -log(satisfaction)."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled, post_processing="log")

        x = torch.randn(1, 5)
        loss = logic_loss.loss(X=x)

        # Log post-processing: loss = -log(satisfaction) = -log(0.5)
        expected_loss = -torch.log(torch.tensor(0.5))
        assert torch.allclose(loss, expected_loss, atol=1e-5)

    def test_post_processing_linear(self) -> None:
        """Test post_processing='linear' applies 1 - satisfaction."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled, post_processing="linear")

        x = torch.randn(1, 5)
        loss = logic_loss.loss(X=x)

        # Linear post-processing: loss = 1 - satisfaction = 1 - 0.7 = 0.3
        assert torch.allclose(loss, torch.tensor(0.3), atol=1e-5)

    def test_post_processing_custom_callable(self) -> None:
        """Test custom post-processing function."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.6)}

        # Custom post-processing: square the violation
        def custom_postprocessing(satisfaction):
            return (1 - satisfaction) ** 2

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(
            compiled, post_processing=custom_postprocessing
        )

        x = torch.randn(1, 5)
        loss = logic_loss.loss(X=x)

        # Custom: (1 - 0.6)^2 = 0.4^2 = 0.16
        assert torch.allclose(loss, torch.tensor(0.16), atol=1e-5)

    def test_invalid_post_processing_raises_error(self) -> None:
        """Test invalid post-processing mode raises ValueError."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled, post_processing="invalid_mode")

        x = torch.randn(1, 5)

        # Should raise ValueError when trying to compute loss
        with pytest.raises(ValueError, match="Unknown post-processing"):
            logic_loss.loss(X=x)


class TestLogicLossReductionModes:
    """Test different reduction modes."""

    def test_reduction_mean(self) -> None:
        """Test reduction='mean' averages over batch."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled, post_processing="linear")

        x = torch.randn(10, 5)
        # reduction requires quantify='none' to get per-batch losses first
        loss = logic_loss.loss(X=x, quantify='none', reduction="mean")

        # Should return scalar (mean of per-sample losses)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        # Linear: loss = 1 - 0.7 = 0.3 for each sample, mean is 0.3
        assert torch.allclose(loss, torch.tensor(0.3), atol=1e-5)

    def test_reduction_sum(self) -> None:
        """Test reduction='sum' sums over batch."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled, post_processing="linear")

        x = torch.randn(10, 5)
        # reduction requires quantify='none' to get per-batch losses first
        loss = logic_loss.loss(X=x, quantify='none', reduction="sum")

        # Should return scalar (sum of per-sample losses)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        # Linear: loss = 1 - 0.7 = 0.3 for each sample, sum over 10 = 3.0
        assert torch.allclose(loss, torch.tensor(3.0), atol=1e-5)

    def test_reduction_none(self) -> None:
        """Test reduction='none' returns per-sample losses."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled, post_processing="linear")

        x = torch.randn(10, 5)
        # quantify='none' gives per-batch losses, reduction='none' keeps them
        loss = logic_loss.loss(X=x, quantify='none', reduction="none")

        # Should return per-sample losses
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == (10,)  # Per-sample
        # Linear: loss = 1 - 0.7 = 0.3 for each sample
        assert torch.allclose(loss, torch.ones(10) * 0.3, atol=1e-5)

    def test_invalid_reduction_raises_error(self) -> None:
        """Test invalid reduction mode raises ValueError."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled)

        x = torch.randn(1, 5)

        # Should raise ValueError for invalid reduction mode
        # reduction requires quantify='none'
        with pytest.raises(ValueError, match="Invalid reduction"):
            logic_loss.loss(X=x, quantify='none', reduction="invalid_mode")


class TestLogicLossCombinedPostProcessingAndReduction:
    """Test combinations of post-processing and reduction."""

    def test_log_postprocessing_mean_reduction(self) -> None:
        """Test log post-processing with mean reduction."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.5)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled, post_processing="log")

        x = torch.randn(10, 5)
        # reduction requires quantify='none'
        loss = logic_loss.loss(X=x, quantify='none', reduction="mean")

        # Log post-processing: -log(0.5), mean reduction
        expected_loss = -torch.log(torch.tensor(0.5))
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        assert torch.allclose(loss, expected_loss, atol=1e-5)

    def test_linear_postprocessing_sum_reduction(self) -> None:
        """Test linear post-processing with sum reduction."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled, post_processing="linear")

        x = torch.randn(10, 5)
        # reduction requires quantify='none'
        loss = logic_loss.loss(X=x, quantify='none', reduction="sum")

        # Linear post-processing: 1 - 0.7 = 0.3, sum over batch of 10 = 3.0
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        assert torch.allclose(loss, torch.tensor(3.0), atol=1e-5)

    def test_all_combinations_valid(self) -> None:
        """Test all valid combinations of post-processing and reduction."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        # Test all combinations
        post_processings = ["linear", "log"]
        reductions = ["mean", "sum", "none"]

        for post_proc in post_processings:
            for reduction in reductions:
                logic_loss = LogicLoss(compiled, post_processing=post_proc)
                x = torch.randn(10, 5)
                # reduction requires quantify='none'
                loss = logic_loss.loss(X=x, quantify='none', reduction=reduction)

                # Verify loss is computed without error
                assert isinstance(loss, torch.Tensor)
                if reduction == "none":
                    assert loss.shape == (10,)
                else:
                    assert loss.shape == ()  # Scalar


class TestLogicLossGradientFlow:
    """Test gradient flow through LogicLoss."""

    def test_gradients_flow_through_satisfaction(self) -> None:
        """Test gradients flow when calling LogicLoss()."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        # Create a simple model as predicate
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([0.5]))

            def forward(self, x):
                # Return sigmoid of weight for each sample in batch
                return torch.sigmoid(self.weight).expand(x.shape[0])

        model = SimpleModel()
        predicates = {"P": model}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled)

        x = torch.randn(1, 5)
        satisfaction = logic_loss.satisfaction(X=x)

        # Compute some scalar from satisfaction and backprop
        scalar_output = satisfaction.mean()
        scalar_output.backward()

        # Check gradients flow to model parameters
        assert model.weight.grad is not None
        assert not torch.allclose(model.weight.grad, torch.tensor(0.0))

    def test_gradients_flow_through_loss(self) -> None:
        """Test gradients flow through loss() method."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        # Create a simple model as predicate
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([0.5]))

            def forward(self, x):
                return torch.sigmoid(self.weight).expand(x.shape[0])

        model = SimpleModel()
        predicates = {"P": model}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled, post_processing="linear")

        x = torch.randn(10, 5)
        # reduction requires quantify='none'
        loss = logic_loss.loss(X=x, quantify='none', reduction="mean")

        # Backprop through loss
        loss.backward()

        # Check gradients flow to model parameters
        assert model.weight.grad is not None
        assert not torch.allclose(model.weight.grad, torch.tensor(0.0))

    def test_gradients_with_log_postprocessing(self) -> None:
        """Test gradients with log post-processing."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        # Create a simple model as predicate
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([0.5]))

            def forward(self, x):
                return torch.sigmoid(self.weight).expand(x.shape[0])

        model = SimpleModel()
        predicates = {"P": model}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled, post_processing="log")

        x = torch.randn(10, 5)
        # reduction requires quantify='none'
        loss = logic_loss.loss(X=x, quantify='none', reduction="mean")

        # Backprop through log post-processing
        loss.backward()

        # Check gradients flow to model parameters
        assert model.weight.grad is not None
        assert not torch.allclose(model.weight.grad, torch.tensor(0.0))

    def test_gradients_with_linear_postprocessing(self) -> None:
        """Test gradients with linear post-processing."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        # Create a simple model as predicate
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([0.5]))

            def forward(self, x):
                return torch.sigmoid(self.weight).expand(x.shape[0])

        model = SimpleModel()
        predicates = {"P": model}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled, post_processing="linear")

        x = torch.randn(10, 5)
        # reduction requires quantify='none'
        loss = logic_loss.loss(X=x, quantify='none', reduction="mean")

        # Backprop through linear post-processing
        loss.backward()

        # Check gradients flow to model parameters
        assert model.weight.grad is not None
        assert not torch.allclose(model.weight.grad, torch.tensor(0.0))


class TestLogicLossTrainableParameters:
    """Test get_trainable_parameters() method."""

    def test_get_trainable_parameters_with_models(self) -> None:
        """Test extracting parameters from model-based predicates."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = P(X) & Q(X)

        # Create two simple models using Sequential (auto-detectable)
        model_p = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        model_q = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())

        # Wrap models in Predicate objects for parameter extraction
        predicates = {"P": Predicate(model_p), "Q": Predicate(model_q)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled)

        # Get trainable parameters
        params = logic_loss.trainable_parameters

        # Should have 4 parameters (weight and bias from each Linear layer)
        params_list = list(params)
        assert len(params_list) == 4  # 2 models * (weight + bias)

    def test_get_trainable_parameters_no_models(self) -> None:
        """Test with no model-based predicates returns empty list."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        # Use lambda function (no trainable parameters)
        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled)

        # Get trainable parameters
        params = logic_loss.trainable_parameters

        # Should be empty
        params_list = list(params)
        assert len(params_list) == 0

    def test_parameters_can_be_optimized(self) -> None:
        """Test extracted parameters can be used with optimizer."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        # Create a simple model using Sequential (auto-detectable)
        model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())

        # Wrap model in Predicate for parameter extraction
        predicates = {"P": Predicate(model)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled, post_processing="linear")

        # Create optimizer with extracted parameters
        params = logic_loss.trainable_parameters
        optimizer = torch.optim.SGD(params, lr=0.01)

        # Run one optimization step
        x = torch.randn(10, 5)
        # reduction requires quantify='none'
        loss = logic_loss.loss(X=x, quantify='none', reduction="mean")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that optimizer worked (no errors raised)
        assert True


class TestLogicLossBoundaryConditions:
    """Test boundary conditions for satisfaction and loss."""

    def test_perfect_satisfaction_zero_loss(self) -> None:
        """Test satisfaction=1.0 gives loss~0 (with log handling)."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        # Predicate always returns 1.0 (perfect satisfaction)
        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]))}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        # Test with linear post-processing
        logic_loss_linear = LogicLoss(compiled, post_processing="linear")
        x = torch.randn(10, 5)
        # reduction requires quantify='none'
        loss_linear = logic_loss_linear.loss(X=x, quantify='none', reduction="mean")

        # Linear: loss = 1 - 1 = 0
        assert torch.allclose(loss_linear, torch.tensor(0.0), atol=1e-5)

        # Test with log post-processing (log(1) = 0)
        logic_loss_log = LogicLoss(compiled, post_processing="log")
        loss_log = logic_loss_log.loss(X=x, quantify='none', reduction="mean")

        # Log: loss = -log(1) = 0
        assert torch.allclose(loss_log, torch.tensor(0.0), atol=1e-5)

    def test_zero_satisfaction_high_loss(self) -> None:
        """Test satisfaction=0.0 gives high loss."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        # Predicate returns very small value (near 0)
        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 1e-7)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)

        # Test with linear post-processing
        logic_loss_linear = LogicLoss(compiled, post_processing="linear")
        x = torch.randn(10, 5)
        # reduction requires quantify='none'
        loss_linear = logic_loss_linear.loss(X=x, quantify='none', reduction="mean")

        # Linear: loss = 1 - 1e-7 ≈ 1
        assert torch.allclose(loss_linear, torch.tensor(1.0), atol=1e-5)

        # Test with log post-processing (log should give large positive value)
        logic_loss_log = LogicLoss(compiled, post_processing="log")
        loss_log = logic_loss_log.loss(X=x, quantify='none', reduction="mean")

        # Log: loss = -log(1e-7) is large and positive
        expected_log_loss = -torch.log(torch.tensor(1e-7))
        assert torch.allclose(loss_log, expected_log_loss, atol=1e-3)
        assert loss_log > 10.0  # Should be large

    def test_numerical_stability_near_zero(self) -> None:
        """Test numerical stability when satisfaction near 0."""
        X, Y = Variable("X Y")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X, Y)

        # Test various small values
        small_values = [1e-6, 1e-7, 1e-8]

        for val in small_values:
            predicates = {"P": Predicate(lambda x, v: torch.ones(x.shape[0]) * v)}

            compiler = TNormCompiler()
            compiled = compiler.compile(expr, predicates)

            # Linear post-processing should be stable
            logic_loss_linear = LogicLoss(
                compiled,
                post_processing="linear",
            )
            inputs = {"X": torch.randn(10, 5), "Y": val}
            # reduction requires quantify='none'
            loss_linear = logic_loss_linear.loss(
                **inputs, quantify='none', reduction="mean"
            )

            # Should not produce NaN or inf
            assert not torch.isnan(loss_linear)
            assert not torch.isinf(loss_linear)
            assert loss_linear >= 0.0

            # Log post-processing should also be stable (though large)
            logic_loss_log = LogicLoss(compiled, post_processing="log")
            loss_log = logic_loss_log.loss(**inputs, quantify='none', reduction="mean")

            # Should not produce NaN (inf is possible for very small values)
            assert not torch.isnan(loss_log)
            assert loss_log >= 0.0

    def test_numerical_stability_near_one(self) -> None:
        """Test numerical stability when satisfaction near 1."""
        X, Y = Variable("X Y")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X, Y)

        # Test various values close to 1
        near_one_values = [0.999999, 0.9999999, 0.99999999]

        for val in near_one_values:
            predicates = {"P": Predicate(lambda x, v: torch.ones(x.shape[0]) * v)}

            compiler = TNormCompiler()
            compiled = compiler.compile(expr, predicates)

            # Linear post-processing should be stable
            logic_loss_linear = LogicLoss(
                compiled, post_processing="linear"
            )
            inputs = {"X": torch.randn(10, 5), "Y": val}
            # reduction requires quantify='none'
            loss_linear = logic_loss_linear.loss(
                **inputs, quantify='none', reduction="mean"
            )

            # Should not produce NaN or inf
            assert not torch.isnan(loss_linear)
            assert not torch.isinf(loss_linear)
            assert loss_linear >= 0.0
            assert loss_linear < 1.0

            # Log post-processing should also be stable
            logic_loss_log = LogicLoss(compiled, post_processing="log")
            loss_log = logic_loss_log.loss(**inputs, quantify='none', reduction="mean")

            # Should not produce NaN or inf
            assert not torch.isnan(loss_log)
            assert not torch.isinf(loss_log)
            assert loss_log >= 0.0


class TestLogicLossInputHandling:
    """Test LogicLoss handles different input types."""

    def test_logic_loss_with_single_tensor_input(self) -> None:
        """Test with single tensor input."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = P(X) & Q(X)

        # Both predicates receive the same input tensor
        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7),
        }

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled, post_processing="linear")

        # Single tensor input
        x = torch.randn(10, 5)
        # Use quantify='none' to get per-batch results
        satisfaction = logic_loss.satisfaction(X=x, quantify='none')
        # reduction requires quantify='none'
        loss = logic_loss.loss(X=x, quantify='none', reduction="mean")

        # Should work without errors
        assert isinstance(satisfaction, torch.Tensor)
        assert satisfaction.shape == (10,)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()

    def test_logic_loss_with_dict_input(self) -> None:
        """Test with dict input."""
        X, Y = Variable("X Y")
        # pylint: disable=invalid-name
        P, Q = Symbol("P Q")
        expr = P(X) & Q(Y)

        # Each predicate receives different input
        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.8),
            "Q": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7),
        }

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled, post_processing="linear")

        # Keyword arguments with different tensors for P and Q
        x = torch.randn(10, 5)
        y = torch.randn(10, 3)
        # Use quantify='none' to get per-batch results
        satisfaction = logic_loss.satisfaction(X=x, Y=y, quantify='none')
        # reduction requires quantify='none'
        loss = logic_loss.loss(X=x, Y=y, quantify='none', reduction="mean")

        # Should work without errors
        assert isinstance(satisfaction, torch.Tensor)
        assert satisfaction.shape == (10,)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()

    def test_logic_loss_with_varying_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        X = Variable("X")
        # pylint: disable=invalid-name
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.7)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled, post_processing="linear")

        # Test with different batch sizes
        batch_sizes = [1, 5, 10, 32, 100]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 5)
            # Use quantify='none' to get per-batch results
            satisfaction = logic_loss.satisfaction(X=x, quantify='none')
            # reduction requires quantify='none'
            loss_none = logic_loss.loss(X=x, quantify='none', reduction="none")
            loss_mean = logic_loss.loss(X=x, quantify='none', reduction="mean")
            loss_sum = logic_loss.loss(X=x, quantify='none', reduction="sum")

            # Check shapes
            assert satisfaction.shape == (batch_size,)
            assert loss_none.shape == (batch_size,)
            assert loss_mean.shape == ()
            assert loss_sum.shape == ()

            # Check values
            # Linear: 1 - 0.7 = 0.3 for each sample
            expected_per_sample = torch.ones(batch_size) * 0.3
            assert torch.allclose(loss_none, expected_per_sample, atol=1e-5)
            assert torch.allclose(loss_mean, torch.tensor(0.3), atol=1e-5)
            assert torch.allclose(loss_sum, torch.tensor(0.3 * batch_size), atol=1e-5)
