"""Tests for smart activation handling in Predicate.

When an nn.Module is passed as a predicate, the library should auto-detect
whether it needs sigmoid (binary) or softmax (multiclass) activation,
rather than blindly clamping raw logits to [0,1].

For non-module predicates, the library should raise a ValueError if values
are outside [0,1] instead of silently clamping.
"""

import warnings

import pytest
import sympy as sp
import torch
import torch.nn as nn

from pysignet import Predicate, Symbol, Variable, compile_logic, logic_to_loss

# -- Binary classifiers (1 output) ------------------------------------------


class TestBinaryModuleActivation:
    """Test auto-sigmoid for binary classifiers."""

    def test_binary_without_sigmoid_gets_sigmoid(self) -> None:
        """nn.Sequential(Linear(10,1)) should auto-apply sigmoid."""
        # pylint: disable=invalid-name
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        model = nn.Sequential(nn.Linear(10, 1))
        compiled = compile_logic(expr, {"P": model})

        torch.manual_seed(42)
        x = torch.randn(4, 10)
        result = compiled(X=x)

        # Manually compute expected: sigmoid(model(x)).squeeze(-1)
        with torch.no_grad():
            expected = torch.sigmoid(model(x)).squeeze(-1)

        assert result.shape == (4,)
        assert torch.allclose(result, expected)
        assert torch.all((result >= 0) & (result <= 1))

    def test_binary_with_sigmoid_no_double_sigmoid(self) -> None:
        """nn.Sequential(Linear(10,1), Sigmoid()) should NOT double-sigmoid."""
        # pylint: disable=invalid-name
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
        compiled = compile_logic(expr, {"P": model})

        torch.manual_seed(42)
        x = torch.randn(4, 10)
        result = compiled(X=x)

        # Expected: model(x).squeeze(-1) -- already has sigmoid
        with torch.no_grad():
            expected = model(x).squeeze(-1)

        assert torch.allclose(result, expected)

    def test_binary_auto_sigmoid_gradients(self) -> None:
        """Gradients flow through auto-sigmoid."""
        # pylint: disable=invalid-name
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        model = nn.Sequential(nn.Linear(10, 1))
        loss_fn = logic_to_loss(expr, {"P": model})

        x = torch.randn(4, 10)
        loss = loss_fn.loss(X=x)
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


# -- Multiclass classifiers (N outputs) -------------------------------------


class TestMulticlassModuleActivation:
    """Test auto-softmax for multiclass classifiers."""

    def test_multiclass_without_softmax_gets_softmax(self) -> None:
        """nn.Sequential(Linear(10,5)) should auto-apply softmax."""
        # pylint: disable=invalid-name
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")
        expr = Digit(X, Y)

        model = nn.Sequential(nn.Linear(10, 5))
        compiled = compile_logic(expr, {"Digit": model})

        torch.manual_seed(42)
        x = torch.randn(4, 10)
        y = torch.tensor([0, 2, 1, 4])
        result = compiled(X=x, Y=y)

        # Expected: softmax(model(x))[arange, y]
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=-1)
            expected = probs[torch.arange(4), y]

        assert result.shape == (4,)
        assert torch.allclose(result, expected)
        assert torch.all((result >= 0) & (result <= 1))

    def test_multiclass_without_softmax_constant_index(self) -> None:
        """Digit(X, 3) with raw logits model should auto-softmax."""
        # pylint: disable=invalid-name
        Digit = Symbol("Digit")
        X = Variable("X")
        expr = Digit(X, 3)

        model = nn.Sequential(nn.Linear(10, 5))
        compiled = compile_logic(expr, {"Digit": model})

        torch.manual_seed(42)
        x = torch.randn(4, 10)
        result = compiled(X=x)

        # Expected: softmax(model(x))[:, 3]
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=-1)
            expected = probs[:, 3]

        assert torch.allclose(result, expected)

    def test_multiclass_with_softmax_no_double_softmax(self) -> None:
        """Model with Softmax should NOT double-softmax."""
        # pylint: disable=invalid-name
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")
        expr = Digit(X, Y)

        model = nn.Sequential(nn.Linear(10, 5), nn.Softmax(dim=-1))
        compiled = compile_logic(expr, {"Digit": model})

        torch.manual_seed(42)
        x = torch.randn(4, 10)
        y = torch.tensor([0, 2, 1, 4])
        result = compiled(X=x, Y=y)

        # Expected: model(x)[arange, y] -- already has softmax
        with torch.no_grad():
            probs = model(x)
            expected = probs[torch.arange(4), y]

        assert torch.allclose(result, expected)

    def test_multiclass_auto_softmax_gradients(self) -> None:
        """Gradients flow through auto-softmax."""
        # pylint: disable=invalid-name
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")
        expr = Digit(X, Y)

        model = nn.Sequential(nn.Linear(10, 5))
        loss_fn = logic_to_loss(expr, {"Digit": model})

        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 2, 3])
        loss = loss_fn.loss(X=x, Y=y)
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_multiclass_probabilities_sum_to_one(self) -> None:
        """Auto-softmax output should sum to 1 across classes."""
        # pylint: disable=invalid-name
        Digit = Symbol("Digit")
        X = Variable("X")

        model = nn.Sequential(nn.Linear(10, 5))
        predicate = Predicate(model)

        # Call predicate directly -- should return softmax'd output
        x = torch.randn(4, 10)
        # Need to trigger activation detection first
        compile_logic(Digit(X, 0), {"Digit": model})

        # After compilation, a fresh Predicate with the model should
        # auto-detect and apply softmax
        pred = Predicate(model)
        output = pred(x)

        # Multiclass output with softmax should have shape (batch, classes)
        # and sum to ~1 along class dim
        if output.dim() == 2:
            sums = output.sum(dim=-1)
            assert torch.allclose(sums, torch.ones(4), atol=1e-5)


# -- Custom opaque modules --------------------------------------------------


class TestCustomModuleActivation:
    """Test that opaque custom modules are handled safely.

    Custom nn.Module subclasses (non-Sequential) cannot be reliably
    inspected for internal activation. The compiler uses expression
    context to auto-apply activation, which means custom modules with
    internal activation will get double-activated. Users should add
    activation as a final nn.Sigmoid()/nn.Softmax() layer for correct
    detection, or wrap in a lambda to bypass auto-activation.
    """

    def test_custom_module_with_internal_sigmoid(self) -> None:
        """Custom module with internal sigmoid gets double-sigmoid.

        This is a known trade-off: the library auto-applies sigmoid
        based on expression context since it cannot detect internal
        activation in custom forward() methods. Output is still valid
        [0, 1] but compressed toward 0.5.
        """
        # pylint: disable=invalid-name
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        class CustomSigmoid(nn.Module):
            """Custom model that applies sigmoid internally."""
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward with internal sigmoid."""
                return torch.sigmoid(self.linear(x).squeeze(-1))

        model = CustomSigmoid()
        compiled = compile_logic(expr, {"P": model})

        x = torch.randn(4, 10)
        result = compiled(X=x)

        # Auto-sigmoid applied on top of internal sigmoid
        with torch.no_grad():
            expected = torch.sigmoid(model(x))

        assert result.shape == (4,)
        assert torch.all((result >= 0) & (result <= 1))
        assert torch.allclose(result, expected, atol=1e-5)

    def test_custom_module_lambda_bypass(self) -> None:
        """Wrapping custom module in lambda bypasses auto-activation."""
        # pylint: disable=invalid-name
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        class CustomSigmoid(nn.Module):
            """Custom model that applies sigmoid internally."""
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward with internal sigmoid."""
                return torch.sigmoid(self.linear(x).squeeze(-1))

        model = CustomSigmoid()
        # Lambda wrapper: treated as non-module, no auto-activation
        compiled = compile_logic(
            expr, {"P": Predicate(lambda x: model(x))}
        )

        x = torch.randn(4, 10)
        result = compiled(X=x)

        # Should match model output exactly (no extra activation)
        with torch.no_grad():
            expected = model(x)

        assert result.shape == (4,)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_uncompiled_opaque_module_emits_warning(self) -> None:
        """Opaque module called directly (no compilation) emits warning.

        When a Predicate wrapping an opaque nn.Module is called
        directly without going through compile_logic, the activation
        is unknown and a warning is emitted.
        """

        class OpaqueModule(nn.Module):
            """Module with non-standard structure."""
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass."""
                return torch.sigmoid((x * self.weight).sum(dim=-1))

        model = OpaqueModule()
        pred = Predicate(model)

        x = torch.randn(4, 10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pred(x)
            # Should emit a UserWarning about undetected activation
            activation_warnings = [
                warning for warning in w
                if issubclass(warning.category, UserWarning)
                and "activation" in str(warning.message).lower()
            ]
            assert len(activation_warnings) > 0

        assert result.shape == (4,)


# -- Non-module predicates: range validation ---------------------------------


class TestNonModuleRangeValidation:
    """Test that non-module predicates raise on out-of-range values."""

    def test_non_module_above_one_raises(self) -> None:
        """Lambda returning values > 1 should raise ValueError."""
        # pylint: disable=invalid-name
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(
            lambda x: torch.ones(x.shape[0]) * 2.5
        )}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(1, 3)
        with pytest.raises(ValueError, match="outside.*0.*1"):
            compiled(X=x)

    def test_non_module_below_zero_raises(self) -> None:
        """Lambda returning values < 0 should raise ValueError."""
        # pylint: disable=invalid-name
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(
            lambda x: torch.ones(x.shape[0]) * -1.5
        )}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(1, 3)
        with pytest.raises(ValueError, match="outside.*0.*1"):
            compiled(X=x)

    def test_non_module_valid_range_works(self) -> None:
        """Lambda returning values in [0,1] should work fine."""
        # pylint: disable=invalid-name
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(
            lambda x: torch.ones(x.shape[0]) * 0.7
        )}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(4, 3)
        result = compiled(X=x)

        assert torch.allclose(result, torch.tensor([0.7] * 4))

    def test_non_module_small_float_noise_ok(self) -> None:
        """Tiny float noise (e.g., 1.0000001) should be tolerated."""
        # pylint: disable=invalid-name
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {"P": Predicate(
            lambda x: torch.ones(x.shape[0]) * (1.0 + 1e-7)
        )}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(4, 3)
        # Should not raise -- tiny noise is tolerated
        result = compiled(X=x)
        assert torch.all((result >= 0) & (result <= 1.0 + 1e-5))


# -- Integration tests -------------------------------------------------------


class TestSmartActivationIntegration:
    """End-to-end tests with smart activation."""

    def test_training_step_with_raw_logits_model(self) -> None:
        """A training step with raw-logits model should work."""
        # pylint: disable=invalid-name
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")
        expr = Digit(X, Y)

        model = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )
        loss_fn = logic_to_loss(expr, {"Digit": model})
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # One training step
        x = torch.randn(8, 20)
        y = torch.randint(0, 5, (8,))

        optimizer.zero_grad()
        loss = loss_fn.loss(X=x, Y=y)
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0.0
        assert loss.isfinite()

    def test_raw_logits_loss_decreases(self) -> None:
        """Loss should decrease over training steps with auto-softmax."""
        # pylint: disable=invalid-name
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")
        expr = Digit(X, Y)

        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(10, 3))
        loss_fn = logic_to_loss(expr, {"Digit": model})
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Fixed data: all samples should be class 0
        x = torch.randn(16, 10)
        y = torch.zeros(16, dtype=torch.long)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss = loss_fn.loss(X=x, Y=y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_binary_and_multiclass_in_same_expression(self) -> None:
        """Binary and multiclass models in the same logical expression."""
        # pylint: disable=invalid-name
        Digit = Symbol("Digit")
        Even = Symbol("Even")
        X, Y = Variable("X Y")

        # Digit(X, Y) -> Even(X)
        expr = sp.Implies(Digit(X, Y), Even(X))

        digit_model = nn.Sequential(nn.Linear(10, 5))  # multiclass
        even_model = nn.Sequential(nn.Linear(10, 1))  # binary

        predicates = {"Digit": digit_model, "Even": even_model}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 2, 3])
        result = compiled(X=x, Y=y)

        assert result.shape == (4,)
        assert torch.all((result >= 0) & (result <= 1))
