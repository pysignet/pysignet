"""Tests for nn.Module predicates with variable output indices.

When a multiclass nn.Module (e.g., a 10-class classifier) is used with
a variable as the class index -- Digit(X, Y) where Y is bound to a
tensor of integer labels -- the library should:
1. Call the model with only the input variable(s): model(X)
2. Use the index variable Y for per-element output selection:
   output[batch_idx, Y[batch_idx]]

This is distinct from the constant-index path (Digit(X, 3)) which
already works. The variable-index path is needed for supervised
training where labels come from a dataset.
"""

import sympy as sp
import torch
import torch.nn as nn

from pysignet import Symbol, Variable, compile_logic, logic_to_loss


# -- Test fixtures -----------------------------------------------------------


def _make_multiclass_model(
    input_dim: int, num_classes: int, with_softmax: bool = False
) -> nn.Module:
    """Create a simple multiclass model for testing."""
    layers = [nn.Linear(input_dim, num_classes)]
    if with_softmax:
        layers.append(nn.Softmax(dim=-1))
    return nn.Sequential(*layers)


# -- Basic variable-index tests ----------------------------------------------


class TestModuleVariableIndex:
    """Test nn.Module predicates where a variable is used as class index."""

    def test_basic_variable_index_compile_logic(self) -> None:
        """Digit(X, Y) with nn.Module should use Y as output index."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        expr = Digit(X, Y)

        model = _make_multiclass_model(10, 5, with_softmax=True)
        compiled = compile_logic(expr, {"Digit": model})

        batch_size = 4
        x = torch.randn(batch_size, 10)
        y = torch.tensor([0, 2, 1, 4])

        result = compiled(X=x, Y=y)

        assert result.shape == (batch_size,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_basic_variable_index_logic_to_loss(self) -> None:
        """Digit(X, Y) through logic_to_loss should work."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        expr = Digit(X, Y)

        model = _make_multiclass_model(10, 5, with_softmax=True)
        loss_fn = logic_to_loss(expr, {"Digit": model})

        batch_size = 8
        x = torch.randn(batch_size, 10)
        y = torch.randint(0, 5, (batch_size,))

        # satisfaction should return a scalar (default quantify='forall')
        sat = loss_fn.satisfaction(X=x, Y=y)
        assert sat.shape == ()
        assert sat.item() >= 0.0
        assert sat.item() <= 1.0

    def test_variable_index_without_softmax(self) -> None:
        """Model without softmax: library should auto-apply softmax."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        expr = Digit(X, Y)

        # Model outputs raw logits (no softmax)
        model = _make_multiclass_model(10, 5, with_softmax=False)
        compiled = compile_logic(expr, {"Digit": model})

        batch_size = 4
        x = torch.randn(batch_size, 10)
        y = torch.tensor([0, 3, 1, 4])

        result = compiled(X=x, Y=y)

        assert result.shape == (batch_size,)
        # After softmax, values should be in [0, 1]
        assert torch.all((result >= 0) & (result <= 1))

        # Should match manual softmax computation
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=-1)
            expected = probs[torch.arange(batch_size), y]
        assert torch.allclose(result, expected)


class TestModuleVariableIndexCorrectness:
    """Verify that variable indexing produces correct values."""

    def test_matches_manual_indexing(self) -> None:
        """Variable-index result should match manual gather."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        expr = Digit(X, Y)

        model = _make_multiclass_model(10, 5, with_softmax=True)
        compiled = compile_logic(expr, {"Digit": model})

        torch.manual_seed(42)
        batch_size = 8
        x = torch.randn(batch_size, 10)
        y = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])

        result = compiled(X=x, Y=y)

        # Manually compute expected result
        with torch.no_grad():
            probs = model(x)  # already has softmax
            expected = probs[torch.arange(batch_size), y]

        assert torch.allclose(result, expected)

    def test_matches_constant_index_path(self) -> None:
        """Variable Y=3 should match constant Digit(X, 3)."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        model = _make_multiclass_model(10, 5, with_softmax=True)

        # Variable-index path: Digit(X, Y) with Y=3
        compiled_var = compile_logic(Digit(X, Y), {"Digit": model})

        # Constant-index path: Digit(X, 3)
        compiled_const = compile_logic(Digit(X, 3), {"Digit": model})

        batch_size = 4
        x = torch.randn(batch_size, 10)
        y_all_three = torch.full((batch_size,), 3, dtype=torch.long)

        result_var = compiled_var(X=x, Y=y_all_three)
        result_const = compiled_const(X=x)

        assert torch.allclose(result_var, result_const)

    def test_different_labels_per_sample(self) -> None:
        """Each sample in batch can have a different label."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        expr = Digit(X, Y)

        model = _make_multiclass_model(10, 5, with_softmax=True)
        compiled = compile_logic(expr, {"Digit": model})

        x = torch.randn(3, 10)
        y = torch.tensor([0, 2, 4])  # Different class per sample

        result = compiled(X=x, Y=y)

        # Manually verify each element
        with torch.no_grad():
            probs = model(x)
            for i in range(3):
                expected_i = probs[i, y[i]]
                assert torch.isclose(result[i], expected_i)


class TestModuleVariableIndexInExpressions:
    """Test variable-indexed modules in compound logical expressions."""

    def test_in_and_expression(self) -> None:
        """Digit(X, Y) inside an And expression."""
        Digit = Symbol("Digit")
        P = Symbol("P")
        X, Y = Variable("X Y")

        expr = sp.And(Digit(X, Y), P(X))

        digit_model = _make_multiclass_model(10, 5, with_softmax=True)
        unary_model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

        predicates = {"Digit": digit_model, "P": unary_model}
        compiled = compile_logic(expr, predicates)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        y = torch.tensor([0, 1, 2, 3])

        result = compiled(X=x, Y=y)

        assert result.shape == (batch_size,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_in_implies_expression(self) -> None:
        """P(X) -> Digit(X, Y) should work."""
        Digit = Symbol("Digit")
        P = Symbol("P")
        X, Y = Variable("X Y")

        expr = sp.Implies(P(X), Digit(X, Y))

        digit_model = _make_multiclass_model(10, 5, with_softmax=True)
        unary_model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

        predicates = {"Digit": digit_model, "P": unary_model}
        compiled = compile_logic(expr, predicates)

        batch_size = 4
        x = torch.randn(batch_size, 10)
        y = torch.tensor([0, 1, 2, 3])

        result = compiled(X=x, Y=y)

        assert result.shape == (batch_size,)

    def test_in_not_expression(self) -> None:
        """Not(Digit(X, Y)) should work."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        expr = sp.Not(Digit(X, Y))

        model = _make_multiclass_model(10, 5, with_softmax=True)
        compiled = compile_logic(expr, {"Digit": model})

        batch_size = 4
        x = torch.randn(batch_size, 10)
        y = torch.tensor([0, 1, 2, 3])

        result = compiled(X=x, Y=y)

        assert result.shape == (batch_size,)
        assert torch.all((result >= 0) & (result <= 1))


class TestModuleVariableIndexGradients:
    """Test gradient flow with variable-indexed module predicates."""

    def test_gradients_flow_through_model(self) -> None:
        """Gradients should flow from loss through variable-indexed output."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        expr = Digit(X, Y)

        model = _make_multiclass_model(10, 5, with_softmax=True)
        loss_fn = logic_to_loss(expr, {"Digit": model})

        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 2, 3])

        loss = loss_fn.loss(X=x, Y=y)
        loss.backward()

        # All model parameters should have gradients
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_gradients_in_compound_expression(self) -> None:
        """Gradients flow through variable-indexed module in And expr."""
        Digit = Symbol("Digit")
        P = Symbol("P")
        X, Y = Variable("X Y")

        expr = sp.And(Digit(X, Y), P(X))

        digit_model = _make_multiclass_model(10, 5, with_softmax=True)
        unary_model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

        predicates = {"Digit": digit_model, "P": unary_model}
        loss_fn = logic_to_loss(expr, predicates)

        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 2, 3])

        loss = loss_fn.loss(X=x, Y=y)
        loss.backward()

        # Both models should receive gradients
        for param in digit_model.parameters():
            assert param.grad is not None
        for param in unary_model.parameters():
            assert param.grad is not None


class TestModuleVariableIndexLoss:
    """Test loss computation with variable-indexed module predicates."""

    def test_loss_returns_scalar(self) -> None:
        """loss() should return a scalar for optimization."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        expr = Digit(X, Y)

        model = _make_multiclass_model(10, 5, with_softmax=True)
        loss_fn = logic_to_loss(expr, {"Digit": model})

        x = torch.randn(8, 10)
        y = torch.randint(0, 5, (8,))

        loss = loss_fn.loss(X=x, Y=y)

        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_satisfaction_per_batch(self) -> None:
        """satisfaction(quantify='none') returns per-batch tensor."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        expr = Digit(X, Y)

        model = _make_multiclass_model(10, 5, with_softmax=True)
        loss_fn = logic_to_loss(expr, {"Digit": model})

        batch_size = 8
        x = torch.randn(batch_size, 10)
        y = torch.randint(0, 5, (batch_size,))

        sat = loss_fn.satisfaction(X=x, Y=y, quantify="none")

        assert sat.shape == (batch_size,)


class TestModuleVariableIndexEdgeCases:
    """Edge cases for variable-indexed module predicates."""

    def test_single_sample_batch(self) -> None:
        """Works with batch_size=1."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        expr = Digit(X, Y)

        model = _make_multiclass_model(10, 5, with_softmax=True)
        compiled = compile_logic(expr, {"Digit": model})

        x = torch.randn(1, 10)
        y = torch.tensor([3])

        result = compiled(X=x, Y=y)

        assert result.shape == (1,)

    def test_large_batch(self) -> None:
        """Works with a large batch."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        expr = Digit(X, Y)

        model = _make_multiclass_model(10, 5, with_softmax=True)
        compiled = compile_logic(expr, {"Digit": model})

        batch_size = 256
        x = torch.randn(batch_size, 10)
        y = torch.randint(0, 5, (batch_size,))

        result = compiled(X=x, Y=y)

        assert result.shape == (batch_size,)

    def test_ten_class_mnist_like(self) -> None:
        """Realistic MNIST-like setup: 784 -> 10 classes."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        expr = Digit(X, Y)

        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        compiled = compile_logic(expr, {"Digit": model})

        batch_size = 32
        x = torch.randn(batch_size, 784)
        y = torch.randint(0, 10, (batch_size,))

        result = compiled(X=x, Y=y)

        assert result.shape == (batch_size,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_custom_module_not_sequential(self) -> None:
        """Works with non-Sequential custom nn.Module."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        expr = Digit(X, Y)

        class CustomClassifier(nn.Module):
            """Custom classifier module."""
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass."""
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        model = CustomClassifier()
        compiled = compile_logic(expr, {"Digit": model})

        batch_size = 4
        x = torch.randn(batch_size, 10)
        y = torch.tensor([0, 1, 2, 3])

        result = compiled(X=x, Y=y)

        assert result.shape == (batch_size,)
        assert torch.all((result >= 0) & (result <= 1))
