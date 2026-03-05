"""Tests for custom nn.Module support with FOL interface.

Tests that custom PyTorch modules (not nn.Sequential) work correctly
with the FOL interface, including gradient flow and parameter extraction.
"""

import pytest
import torch
import torch.nn as nn
import sympy as sp

from pysignet import Symbol, Variable, compile_logic, logic_to_loss, Predicate
from pysignet.compilation import TNormCompiler
from pysignet.loss import LogicLoss


class TestCustomUnaryModules:
    """Test custom nn.Module with unary predicates."""

    def test_custom_unary_module_compiles(self):
        """Custom unary module compiles and runs."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        class CustomUnary(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return torch.sigmoid(self.linear(x).squeeze(-1))

        model = CustomUnary()
        compiled = compile_logic(expr, {"P": model})

        batch_size = 5
        x = torch.randn(batch_size, 10)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify="none")

        assert result.shape == (batch_size,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_custom_unary_gradient_flow(self):
        """Gradients flow through custom unary modules."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        class CustomUnary(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10))

            def forward(self, x):
                return torch.sigmoid((x * self.weight).sum(dim=-1))

        model = CustomUnary()
        logic_loss = logic_to_loss(expr, {"P": model})

        x = torch.randn(1, 10)
        loss = logic_loss.loss(X=x)
        loss.backward()

        assert model.weight.grad is not None
        assert not torch.all(model.weight.grad == 0)

    def test_custom_unary_with_complex_expression(self):
        """Custom unary modules work in complex expressions."""
        X = Variable("X")
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), sp.Not(Q(X)))

        class CustomModel(nn.Module):
            def __init__(self, name):
                super().__init__()
                self.name = name
                self.fc = nn.Linear(8, 1)

            def forward(self, x):
                return torch.sigmoid(self.fc(x).squeeze(-1))

        model_p = CustomModel("P")
        model_q = CustomModel("Q")

        compiled = compile_logic(expr, {"P": model_p, "Q": model_q})

        batch_size = 3
        x = torch.randn(batch_size, 8)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify="none")

        assert result.shape == (batch_size,)

    def test_custom_unary_training_loop(self):
        """Can train custom unary modules."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 1)

            def forward(self, x):
                return torch.sigmoid(self.linear(x).squeeze(-1))

        model = CustomModel()
        logic_loss = logic_to_loss(expr, {"P": model})

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        initial_weight = model.linear.weight.data.clone()

        # Training step
        x = torch.randn(1, 5)
        loss = logic_loss.loss(X=x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.allclose(model.linear.weight.data, initial_weight)


class TestCustomBinaryModules:
    """Test custom nn.Module with binary predicates."""

    def test_custom_binary_module_compiles(self):
        """Custom binary module compiles and runs.

        Note: For binary predicates with nn.Modules, the module outputs
        multiple channels and the second argument is used for indexing.
        The module forward() takes only the first argument.
        """
        X = Variable("X")
        P = Symbol("P")
        # P(X, 2) - select channel 2
        expr = P(X, 2)

        class CustomBinary(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                # Returns (batch, 5) - multi-output for binary predicate
                return torch.softmax(self.linear(x), dim=-1)

        model = CustomBinary()
        compiled = compile_logic(expr, {"P": model})

        batch_size = 3
        x = torch.randn(batch_size, 10)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify="none")

        assert result.shape == (batch_size,)

    def test_custom_binary_with_constant(self):
        """Custom binary module with constant index."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X, 2)

        class CustomClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)
                )

            def forward(self, x):
                return torch.softmax(self.layers(x), dim=-1)

        model = CustomClassifier()
        compiled = compile_logic(expr, {"P": model})

        batch_size = 5
        x = torch.randn(batch_size, 8)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify="none")

        assert result.shape == (batch_size,)

    def test_custom_binary_gradient_flow(self):
        """Gradients flow through custom binary modules."""
        X = Variable("X")
        Digit = Symbol("Digit")

        # ForAll(Y, [0,1,2], Digit(X, Y))
        from pysignet.logic.quantifier import ForAll

        Y = Variable("Y")
        expr = ForAll(Y, [0, 1, 2], Digit(X, Y))

        class DigitClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(1, 8, 3, padding=1)
                self.fc = nn.Linear(8, 10)

            def forward(self, x):
                # x: (batch, features)
                batch_size = x.shape[0]
                x = x.unsqueeze(1)  # (batch, 1, features)
                x = self.conv(x)  # (batch, 8, features)
                x = x.mean(dim=-1)  # (batch, 8)
                x = self.fc(x)  # (batch, 10)
                return torch.softmax(x, dim=-1)

        model = DigitClassifier()
        logic_loss = logic_to_loss(expr, {"Digit": model})

        x = torch.randn(1, 10)
        loss = logic_loss.loss(X=x)
        loss.backward()

        # Check gradients exist
        assert model.conv.weight.grad is not None
        assert model.fc.weight.grad is not None


class TestCustomModulesWithPredicate:
    """Test custom modules wrapped in Predicate for parameter extraction."""

    def test_predicate_wrapped_custom_module(self):
        """Predicate-wrapped custom modules work."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.tensor([1.0]))

            def forward(self, x):
                return torch.sigmoid(self.param * x.sum(dim=-1))

        model = CustomModel()
        predicates = {"P": Predicate(model)}

        logic_loss = logic_to_loss(expr, predicates)

        batch_size = 5
        x = torch.randn(batch_size, 3)
        # Use quantify='none' to get per-batch results
        result = logic_loss.loss(X=x, quantify="none")

        assert result.shape == (batch_size,)

    def test_get_trainable_parameters_from_custom_module(self):
        """Can extract parameters from custom modules."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(5))
                self.bias = nn.Parameter(torch.randn(1))

            def forward(self, x):
                return torch.sigmoid((x * self.weight).sum() + self.bias)

        model = CustomModel()
        predicates = {"P": Predicate(model)}

        compiler = TNormCompiler()
        compiled = compiler.compile(expr, predicates)
        logic_loss = LogicLoss(compiled, predicates)

        params = logic_loss.trainable_parameters
        params_list = list(params)

        # Should have weight and bias
        assert len(params_list) == 2

        # Check parameters are in the list (use id comparison)
        param_ids = {id(p) for p in params_list}
        assert id(model.weight) in param_ids
        assert id(model.bias) in param_ids


class TestMixedCustomAndSequential:
    """Test mixing custom modules with nn.Sequential."""

    def test_mixed_module_types(self):
        """Mix custom modules with Sequential modules."""
        X = Variable("X")
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X))

        # Custom module
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)

            def forward(self, x):
                return torch.sigmoid(self.fc(x).squeeze(-1))

        # Sequential module
        seq_model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

        compiled = compile_logic(expr, {"P": CustomModel(), "Q": seq_model})

        batch_size = 5
        x = torch.randn(batch_size, 10)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify="none")

        assert result.shape == (batch_size,)
