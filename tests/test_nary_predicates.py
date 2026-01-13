"""Tests for n-ary predicates (ternary, quaternary, etc.).

N-ary predicates are implemented using lambdas with multiple arguments.
Tests verify that gradients flow correctly and that PyTorch goodness works.
"""

import pytest
import torch
import torch.nn as nn
import sympy as sp

from pysignet import Symbol, Variable, compile_logic
from pysignet.logic.quantifier import ForAll


class TestTernaryPredicates:
    """Test ternary predicates (3 arguments)."""

    def test_ternary_predicate_basic(self):
        """Basic ternary predicate evaluation."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")
        expr = P(X, Y, Z)

        # Ternary predicate: takes 3 inputs
        def ternary_func(x, y, z):
            # x, y, z are all tensors or constants
            # Return batch tensor
            batch_size = x.shape[0] if isinstance(x, torch.Tensor) else y.shape[0]
            return torch.ones(batch_size) * 0.8

        logic_loss = compile_logic(expr, {"P": ternary_func})

        x = torch.randn(5, 10)
        y = torch.randn(5, 8)
        z = torch.randn(5, 6)

        result = logic_loss({"X": x, "Y": y, "Z": z})

        assert result.shape == (5,)
        assert torch.allclose(result, torch.ones(5) * 0.8)

    def test_ternary_with_two_variables_one_constant(self):
        """Ternary predicate with two variables and one constant."""
        X, Y = Variable("X Y")
        P = Symbol("P")
        expr = P(X, Y, 0)

        # Ternary model backed by neural network
        model = nn.Sequential(nn.Linear(10, 12), nn.Softmax(dim=-1))

        def ternary_pred(x, y, z):
            # x, y are variable inputs (tensors)
            # z is constant (int)
            # Concatenate inputs and select output channel
            batch_size = x.shape[0]
            combined = torch.cat([x, y], dim=-1)  # (batch, 10+8)
            output = model(combined)  # (batch, 12)

            # z is the channel index
            if isinstance(z, int):
                return output[:, z]
            else:
                # z might be a tensor of indices
                return torch.gather(output, 1, z.unsqueeze(-1)).squeeze(-1)

        # Adjust model input size
        model[0] = nn.Linear(18, 12)  # 10 + 8 features

        logic_loss = compile_logic(expr, {"P": ternary_pred})

        x = torch.randn(3, 10)
        y = torch.randn(3, 8)

        result = logic_loss({"X": x, "Y": y})

        assert result.shape == (3,)

    def test_ternary_gradient_flow(self):
        """Gradients flow through ternary predicates."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")
        expr = P(X, Y, Z)

        # Ternary predicate with trainable parameters
        model_x = nn.Linear(5, 8)
        model_y = nn.Linear(4, 8)
        model_z = nn.Linear(3, 8)
        combiner = nn.Linear(24, 1)

        def ternary_pred(x, y, z):
            # Process each input through its model
            feat_x = model_x(x)
            feat_y = model_y(y)
            feat_z = model_z(z)

            # Combine features
            combined = torch.cat([feat_x, feat_y, feat_z], dim=-1)
            output = torch.sigmoid(combiner(combined).squeeze(-1))
            return output

        logic_loss = compile_logic(expr, {"P": ternary_pred})

        x = torch.randn(10, 5)
        y = torch.randn(10, 4)
        z = torch.randn(10, 3)

        loss = logic_loss.loss({"X": x, "Y": y, "Z": z})
        loss.backward()

        # Check gradients flow to all models
        assert model_x.weight.grad is not None
        assert model_y.weight.grad is not None
        assert model_z.weight.grad is not None
        assert combiner.weight.grad is not None

    def test_ternary_with_quantifier(self):
        """Ternary predicates with quantifiers."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        # ForAll(Y, [0, 1], ForAll(Z, [2, 3], P(X, Y, Z)))
        inner = ForAll(Z, [2, 3], P(X, Y, Z))
        expr = ForAll(Y, [0, 1], inner)

        # Ternary predicate backed by model
        model = nn.Sequential(nn.Linear(5, 12), nn.Softmax(dim=-1))

        def ternary_pred(x, y, z):
            # x is variable input (tensor)
            # y, z are constants (ints from quantifiers)
            output = model(x)  # (batch, 12)

            # Compute index from y and z
            # y ∈ {0, 1}, z ∈ {2, 3}
            # Map to indices: (0,2)→0, (0,3)→1, (1,2)→2, (1,3)→3
            idx = y * 2 + (z - 2)

            return output[:, idx]

        logic_loss = compile_logic(expr, {"P": ternary_pred})

        x = torch.randn(3, 5)
        result = logic_loss(x)

        assert result.shape == (3,)


class TestQuaternaryPredicates:
    """Test quaternary predicates (4 arguments)."""

    def test_quaternary_predicate_basic(self):
        """Basic quaternary predicate evaluation."""
        W, X, Y, Z = Variable("W X Y Z")
        P = Symbol("P")
        expr = P(W, X, Y, Z)

        def quaternary_func(w, x, y, z):
            batch_size = w.shape[0]
            # Simple aggregation
            score = (w.mean(dim=-1) + x.mean(dim=-1) +
                     y.mean(dim=-1) + z.mean(dim=-1)) / 4
            return torch.sigmoid(score)

        logic_loss = compile_logic(expr, {"P": quaternary_func})

        w = torch.randn(4, 5)
        x = torch.randn(4, 5)
        y = torch.randn(4, 5)
        z = torch.randn(4, 5)

        result = logic_loss({"W": w, "X": x, "Y": y, "Z": z})

        assert result.shape == (4,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_quaternary_gradient_flow(self):
        """Gradients flow through quaternary predicates."""
        W, X, Y, Z = Variable("W X Y Z")
        P = Symbol("P")
        expr = P(W, X, Y, Z)

        # Four separate models, one for each input
        model_w = nn.Linear(3, 4)
        model_x = nn.Linear(3, 4)
        model_y = nn.Linear(3, 4)
        model_z = nn.Linear(3, 4)
        final = nn.Linear(16, 1)

        def quaternary_pred(w, x, y, z):
            feat_w = model_w(w)
            feat_x = model_x(x)
            feat_y = model_y(y)
            feat_z = model_z(z)

            combined = torch.cat([feat_w, feat_x, feat_y, feat_z], dim=-1)
            return torch.sigmoid(final(combined).squeeze(-1))

        logic_loss = compile_logic(expr, {"P": quaternary_pred})

        w = torch.randn(5, 3)
        x = torch.randn(5, 3)
        y = torch.randn(5, 3)
        z = torch.randn(5, 3)

        loss = logic_loss.loss({"W": w, "X": x, "Y": y, "Z": z})
        loss.backward()

        # Verify gradients exist for all models
        assert model_w.weight.grad is not None
        assert model_x.weight.grad is not None
        assert model_y.weight.grad is not None
        assert model_z.weight.grad is not None
        assert final.weight.grad is not None

    def test_quaternary_training(self):
        """Can train through quaternary predicates."""
        W, X, Y, Z = Variable("W X Y Z")
        P = Symbol("P")
        expr = P(W, X, Y, Z)

        model = nn.Linear(12, 1)  # 4 inputs * 3 features each

        def quaternary_pred(w, x, y, z):
            combined = torch.cat([w, x, y, z], dim=-1)
            return torch.sigmoid(model(combined).squeeze(-1))

        logic_loss = compile_logic(expr, {"P": quaternary_pred})

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        initial_weight = model.weight.data.clone()

        # Training step
        w = torch.randn(10, 3)
        x = torch.randn(10, 3)
        y = torch.randn(10, 3)
        z = torch.randn(10, 3)

        loss = logic_loss.loss({"W": w, "X": x, "Y": y, "Z": z})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Weights should change
        assert not torch.allclose(model.weight.data, initial_weight)


class TestHighArityPredicates:
    """Test predicates with even higher arity (5+)."""

    def test_quinary_predicate(self):
        """Test 5-argument predicate."""
        vars = Variable("V1 V2 V3 V4 V5")
        V1, V2, V3, V4, V5 = vars
        P = Symbol("P")
        expr = P(V1, V2, V3, V4, V5)

        def quinary_func(v1, v2, v3, v4, v5):
            # Average across all inputs
            avg = (v1.mean(dim=-1) + v2.mean(dim=-1) + v3.mean(dim=-1) +
                   v4.mean(dim=-1) + v5.mean(dim=-1)) / 5
            return torch.sigmoid(avg)

        logic_loss = compile_logic(expr, {"P": quinary_func})

        inputs = {f"V{i}": torch.randn(3, 2) for i in range(1, 6)}
        result = logic_loss(inputs)

        assert result.shape == (3,)

    def test_high_arity_with_mixed_constants(self):
        """High arity predicate with some constant arguments."""
        X, Y = Variable("X Y")
        P = Symbol("P")
        # P(X, Y, 0, 1, 2) - 5 arguments, last 3 are constants
        expr = P(X, Y, 0, 1, 2)

        model = nn.Sequential(nn.Linear(10, 8), nn.Softmax(dim=-1))

        def high_arity_pred(x, y, c1, c2, c3):
            # Combine variable inputs
            combined = torch.cat([x, y], dim=-1)
            output = model(combined)

            # Use constants to compute index
            idx = c1 * 4 + c2 * 2 + c3  # 0*4 + 1*2 + 2 = 4
            return output[:, idx]

        # Adjust model input size
        model[0] = nn.Linear(8, 8)  # 5 + 3 features

        logic_loss = compile_logic(expr, {"P": high_arity_pred})

        x = torch.randn(4, 5)
        y = torch.randn(4, 3)

        result = logic_loss({"X": x, "Y": y})

        assert result.shape == (4,)


class TestNaryPredicatesWithComplexExpressions:
    """Test n-ary predicates in complex logical expressions."""

    def test_ternary_with_and_or(self):
        """Ternary predicates in AND/OR expressions."""
        X, Y, Z = Variable("X Y Z")
        P, Q = Symbol("P Q")

        expr = sp.And(P(X, Y, Z), Q(X, Y, Z))

        def ternary_p(x, y, z):
            return torch.sigmoid((x + y + z).mean(dim=-1))

        def ternary_q(x, y, z):
            return torch.sigmoid((x * 2 + y - z).mean(dim=-1))

        logic_loss = compile_logic(expr, {"P": ternary_p, "Q": ternary_q})

        x = torch.randn(5, 4)
        y = torch.randn(5, 4)
        z = torch.randn(5, 4)

        result = logic_loss({"X": x, "Y": y, "Z": z})

        assert result.shape == (5,)

    def test_mixed_arity_predicates(self):
        """Mix predicates of different arities."""
        X, Y, Z = Variable("X Y Z")
        P, Q, R = Symbol("P Q R")

        # P(X) - unary, Q(X, Y) - binary, R(X, Y, Z) - ternary
        expr = sp.And(P(X), sp.And(Q(X, Y), R(X, Y, Z)))

        unary_model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())

        def unary_pred(x):
            return unary_model(x).squeeze(-1)

        binary_model = nn.Sequential(nn.Linear(8, 3), nn.Softmax(dim=-1))

        def binary_pred(x, y):
            combined = torch.cat([x, y], dim=-1)
            # Adjust model
            binary_model[0] = nn.Linear(8, 3)
            output = binary_model(combined)
            return output[:, 0]  # Select first channel

        def ternary_pred(x, y, z):
            feat = torch.cat([x, y, z], dim=-1)
            return torch.sigmoid(feat.mean(dim=-1))

        logic_loss = compile_logic(expr, {
            "P": unary_pred,
            "Q": binary_pred,
            "R": ternary_pred
        })

        x = torch.randn(3, 5)
        y = torch.randn(3, 3)
        z = torch.randn(3, 2)

        result = logic_loss({"X": x, "Y": y, "Z": z})

        assert result.shape == (3,)
