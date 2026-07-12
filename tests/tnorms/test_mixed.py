"""Tests for MixedTNorm - uses Godel for large arities, RProduct otherwise.

MixedTNorm addresses numerical stability issues with product t-norms when
many values are combined (e.g., 0.9^10 = 0.35, gradient vanishing).
"""

import pytest
import torch

from pysignet.tnorms import GodelTNorm, MixedTNorm, RProductTNorm


class TestMixedTNormInitialization:
    """Test MixedTNorm initialization."""

    def test_default_threshold(self):
        """Test default threshold is 4."""
        tnorm = MixedTNorm()
        assert tnorm.threshold == 4

    def test_custom_threshold(self):
        """Test custom threshold."""
        tnorm = MixedTNorm(threshold=3)
        assert tnorm.threshold == 3

        tnorm2 = MixedTNorm(threshold=10)
        assert tnorm2.threshold == 10

    def test_recommended_postprocessing(self):
        """Test recommended_postprocessing returns 'log' (like RProduct)."""
        tnorm = MixedTNorm()
        assert tnorm.recommended_postprocessing == "log"


class TestMixedTNormConjunction:
    """Test conjunction (AND) behavior."""

    def test_small_arity_uses_rproduct(self):
        """Test that small arities (<=threshold) use RProduct (product)."""
        tnorm = MixedTNorm(threshold=4)
        rproduct = RProductTNorm()

        # 2 args (< threshold)
        values = torch.tensor([[0.8], [0.6]])
        result = tnorm.conjunction(values)
        expected = rproduct.conjunction(values)
        assert torch.allclose(result, expected)

        # 4 args (== threshold)
        values = torch.tensor([[0.9], [0.8], [0.7], [0.6]])
        result = tnorm.conjunction(values)
        expected = rproduct.conjunction(values)
        assert torch.allclose(result, expected)

    def test_large_arity_uses_godel(self):
        """Test that large arities (>threshold) use Godel (min)."""
        tnorm = MixedTNorm(threshold=4)
        godel = GodelTNorm()

        # 5 args (> threshold)
        values = torch.tensor([[0.9], [0.8], [0.7], [0.6], [0.5]])
        result = tnorm.conjunction(values)
        expected = godel.conjunction(values)
        assert torch.allclose(result, expected)

        # 10 args (>> threshold)
        values = torch.rand(10, 1)
        result = tnorm.conjunction(values)
        expected = godel.conjunction(values)
        assert torch.allclose(result, expected)

    def test_threshold_boundary(self):
        """Test behavior exactly at threshold boundary."""
        tnorm = MixedTNorm(threshold=3)
        rproduct = RProductTNorm()
        godel = GodelTNorm()

        # 3 args (== threshold) -> RProduct
        values = torch.tensor([[0.8], [0.7], [0.6]])
        result = tnorm.conjunction(values)
        expected = rproduct.conjunction(values)
        assert torch.allclose(result, expected)

        # 4 args (> threshold) -> Godel
        values = torch.tensor([[0.8], [0.7], [0.6], [0.5]])
        result = tnorm.conjunction(values)
        expected = godel.conjunction(values)
        assert torch.allclose(result, expected)

    def test_conjunction_batch_dimension(self):
        """Test conjunction preserves batch dimension."""
        tnorm = MixedTNorm(threshold=4)
        batch_size = 8

        # Small arity with batch
        values = torch.rand(3, batch_size)
        result = tnorm.conjunction(values)
        assert result.shape == (batch_size,)

        # Large arity with batch
        values = torch.rand(6, batch_size)
        result = tnorm.conjunction(values)
        assert result.shape == (batch_size,)


class TestMixedTNormDisjunction:
    """Test disjunction (OR) behavior."""

    def test_small_arity_uses_rproduct(self):
        """Test that small arities (<=threshold) use RProduct."""
        tnorm = MixedTNorm(threshold=4)
        rproduct = RProductTNorm()

        # 2 args
        values = torch.tensor([[0.8], [0.6]])
        result = tnorm.disjunction(values)
        expected = rproduct.disjunction(values)
        assert torch.allclose(result, expected)

        # 4 args (== threshold)
        values = torch.tensor([[0.3], [0.4], [0.5], [0.6]])
        result = tnorm.disjunction(values)
        expected = rproduct.disjunction(values)
        assert torch.allclose(result, expected)

    def test_large_arity_uses_godel(self):
        """Test that large arities (>threshold) use Godel (max)."""
        tnorm = MixedTNorm(threshold=4)
        godel = GodelTNorm()

        # 5 args (> threshold)
        values = torch.tensor([[0.3], [0.4], [0.5], [0.6], [0.7]])
        result = tnorm.disjunction(values)
        expected = godel.disjunction(values)
        assert torch.allclose(result, expected)

    def test_disjunction_batch_dimension(self):
        """Test disjunction preserves batch dimension."""
        tnorm = MixedTNorm(threshold=4)
        batch_size = 8

        # Small arity with batch
        values = torch.rand(3, batch_size)
        result = tnorm.disjunction(values)
        assert result.shape == (batch_size,)

        # Large arity with batch
        values = torch.rand(6, batch_size)
        result = tnorm.disjunction(values)
        assert result.shape == (batch_size,)


class TestMixedTNormOtherOperations:
    """Test negation, implication, equivalence."""

    def test_negation(self):
        """Test negation (inherited from TNorm base)."""
        tnorm = MixedTNorm()
        a = torch.tensor([0.3, 0.7, 0.0, 1.0])
        result = tnorm.negation(a)
        expected = torch.tensor([0.7, 0.3, 1.0, 0.0])
        assert torch.allclose(result, expected)

    def test_implication_uses_rproduct(self):
        """Test implication uses RProduct (binary operation)."""
        tnorm = MixedTNorm()
        rproduct = RProductTNorm()

        a = torch.tensor([0.8, 0.3, 0.9])
        b = torch.tensor([0.6, 0.7, 0.9])

        result = tnorm.implication(a, b)
        expected = rproduct.implication(a, b)
        assert torch.allclose(result, expected)

    def test_equivalence_uses_rproduct(self):
        """Test equivalence uses RProduct (conjunction of 2 implications)."""
        tnorm = MixedTNorm()
        rproduct = RProductTNorm()

        a = torch.tensor([0.8, 0.5, 0.3])
        b = torch.tensor([0.8, 0.7, 0.3])

        result = tnorm.equivalence(a, b)
        expected = rproduct.equivalence(a, b)
        assert torch.allclose(result, expected)


class TestMixedTNormNumericalStability:
    """Test numerical stability benefits of MixedTNorm."""

    def test_large_conjunction_stability(self):
        """Test that large conjunctions don't vanish to zero."""
        tnorm = MixedTNorm(threshold=4)

        # 20 values of 0.9 - with RProduct this would be 0.9^20 = 0.12
        # With Godel (min) it stays at 0.9
        values = torch.ones(20, 1) * 0.9
        result = tnorm.conjunction(values)

        # Should use Godel -> min = 0.9
        assert torch.allclose(result, torch.tensor([0.9]))

    def test_large_disjunction_stability(self):
        """Test that large disjunctions don't saturate to one."""
        tnorm = MixedTNorm(threshold=4)

        # 20 values of 0.1 - with RProduct this would approach 1.0
        # With Godel (max) it stays at 0.1
        values = torch.ones(20, 1) * 0.1
        result = tnorm.disjunction(values)

        # Should use Godel -> max = 0.1
        assert torch.allclose(result, torch.tensor([0.1]))


class TestMixedTNormGradients:
    """Test gradient flow through MixedTNorm."""

    def test_conjunction_gradients_small_arity(self):
        """Test gradients flow for small arity (RProduct)."""
        tnorm = MixedTNorm(threshold=4)

        values = torch.tensor([[0.8], [0.6]], requires_grad=True)
        result = tnorm.conjunction(values)
        result.sum().backward()

        # RProduct gradients: d/d(v_i) prod(v) = prod(v) / v_i
        # All inputs should have non-zero gradients
        assert values.grad is not None
        assert (values.grad != 0).all()

    def test_conjunction_gradients_large_arity(self):
        """Test gradients flow for large arity (Godel)."""
        tnorm = MixedTNorm(threshold=4)

        values = torch.tensor([[0.9], [0.8], [0.7], [0.6], [0.5]], requires_grad=True)
        result = tnorm.conjunction(values)
        result.sum().backward()

        # Godel gradients: only the min element gets gradient
        assert values.grad is not None
        # At least one gradient should be non-zero
        assert (values.grad != 0).any()

    def test_disjunction_gradients(self):
        """Test gradients flow through disjunction."""
        tnorm = MixedTNorm(threshold=4)

        # Small arity
        values = torch.tensor([[0.3], [0.4]], requires_grad=True)
        result = tnorm.disjunction(values)
        result.sum().backward()
        assert values.grad is not None
        assert (values.grad != 0).all()


class TestMixedTNormIntegration:
    """Test MixedTNorm with logic compilation."""

    def test_with_compile_logic(self):
        """Test MixedTNorm works with compile_logic."""
        import sympy as sp

        from pysignet import Symbol, Variable, logic_to_loss
        from pysignet.compilation import TNormCompiler

        X = Variable("X")
        P, Q, R = Symbol("P Q R")

        # Simple expression (small arity)
        expr = sp.And(P(X), Q(X))

        compiler = TNormCompiler(tnorm=MixedTNorm())
        predicates = {
            "P": lambda x: torch.ones(x.shape[0]) * 0.8,
            "Q": lambda x: torch.ones(x.shape[0]) * 0.6,
            "R": lambda x: torch.ones(x.shape[0]) * 0.7,
        }

        compiled = compiler.compile(expr, predicates)
        x = torch.randn(4, 10)
        result = compiled(X=x)

        assert result.shape == (4,)
        # Small arity -> RProduct: 0.8 * 0.6 = 0.48
        assert torch.allclose(result, torch.ones(4) * 0.48, atol=1e-5)

    def test_with_large_expression(self):
        """Test MixedTNorm with large conjunction."""
        import sympy as sp

        from pysignet import Symbol, Variable
        from pysignet.compilation import TNormCompiler

        X = Variable("X")
        # Create 6 predicates (> threshold of 4)
        P1, P2, P3, P4, P5, P6 = Symbol("P1 P2 P3 P4 P5 P6")

        expr = sp.And(P1(X), P2(X), P3(X), P4(X), P5(X), P6(X))

        compiler = TNormCompiler(tnorm=MixedTNorm(threshold=4))
        predicates = {
            "P1": lambda x: torch.ones(x.shape[0]) * 0.9,
            "P2": lambda x: torch.ones(x.shape[0]) * 0.8,
            "P3": lambda x: torch.ones(x.shape[0]) * 0.7,
            "P4": lambda x: torch.ones(x.shape[0]) * 0.6,
            "P5": lambda x: torch.ones(x.shape[0]) * 0.5,
            "P6": lambda x: torch.ones(x.shape[0]) * 0.4,
        }

        compiled = compiler.compile(expr, predicates)
        x = torch.randn(4, 10)
        result = compiled(X=x)

        assert result.shape == (4,)
        # Large arity -> Godel: min(0.9, 0.8, 0.7, 0.6, 0.5, 0.4) = 0.4
        assert torch.allclose(result, torch.ones(4) * 0.4, atol=1e-5)
