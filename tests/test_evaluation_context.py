"""Tests for EvaluationContext - caching mechanism for multi-output networks.

This test file follows TDD: tests are written BEFORE implementation.
All tests should FAIL initially, then pass after implementation.
"""

import pytest
import torch
import torch.nn as nn

# This import will fail initially - that's expected in TDD RED phase
from pysignet.context import EvaluationContext


class TestEvaluationContextBasic:
    """Test basic EvaluationContext functionality."""

    def test_create_context(self):
        """Test creating an EvaluationContext."""
        ctx = EvaluationContext()
        assert ctx is not None
        assert hasattr(ctx, 'cache')

    def test_empty_cache_initially(self):
        """Test that cache is empty when created."""
        ctx = EvaluationContext()
        assert len(ctx.cache) == 0


class TestGetOrCompute:
    """Test the get_or_compute caching mechanism."""

    def test_compute_on_first_call(self):
        """Test that function is called on first access."""
        ctx = EvaluationContext()
        call_count = 0

        def compute_fn():
            nonlocal call_count
            call_count += 1
            return torch.tensor([1.0, 2.0, 3.0])

        cache_key = "test_key"
        result = ctx.get_or_compute(cache_key, compute_fn)

        assert call_count == 1
        assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_use_cache_on_second_call(self):
        """Test that cached value is used on subsequent calls."""
        ctx = EvaluationContext()
        call_count = 0

        def compute_fn():
            nonlocal call_count
            call_count += 1
            return torch.tensor([1.0, 2.0, 3.0])

        cache_key = "test_key"

        # First call
        result1 = ctx.get_or_compute(cache_key, compute_fn)
        assert call_count == 1

        # Second call - should use cache
        result2 = ctx.get_or_compute(cache_key, compute_fn)
        assert call_count == 1  # Still 1, not 2!

        # Results should be the same tensor
        assert result1 is result2

    def test_different_keys_compute_separately(self):
        """Test that different keys trigger separate computations."""
        ctx = EvaluationContext()
        call_count = 0

        def compute_fn():
            nonlocal call_count
            call_count += 1
            return torch.tensor([call_count])

        # Different keys
        result1 = ctx.get_or_compute("key1", compute_fn)
        result2 = ctx.get_or_compute("key2", compute_fn)

        assert call_count == 2  # Called twice
        assert result1[0] == 1
        assert result2[0] == 2

    def test_cache_preserves_tensors(self):
        """Test that tensors are preserved correctly in cache."""
        ctx = EvaluationContext()

        def create_tensor():
            return torch.randn(1, 10)

        cache_key = "tensor_key"
        tensor1 = ctx.get_or_compute(cache_key, create_tensor)
        tensor2 = ctx.get_or_compute(cache_key, create_tensor)

        # Should be the exact same tensor object
        assert tensor1 is tensor2
        assert torch.equal(tensor1, tensor2)


class TestCacheIsolation:
    """Test that different contexts don't share caches."""

    def test_separate_contexts_separate_caches(self):
        """Test that two contexts have independent caches."""
        ctx1 = EvaluationContext()
        ctx2 = EvaluationContext()

        call_count1 = 0
        call_count2 = 0

        def compute_fn1():
            nonlocal call_count1
            call_count1 += 1
            return torch.tensor([1.0])

        def compute_fn2():
            nonlocal call_count2
            call_count2 += 1
            return torch.tensor([2.0])

        cache_key = "same_key"

        # Use same key in different contexts
        result1 = ctx1.get_or_compute(cache_key, compute_fn1)
        result2 = ctx2.get_or_compute(cache_key, compute_fn2)

        # Both should have computed
        assert call_count1 == 1
        assert call_count2 == 1

        # Results should be different
        assert result1[0] == 1.0
        assert result2[0] == 2.0

    def test_context_cache_not_shared(self):
        """Test that contexts don't interfere with each other."""
        ctx1 = EvaluationContext()
        ctx2 = EvaluationContext()

        # Add to ctx1 cache
        ctx1.get_or_compute("key1", lambda: torch.tensor([1.0]))

        # ctx2 should not have this key
        assert "key1" not in ctx2.cache


class TestCacheKeys:
    """Test cache key behavior."""

    def test_same_key_same_result(self):
        """Test that same key returns same cached result."""
        ctx = EvaluationContext()

        def create_random_tensor():
            return torch.randn(10)

        cache_key = "random"

        result1 = ctx.get_or_compute(cache_key, create_random_tensor)
        result2 = ctx.get_or_compute(cache_key, create_random_tensor)

        # Should be identical (cached)
        assert torch.equal(result1, result2)

    def test_tuple_cache_keys(self):
        """Test that tuple keys work correctly."""
        ctx = EvaluationContext()

        result1 = ctx.get_or_compute((1, 2), lambda: torch.tensor([1.0]))
        result2 = ctx.get_or_compute((1, 2), lambda: torch.tensor([2.0]))

        # Should use cache
        assert torch.equal(result1, result2)
        assert result1[0] == 1.0  # Not 2.0

    def test_different_tuple_keys(self):
        """Test that different tuples create different cache entries."""
        ctx = EvaluationContext()

        result1 = ctx.get_or_compute((1, 2), lambda: torch.tensor([1.0]))
        result2 = ctx.get_or_compute((1, 3), lambda: torch.tensor([2.0]))

        # Should be different
        assert result1[0] == 1.0
        assert result2[0] == 2.0

    def test_id_based_cache_keys(self):
        """Test cache keys based on object identity."""
        ctx = EvaluationContext()

        obj1 = nn.Linear(10, 5)
        obj2 = nn.Linear(10, 5)

        inputs = torch.randn(1, 10)

        # Cache key based on id
        key1 = (id(obj1), id(inputs))
        key2 = (id(obj2), id(inputs))

        result1 = ctx.get_or_compute(key1, lambda: obj1(inputs))
        result2 = ctx.get_or_compute(key2, lambda: obj2(inputs))

        # Different objects, different results
        assert not torch.equal(result1, result2)


class TestGradientPreservation:
    """Test that cached tensors preserve gradient information."""

    def test_cached_tensor_preserves_gradients(self):
        """Test that gradients flow through cached tensors."""
        ctx = EvaluationContext()

        network = nn.Linear(10, 5)
        inputs = torch.randn(1, 10, requires_grad=True)

        def compute():
            return network(inputs)

        cache_key = "network_output"

        # First call - compute and cache
        output1 = ctx.get_or_compute(cache_key, compute)

        # Second call - use cache
        output2 = ctx.get_or_compute(cache_key, compute)

        # Compute loss from cached output
        loss = output2.sum()
        loss.backward()

        # Gradients should flow back to inputs and network
        assert inputs.grad is not None
        assert network.weight.grad is not None

    def test_multiple_uses_accumulate_gradients(self):
        """Test that using cached tensor multiple times accumulates gradients."""
        ctx = EvaluationContext()

        network = nn.Linear(10, 5)
        inputs = torch.randn(1, 10, requires_grad=True)

        def compute():
            return network(inputs)

        cache_key = "output"

        # Get cached output
        output = ctx.get_or_compute(cache_key, compute)

        # Use it twice in loss
        loss1 = output[:, 0].sum()
        loss2 = output[:, 1].sum()
        total_loss = loss1 + loss2

        total_loss.backward()

        # Gradients should have accumulated
        assert inputs.grad is not None
        assert network.weight.grad is not None


class TestCacheSizeAndMemory:
    """Test cache size and memory behavior."""

    def test_cache_grows_with_entries(self):
        """Test that cache size increases with entries."""
        ctx = EvaluationContext()

        for i in range(10):
            ctx.get_or_compute(f"key_{i}", lambda i=i: torch.tensor([i]))

        assert len(ctx.cache) == 10

    def test_cache_stores_references(self):
        """Test that cache stores tensor references, not copies."""
        ctx = EvaluationContext()

        original = torch.tensor([1.0, 2.0, 3.0])

        result = ctx.get_or_compute("key", lambda: original)

        # Should be the same object
        assert result is original


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_cache_with_none_value(self):
        """Test caching None value."""
        ctx = EvaluationContext()

        result = ctx.get_or_compute("none_key", lambda: None)
        assert result is None

        # Should use cached None
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return None

        ctx.get_or_compute("none_key", compute)
        assert call_count == 0  # Should use cache

    def test_cache_with_empty_tensor(self):
        """Test caching empty tensor."""
        ctx = EvaluationContext()

        empty = torch.tensor([])
        result = ctx.get_or_compute("empty", lambda: empty)

        assert result.numel() == 0
        assert result is empty

    def test_compute_function_raises_exception(self):
        """Test behavior when compute function raises exception."""
        ctx = EvaluationContext()

        def failing_compute():
            raise ValueError("Computation failed")

        with pytest.raises(ValueError, match="Computation failed"):
            ctx.get_or_compute("fail_key", failing_compute)

        # Failed computation should not be cached
        assert "fail_key" not in ctx.cache


class TestRealWorldUsage:
    """Test realistic usage patterns."""

    def test_multioutput_network_caching(self):
        """Test caching pattern for multi-output network."""
        ctx = EvaluationContext()

        # Multi-output network
        network = nn.Sequential(
            nn.Linear(10, 5),
            nn.Softmax(dim=-1)
        )

        inputs = torch.randn(1, 10)

        # Cache key based on network and inputs
        cache_key = (id(network), id(inputs))

        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return network(inputs)

        # First access
        full_output = ctx.get_or_compute(cache_key, compute)
        output_0 = full_output[:, 0]

        # Second access (different slice)
        full_output2 = ctx.get_or_compute(cache_key, compute)
        output_1 = full_output2[:, 1]

        # Should only compute once
        assert call_count == 1

        # Both slices should come from same computation
        assert full_output is full_output2

    def test_expression_evaluation_pattern(self):
        """Test typical expression evaluation caching pattern."""
        ctx = EvaluationContext()

        classifier = nn.Linear(10, 3)
        inputs = torch.randn(1, 10)

        def evaluate_classifier():
            return torch.softmax(classifier(inputs), dim=-1)

        cache_key = (id(classifier), id(inputs))

        # Simulate expression: AND(Digit(X,0), Digit(X,1), Digit(X,2))
        # All three should share the same computation (same input X)
        full_output = ctx.get_or_compute(cache_key, evaluate_classifier)
        digit_0 = full_output[:, 0]

        full_output = ctx.get_or_compute(cache_key, evaluate_classifier)
        digit_1 = full_output[:, 1]

        full_output = ctx.get_or_compute(cache_key, evaluate_classifier)
        digit_2 = full_output[:, 2]

        # Should all come from the same tensor
        # With batch_size=1, shapes are (1,)
        assert digit_0.shape == (1,)
        assert digit_1.shape == (1,)
        assert digit_2.shape == (1,)
