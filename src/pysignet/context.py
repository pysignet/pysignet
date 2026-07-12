"""Evaluation context for managing computation caching.

This module provides the EvaluationContext class which manages caching of
network outputs during a single expression evaluation to avoid redundant
forward passes.
"""

from collections.abc import Callable, Hashable
from typing import Any


class EvaluationContext:
    """Manages cache during a single expression evaluation.

    EvaluationContext provides a scoped caching mechanism that:
    - Caches network outputs to avoid redundant forward passes
    - Automatically clears after each evaluation (no cross-batch caching)
    - Preserves gradient flow through cached tensors
    - Is thread-safe (each evaluation creates its own context)

    The cache is a simple dictionary mapping cache keys to computed values.
    Cache keys are typically tuples of (network_id, inputs_id) to uniquely
    identify a computation.

    Example:
        >>> ctx = EvaluationContext()
        >>>
        >>> # First call computes
        >>> result1 = ctx.get_or_compute("key", lambda: expensive_computation())
        >>>
        >>> # Second call uses cache
        >>> result2 = ctx.get_or_compute("key", lambda: expensive_computation())
        >>>
        >>> # result1 is result2 (same cached tensor)

    Attributes:
        cache: Dictionary mapping cache keys to cached values.
    """

    def __init__(self) -> None:
        """Initialize an empty evaluation context."""
        self.cache: dict[Hashable, Any] = {}

    def get_or_compute(
        self, cache_key: Hashable, compute_fn: Callable[[], Any]
    ) -> Any:
        """Get cached value or compute and cache it.

        If the cache_key is already in the cache, returns the cached value
        without calling compute_fn. Otherwise, calls compute_fn, stores the
        result in the cache, and returns it.

        This method is the core of the caching mechanism. It ensures that
        expensive computations (like neural network forward passes) are only
        executed once per unique cache key within a single evaluation.

        Args:
            cache_key: Hashable key identifying this computation. Typically
                      a tuple like (id(network), id(inputs)).
            compute_fn: Function to call if value not cached. Should take no
                       arguments and return the computed value.

        Returns:
            The cached or newly computed value.

        Example:
            >>> ctx = EvaluationContext()
            >>> network = nn.Linear(10, 5)
            >>> inputs = torch.randn(32, 10)
            >>>
            >>> cache_key = (id(network), id(inputs))
            >>> output = ctx.get_or_compute(cache_key, lambda: network(inputs))
            >>>
            >>> # Subsequent calls with same key return cached output
            >>> output2 = ctx.get_or_compute(cache_key, lambda: network(inputs))
            >>> assert output is output2  # Same tensor object

        Note:
            The cache stores references to tensors, not copies. This is
            important for gradient flow - all uses of the cached tensor
            will contribute gradients to the original computation.
        """
        if cache_key not in self.cache:
            self.cache[cache_key] = compute_fn()
        return self.cache[cache_key]
