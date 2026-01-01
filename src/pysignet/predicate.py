"""Predicate class for wrapping named neurons.

Named Neurons and Predicates
-----------------------------

This module is built on the concept of **named neurons**: nodes in a
computation graph that have externally defined meaning. Named neurons bridge
symbolic logic and neural network representations.

A **named neuron** is any node with external semantics (not just arbitrary
activations):

- **Network outputs**: A classifier's output predictions
- **Network inputs**: Input features with semantic meaning
- **Intermediate nodes**: Attention weights, embeddings, or other activations
  with externally assigned meaning

The `Predicate` class wraps named neurons and maps them to logical predicates.
This abstraction is foundational to the entire library.

NN→Predicate Mapping
--------------------

When a neural network P predicts label y for input x, this corresponds to the
predicate P(x, y) being true. For multi-input networks with inputs x1, x2
predicting y, we have P(x1, x2, y).

The input structure is flexible:
- Single tensor: `x_tensor` (shared across all predicates)
- Dict of tensors: `{"x1": x1_tensor, "x2": x2_tensor}` (per-predicate inputs)

Since both inputs and outputs can be named neurons, input-output constraints
(relating input features to output predictions) are naturally supported.
"""

from typing import Callable, Union, Optional, Any

import torch


class Predicate:
    """Wraps a named neuron and maps it to a logical predicate.

    A Predicate wraps any node in a computation graph that has externally
    defined meaning (a "named neuron"). This can be:

    - **Output neuron**: Network output predicting a label
    - **Input neuron**: Input feature with semantic meaning
    - **Intermediate neuron**: Attention weights, embeddings, etc.

    The predicate maps the named neuron to logical reasoning: when network P
    predicts label y for input x, this corresponds to P(x, y) being true.

    Examples:
        Output predicates (network outputs):
            >>> binary_classifier = nn.Sequential(
            ...     nn.Linear(784, 1), nn.Sigmoid()
            ... )
            >>> is_cat = Predicate(binary_classifier)

        Input predicates (input features):
            >>> is_young = Predicate(lambda x: (x['age'] < 18).float())

        Deterministic predicates (simple functions):
            >>> above_threshold = Predicate(lambda x: (x > 0.5).float())

    Args:
        func: Named neuron - torch.nn.Module or callable returning [0, 1]
        is_model: Whether the function is trainable (default: auto-detect)

    Note:
        The predicate name is automatically assigned by the compiler based on
        the dict key used when registering this predicate. The func must return
        values in [0, 1] representing the degree to which the predicate is
        satisfied. Values are automatically clamped to [0, 1].
    """

    def __init__(
        self,
        func: Union[torch.nn.Module, Callable[..., Any]],
        is_model: Optional[bool] = None
    ) -> None:
        """Initialize a predicate wrapping a named neuron.

        Args:
            func: The named neuron (computation node with external semantics)
            is_model: True if func is trainable, False otherwise, None to
                     auto-detect

        Note:
            The predicate name is automatically assigned by the compiler from
            the dict key when the predicate is registered. Before compilation,
            the name will be None.
        """
        self.name: Optional[str] = None  # Assigned by compiler
        self.func = func  # The named neuron being wrapped

        # Auto-detect if it's a trainable model (has parameters)
        if is_model is None:
            self.is_model = isinstance(func, torch.nn.Module)
        else:
            self.is_model = is_model

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Evaluate the named neuron and return satisfaction degree.

        The predicate evaluates its wrapped named neuron (network, function, or
        computation node) and returns a value in [0, 1] representing the degree
        to which the predicate is satisfied.

        Input routing:
        - If LogicCompiler receives a single tensor, it's passed to all
          predicates
        - If LogicCompiler receives a dict, each predicate gets its
          corresponding input

        Args:
            *args: Positional arguments forwarded to the named neuron
            **kwargs: Keyword arguments forwarded to the named neuron

        Returns:
            Tensor of satisfaction degrees in [0, 1]. Shape is typically
            (batch_size,) but can vary based on the named neuron's output.
        """
        # Evaluate the named neuron
        result = self.func(*args, **kwargs)

        # Ensure result is a tensor
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(result, dtype=torch.float32)

        # Clamp to [0, 1] to ensure valid satisfaction degrees
        return torch.clamp(result, 0.0, 1.0)

    def __repr__(self) -> str:
        """Return string representation of the predicate.

        Returns:
            String showing predicate name if assigned, otherwise indicates
            it's unnamed.
        """
        if self.name is not None:
            return f"Predicate(name='{self.name}')"
        return "Predicate(unnamed - name will be assigned by compiler)"
