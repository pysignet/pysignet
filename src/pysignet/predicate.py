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

NN->Predicate Mapping
--------------------

When a neural network P predicts label y for input x, this corresponds to the
predicate P(x, y) being true. For multi-input networks with inputs x1, x2
predicting y, we have P(x1, x2, y).

Inputs are bound using keyword arguments matching variable names:
- `compiled(X=x_tensor)` for single variable
- `compiled(X=x1_tensor, Y=x2_tensor)` for multiple variables

Since both inputs and outputs can be named neurons, input-output constraints
(relating input features to output predictions) are naturally supported.
"""

import warnings
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

# Tolerance for floating-point noise in range validation
_RANGE_TOLERANCE = 1e-6


class Predicate:
    """Wraps a named neuron and maps it to a logical predicate.

    A Predicate wraps any node in a computation graph that has externally
    defined meaning (a "named neuron"). This can be:

    - **Output neuron**: Network output predicting a label
    - **Input neuron**: Input feature with semantic meaning
    - **Intermediate neuron**: Attention weights, embeddings, etc.

    The predicate maps the named neuron to logical reasoning: when network P
    predicts label y for input x, this corresponds to P(x, y) being true.

    For nn.Module predicates, the library auto-detects and applies the
    correct activation function:

    - Binary classifiers (1 output): auto-applies sigmoid if missing
    - Multiclass classifiers (N outputs): auto-applies softmax if missing
    - Modules with existing activation: passes through unchanged
    - Unknown structure: clamps to [0,1] with a warning

    For non-module predicates, values must be in [0,1] or a ValueError
    is raised.

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
    """

    def __init__(
        self,
        func: torch.nn.Module | Callable[..., Any],
        is_model: Optional[bool] = None,
    ) -> None:
        """Initialize a predicate wrapping a named neuron.

        Args:
            func: The named neuron (computation node with external
                semantics)
            is_model: True if func is trainable, False otherwise,
                None to auto-detect
        """
        self.name: Optional[str] = None  # Assigned by compiler
        self.func = func  # The named neuron being wrapped

        # Auto-detect if it's a trainable model (has parameters)
        if is_model is None:
            self.is_model = isinstance(func, torch.nn.Module)
        else:
            self.is_model = is_model

        # Activation metadata for nn.Module predicates
        self._activation: Optional[str] = None
        self._has_activation: Optional[bool] = None
        self._inferred_arity: Optional[int] = None

        if isinstance(func, torch.nn.Module):
            self._detect_activation(func)

    def _detect_activation(self, module: torch.nn.Module) -> None:
        """Detect and store activation metadata for an nn.Module.

        Inspects the module structure to determine:
        - Whether it already has a final activation layer
        - Its inferred arity (1=binary, 2=multiclass, None=unknown)
        - What activation to auto-apply (sigmoid, softmax, or none)

        Args:
            module: The nn.Module to inspect.
        """
        # Lazy import to avoid circular dependency
        # (predicate -> compilation -> base -> predicate)
        from pysignet.compilation.module_utils import (  # pylint: disable=import-outside-toplevel
            has_final_activation,
            infer_module_arity,
        )

        self._has_activation = has_final_activation(module)
        self._inferred_arity = infer_module_arity(module)

        if self._has_activation:
            # Module already has Sigmoid/Softmax -- pass through
            self._activation = None
        elif self._inferred_arity == 1:
            self._activation = "sigmoid"
        elif self._inferred_arity == 2:
            self._activation = "softmax"
        else:
            # Unknown structure -- will clamp + warn at call time
            self._activation = None

    def configure_activation(self, usage_arity: int) -> None:
        """Configure activation based on expression-context arity.

        Called by the compiler when it knows the predicate arity from
        expression usage (e.g., P(X) -> arity 1, Digit(X, Y) -> arity 2).

        This uses expression context to set activation for ALL nn.Module
        predicates that don't already have a detected final activation
        layer, including non-Sequential custom modules. Custom modules
        with internal activation (e.g., sigmoid in forward()) will get
        double-activated -- users should add activation as a final
        nn.Sigmoid()/nn.Softmax() layer for correct detection.

        Args:
            usage_arity: Arity from expression context (1 or 2).
        """
        if not isinstance(self.func, torch.nn.Module):
            return
        if self._has_activation:
            return

        if usage_arity == 1 and self._activation is None:
            self._activation = "sigmoid"
        elif usage_arity == 2 and self._activation is None:
            self._activation = "softmax"

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Evaluate the named neuron and return satisfaction degree.

        For nn.Module predicates, applies the appropriate activation:
        - sigmoid for binary classifiers without final activation
        - softmax for multiclass classifiers without final activation
        - pass-through for modules with existing activation
        - clamp + warning for unknown-structure modules

        For non-module predicates, validates that output is in [0,1]
        and raises ValueError if not.

        Args:
            *args: Positional arguments forwarded to the named neuron.
            **kwargs: Keyword arguments forwarded to the named neuron.

        Returns:
            Tensor of satisfaction degrees in [0, 1].

        Raises:
            ValueError: If non-module predicate returns values outside
                [0, 1].
        """
        result = self.func(*args, **kwargs)

        # Ensure result is a tensor
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(result, dtype=torch.float32)

        if isinstance(self.func, torch.nn.Module):
            return self._apply_module_activation(result)
        return self._validate_non_module_result(result)

    def _apply_module_activation(
        self, result: torch.Tensor
    ) -> torch.Tensor:
        """Apply activation for nn.Module predicates.

        Args:
            result: Raw module output tensor.

        Returns:
            Activated tensor with values in [0, 1].
        """
        if self._activation == "sigmoid":
            activated = torch.sigmoid(result)
            if activated.dim() >= 2 and activated.shape[-1] == 1:
                activated = activated.squeeze(-1)
            return activated

        if self._activation == "softmax":
            return torch.softmax(result, dim=-1)

        # Has existing activation or unknown structure
        if self._has_activation:
            # Pass through -- module already produces [0,1]
            if result.dim() >= 2 and result.shape[-1] == 1:
                result = result.squeeze(-1)
            return result

        # Unknown structure: clamp + warning
        if result.dim() >= 2 and result.shape[-1] == 1:
            result = result.squeeze(-1)
        warnings.warn(
            f"Predicate '{self.name}': Could not determine activation "
            f"for this nn.Module. Output will be clamped to [0, 1]. "
            f"Consider adding a final Sigmoid() or Softmax() layer, "
            f"or wrap your model with "
            f"wrap_module_as_predicate().",
            UserWarning,
            stacklevel=2,
        )
        return torch.clamp(result, 0.0, 1.0)

    def _validate_non_module_result(
        self, result: torch.Tensor
    ) -> torch.Tensor:
        """Validate and return result from non-module predicates.

        Checks that values are within [0, 1] (with small tolerance for
        float noise). Raises ValueError if values are significantly
        outside range.

        Args:
            result: Output tensor from a non-module predicate.

        Returns:
            Tensor with values in [0, 1].

        Raises:
            ValueError: If values are outside [0, 1] beyond tolerance.
        """
        # Auto-squeeze (batch, 1) to (batch,)
        if result.dim() >= 2 and result.shape[-1] == 1:
            result = result.squeeze(-1)

        if result.numel() == 0:
            return result

        min_val = result.min().item()
        max_val = result.max().item()

        if min_val < -_RANGE_TOLERANCE or max_val > 1.0 + _RANGE_TOLERANCE:
            raise ValueError(
                f"Predicate '{self.name}' returned values outside "
                f"[0, 1] range: min={min_val:.4f}, max={max_val:.4f}. "
                f"Non-module predicates must return values in [0, 1]."
            )

        # Clamp tiny float noise
        return torch.clamp(result, 0.0, 1.0)

    def _apply_module_log_activation(
        self, result: torch.Tensor
    ) -> torch.Tensor:
        """Apply log-space activation for nn.Module predicates.

        Uses fused ops (logsigmoid, log_softmax) when the activation
        would be auto-applied, avoiding the need for epsilon.

        Args:
            result: Raw module output tensor (logits).

        Returns:
            Log-activated tensor (values in (-inf, 0]).
        """
        if self._activation == "sigmoid":
            # pylint: disable=not-callable
            activated = F.logsigmoid(result)
            if activated.dim() >= 2 and activated.shape[-1] == 1:
                activated = activated.squeeze(-1)
            return activated

        if self._activation == "softmax":
            return torch.log_softmax(result, dim=-1)

        # Has existing activation or unknown structure:
        # fall back to normal activation then log
        normal = self._apply_module_activation(result)
        return torch.log(normal + 1e-10)

    def log_call(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Evaluate the named neuron and return log-satisfaction.

        Uses fused log-space ops (logsigmoid, log_softmax) when
        possible for numerical stability. Falls back to
        log(output + eps) for non-module predicates or modules
        with existing activations.

        Args:
            *args: Positional arguments forwarded to the named neuron.
            **kwargs: Keyword arguments forwarded to the named neuron.

        Returns:
            Tensor of log-satisfaction degrees in (-inf, 0].
        """
        result = self.func(*args, **kwargs)

        if not isinstance(result, torch.Tensor):
            result = torch.tensor(result, dtype=torch.float32)

        if isinstance(self.func, torch.nn.Module):
            return self._apply_module_log_activation(result)

        validated = self._validate_non_module_result(result)
        return torch.log(validated + 1e-10)

    def __repr__(self) -> str:
        """Return string representation of the predicate.

        Returns:
            String showing predicate name if assigned, otherwise indicates
            it's unnamed.
        """
        if self.name is not None:
            return f"Predicate(name='{self.name}')"
        return "Predicate(unnamed - name will be assigned by compiler)"
