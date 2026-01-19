"""nn.Module introspection and wrapping utilities.

This module provides functions for smart handling of nn.Module predicates:
1. Infer arity from output dimensionality
2. Detect existing activations (Sigmoid/Softmax)
3. Wrap modules with appropriate signatures
4. Auto-add activation only if not already present

Key principles:
- Single responsibility: each function does one thing
- Non-polluting: wrapper doesn't modify original module
- Explicit validation: arity must match module output dimensionality
"""

from typing import Any, Callable, Dict, Optional, cast

import torch
import torch.nn as nn


def infer_module_arity(module: nn.Module) -> Optional[int]:
    """Infer predicate arity from module output dimensionality.

    Args:
        module: PyTorch module to inspect

    Returns:
        1 for unary predicates (output dim = 1)
        2 for binary predicates (output dim > 1)
        None if arity cannot be inferred (custom modules)

    Rules:
        - Linear(*, 1) → arity 1 (unary)
        - Linear(*, N>1) → arity 2 (binary)
        - Sigmoid() → arity 1 (unary)
        - Softmax() → arity 2 (binary)
        - Custom modules → None (arity inferred from expression usage)

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
        >>> infer_module_arity(model)
        1
        >>> model = nn.Sequential(nn.Linear(10, 3), nn.Softmax(dim=-1))
        >>> infer_module_arity(model)
        2
        >>> class CustomModel(nn.Module):
        ...     pass
        >>> infer_module_arity(CustomModel())
        None
    """
    final_layer = _get_final_layer(module)

    # Check layer type and infer arity
    if isinstance(final_layer, nn.Linear):
        out_features = final_layer.out_features
        return 1 if out_features == 1 else 2

    elif isinstance(final_layer, nn.Sigmoid):
        return 1

    elif isinstance(final_layer, nn.Softmax):
        return 2

    else:
        # Cannot infer arity for custom modules - return None
        # Arity will be inferred from expression usage
        return None


def has_final_activation(module: nn.Module) -> bool:
    """Check if module ends with Sigmoid or Softmax activation.

    Args:
        module: PyTorch module to inspect

    Returns:
        True if final layer is Sigmoid or Softmax, False otherwise

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
        >>> has_final_activation(model)
        True
        >>> model = nn.Sequential(nn.Linear(10, 1))
        >>> has_final_activation(model)
        False
    """
    final_layer = _get_final_layer(module)
    return isinstance(final_layer, (nn.Sigmoid, nn.Softmax))


def wrap_module_as_predicate(module: nn.Module, arity: int) -> Callable[..., torch.Tensor]:
    """Wrap nn.Module as predicate with appropriate signature.

    Creates a callable wrapper that:
    - Matches the specified arity (1 for unary, 2 for binary)
    - Adds activation (sigmoid/softmax) if not already present
    - Returns properly shaped output: (batch,) tensor

    Args:
        module: PyTorch module to wrap
        arity: Expected arity (1 for unary, 2 for binary)

    Returns:
        Callable with appropriate signature:
        - Arity 1: lambda x: ...  # Returns (batch,)
        - Arity 2: lambda x, y: ...  # Returns (batch,)

    Raises:
        ValueError: If module arity doesn't match specified arity

    Example:
        >>> # Unary predicate (single output)
        >>> model = nn.Sequential(nn.Linear(10, 1))
        >>> wrapper = wrap_module_as_predicate(model, arity=1)
        >>> x = torch.randn(32, 10)
        >>> output = wrapper(x)  # Shape: (32,)

        >>> # Binary predicate (multiple outputs)
        >>> model = nn.Sequential(nn.Linear(10, 3))
        >>> wrapper = wrap_module_as_predicate(model, arity=2)
        >>> output = wrapper(x, 1)  # Select class 1, shape: (32,)
    """
    # Validate arity matches module
    module_arity = infer_module_arity(module)
    if module_arity is not None and module_arity != arity:
        arity_names: Dict[int, str] = {1: "unary", 2: "binary"}
        raise ValueError(
            f"Arity mismatch: module has {arity_names.get(module_arity, str(module_arity))} arity "
            f"(output dim = {_get_output_dim(module)}) but specified arity is "
            f"{arity_names.get(arity, str(arity))}. Ensure module output dimensionality matches usage."
        )

    # Check if activation already present
    has_activation = has_final_activation(module)

    # Create appropriate wrapper based on arity
    if arity == 1:
        return _wrap_unary(module, has_activation)
    elif arity == 2:
        return _wrap_binary(module, has_activation)
    else:
        raise ValueError(f"Unsupported arity {arity}. Only 1 (unary) and 2 (binary) supported.")


def _get_final_layer(module: nn.Module) -> nn.Module:
    """Get the final layer in a module's computation graph.

    Args:
        module: Module to inspect

    Returns:
        The last layer in the module

    Raises:
        ValueError: If module is empty or has no recognizable structure
    """
    if isinstance(module, nn.Sequential):
        if len(module) == 0:
            raise ValueError("Cannot infer arity from empty Sequential module.")
        return _get_final_layer(module[-1])

    # For custom modules, check if they have children
    children = list(module.children())
    if children:
        # Recursively get final layer of last child
        return _get_final_layer(children[-1])

    # This is a leaf layer - return it
    return module


def _get_output_dim(module: nn.Module) -> int:
    """Get output dimensionality of module.

    Args:
        module: Module to inspect

    Returns:
        Output feature dimension
    """
    final_layer = _get_final_layer(module)

    if isinstance(final_layer, nn.Linear):
        return final_layer.out_features
    elif isinstance(final_layer, nn.Sigmoid):
        return 1
    elif isinstance(final_layer, nn.Softmax):
        # Softmax doesn't change dimensionality, need to check previous layer
        # For now, assume > 1 (this is validated elsewhere)
        return 2  # Placeholder
    else:
        return 0


def _wrap_unary(module: nn.Module, has_activation: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create unary predicate wrapper (arity 1).

    Args:
        module: Module to wrap
        has_activation: Whether module already has Sigmoid activation

    Returns:
        Callable: lambda x: ... returning (batch,) tensor
    """
    if has_activation:
        # Already has sigmoid - just squeeze
        def wrapper(x: torch.Tensor) -> torch.Tensor:
            output = cast(torch.Tensor, module(x))
            # Squeeze last dimension if (batch, 1)
            if output.dim() > 1 and output.shape[-1] == 1:
                return output.squeeze(-1)
            return output
    else:
        # Add sigmoid activation
        def wrapper(x: torch.Tensor) -> torch.Tensor:
            output = cast(torch.Tensor, module(x))
            # Apply sigmoid
            activated = torch.sigmoid(output)
            # Squeeze last dimension if (batch, 1)
            if activated.dim() > 1 and activated.shape[-1] == 1:
                return activated.squeeze(-1)
            return activated

    return wrapper


def _wrap_binary(module: nn.Module, has_activation: bool) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """Create binary predicate wrapper (arity 2).

    Args:
        module: Module to wrap
        has_activation: Whether module already has Softmax activation

    Returns:
        Callable: lambda x, y: ... returning (batch,) tensor
    """
    if has_activation:
        # Already has softmax - just select class
        def wrapper(x: torch.Tensor, y: int) -> torch.Tensor:
            output = cast(torch.Tensor, module(x))  # Shape: (batch, num_classes)
            return output[:, y]  # Shape: (batch,)
    else:
        # Add softmax activation
        def wrapper(x: torch.Tensor, y: int) -> torch.Tensor:
            output = cast(torch.Tensor, module(x))  # Shape: (batch, num_classes)
            probabilities = torch.softmax(output, dim=-1)  # Shape: (batch, num_classes)
            return probabilities[:, y]  # Shape: (batch,)

    return wrapper
