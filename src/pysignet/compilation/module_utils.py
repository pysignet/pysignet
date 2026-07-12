"""nn.Module introspection and wrapping utilities.

This module provides functions for smart handling of nn.Module predicates:
1. Infer arity from output dimensionality
2. Detect existing activations (Sigmoid/Softmax)
3. Wrap modules with appropriate signatures
4. Auto-add activation only if not already present
5. Shared helpers for variable resolution and model/index splitting

Key principles:
- Single responsibility: each function does one thing
- Non-polluting: wrapper doesn't modify original module
- Explicit validation: arity must match module output dimensionality
"""

import inspect
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn as nn


def infer_module_arity(module: nn.Module) -> int | None:
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


def wrap_module_as_predicate(
    module: nn.Module, arity: int
) -> Callable[..., torch.Tensor]:
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
        arity_names: dict[int, str] = {1: "unary", 2: "binary"}
        mod_arity_name = arity_names.get(module_arity, str(module_arity))
        spec_arity_name = arity_names.get(arity, str(arity))
        out_dim = _get_output_dim(module)
        raise ValueError(
            f"Arity mismatch: module has {mod_arity_name} arity "
            f"(output dim = {out_dim}) but specified arity is "
            f"{spec_arity_name}. Ensure module output matches usage."
        )

    # Check if activation already present
    has_activation = has_final_activation(module)

    # Create appropriate wrapper based on arity
    if arity == 1:
        return _wrap_unary(module, has_activation)
    elif arity == 2:
        return _wrap_binary(module, has_activation)
    else:
        raise ValueError(
            f"Unsupported arity {arity}. "
            f"Only 1 (unary) and 2 (binary) supported."
        )


def get_module_forward_param_count(module: nn.Module) -> int:
    """Get the number of input parameters for a module's forward().

    Uses inspect.signature on the bound forward method, which
    automatically excludes 'self'.

    Args:
        module: nn.Module to inspect.

    Returns:
        Number of positional parameters, or -1 if inspection
        fails.
    """
    try:
        sig = inspect.signature(module.forward)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        return len(params)
    except (ValueError, TypeError):
        return -1


def resolve_variable_inputs(
    variables: list[Any],
    inputs: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    """Resolve variable symbols to their bound tensors.

    Args:
        variables: List of variable symbols to resolve. Each
            must be convertible to str to get its name.
        inputs: Dict mapping variable names to tensors.

    Returns:
        List of tensors in the same order as variables.

    Raises:
        ValueError: If a variable is missing from inputs.
    """
    resolved: list[torch.Tensor] = []
    for var in variables:
        var_name = str(var)
        if var_name not in inputs:
            raise ValueError(
                f"Missing input for variable '{var_name}'. "
                f"Expected key in input dict."
            )
        resolved.append(inputs[var_name])
    return resolved


def split_model_and_index_vars(
    module: nn.Module,
    free_vars: list[Any],
) -> tuple[list[Any], list[Any]]:
    """Split free variables into model inputs and index variables.

    For multiclass modules, extra variables beyond what forward()
    accepts are treated as per-element output indices. For example,
    Digit(X, Y) with a model whose forward(x) takes 1 arg splits
    into model_vars=[X] and index_vars=[Y].

    Args:
        module: nn.Module to inspect for forward() param count.
        free_vars: List of free variables from the expression.

    Returns:
        Tuple of (model_vars, index_vars).
    """
    n_forward_params = get_module_forward_param_count(module)
    if 0 < n_forward_params < len(free_vars):
        n_model_inputs = n_forward_params
    else:
        n_model_inputs = len(free_vars)

    model_vars = free_vars[:n_model_inputs]
    index_vars = free_vars[n_model_inputs:]
    return model_vars, index_vars


def _get_final_layer(module: nn.Module) -> nn.Module:
    """Get the final layer in a module's computation graph.

    Only traverses nn.Sequential modules where execution order is
    guaranteed. For non-Sequential custom modules, returns the module
    itself since children order may not reflect execution order and
    forward() may apply additional operations.

    Args:
        module: Module to inspect

    Returns:
        The last layer in the module, or the module itself if
        structure cannot be determined.

    Raises:
        ValueError: If module is empty Sequential.
    """
    if isinstance(module, nn.Sequential):
        if len(module) == 0:
            raise ValueError(
                "Cannot infer arity from empty Sequential module."
            )
        return _get_final_layer(module[-1])

    # For non-Sequential modules, return as-is -- children order
    # does not guarantee execution order.
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


def _wrap_unary(
    module: nn.Module, has_activation: bool
) -> Callable[[torch.Tensor], torch.Tensor]:
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


def _wrap_binary(
    module: nn.Module, has_activation: bool
) -> Callable[[torch.Tensor, int], torch.Tensor]:
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
            output = cast(
                torch.Tensor, module(x)
            )  # Shape: (batch, num_classes)
            return output[:, y]  # Shape: (batch,)

    else:
        # Add softmax activation
        def wrapper(x: torch.Tensor, y: int) -> torch.Tensor:
            output = cast(
                torch.Tensor, module(x)
            )  # Shape: (batch, num_classes)
            probabilities = torch.softmax(
                output, dim=-1
            )  # Shape: (batch, num_classes)
            return probabilities[:, y]  # Shape: (batch,)

    return wrapper
