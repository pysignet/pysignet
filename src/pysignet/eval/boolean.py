"""Soft-to-boolean conversion functions.

Provides functions for converting soft neural network outputs to
hard boolean decisions using appropriate thresholding rules:

- Binary (sigmoid output, shape (batch,)): threshold at 0.5
- Multiclass (softmax output, shape (batch, C)): argmax == class_idx
- Others: threshold at 0.5
"""


import torch


def to_boolean(
    output: torch.Tensor,
    class_idx: int | torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert soft output to a boolean decision tensor.

    Conversion rules:
    - Already boolean: return as-is.
    - Shape (batch, 1): squeeze to (batch,), threshold at 0.5.
    - Shape (batch, C) with C > 1 and class_idx given:
      argmax along last dim == class_idx.
    - Shape (batch, C) with C > 1 and no class_idx:
      max along last dim > 0.5.
    - Shape (batch,) or scalar: threshold at 0.5.

    Args:
        output: Soft prediction tensor with values in [0, 1].
        class_idx: Optional class index for multiclass outputs.
            Can be an int (same class for all examples) or a
            tensor of per-element class indices. When provided
            and output is 2D with multiple columns, uses argmax
            comparison instead of thresholding.

    Returns:
        Boolean tensor of shape (batch,).
    """
    if output.dtype == torch.bool:
        return output

    # Squeeze (batch, 1) -> (batch,)
    if output.dim() >= 2 and output.shape[-1] == 1:
        output = output.squeeze(-1)

    # Multiclass: (batch, C) with C > 1
    if output.dim() >= 2 and output.shape[-1] > 1:
        if class_idx is not None:
            return output.argmax(dim=-1) == class_idx
        return output.max(dim=-1).values > 0.5

    # Binary / scalar: threshold at 0.5
    return output > 0.5
