"""CombinedLoss - loss-level combination of independent LogicLoss objects.

CombinedLoss weights and sums the already-computed loss values of
several independently compiled LogicLoss objects. It never touches
compilation or satisfaction: each LogicLoss keeps its own predicates,
variables, t-norm, and post-processing.

This is NOT equivalent to conjoining expressions with `sp.And` and
compiling once. Conjoining expressions changes what "satisfied" means:
a single t-norm evaluates the whole formula together. CombinedLoss only
changes how independently computed loss values are weighted and summed.
Use `sp.And(expr_a, expr_b)` with a single `compile_logic`/`logic_to_loss`
call when constraints share predicates and variables and should be
evaluated as one formula. Use CombinedLoss when constraints are
logically independent -- different predicates, variables, batch sizes,
data streams, or even different t-norms per constraint.
"""

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn

from pysignet.loss import LogicLoss


class CombinedLoss:
    """Weight and sum the losses of several independent LogicLoss objects.

    Args:
        losses: Dict mapping constraint name to LogicLoss.
        weights: Dict mapping constraint name to a weight. Values may be
            a plain float (static) or an nn.Parameter (learnable, and
            picked up by `trainable_parameters`). If None, every
            constraint gets weight 1.0. Keys must exactly match
            `losses`.
        normalize: If True, divide the weighted sum by the sum of
            weights, keeping the scale independent of constraint count
            and weight magnitude. Default False (raw weighted sum,
            matching manual `loss_a.loss() + loss_b.loss()` when
            weights are 1.0).

    Raises:
        ValueError: If `losses` is empty, or `weights` keys do not
            exactly match `losses` keys.

    Example:
        ```python
        combined = CombinedLoss(
            {"symmetry": symmetry_loss, "addition": addition_loss},
            weights={"symmetry": 0.1, "addition": 1.0},
        )
        loss = combined.loss({
            "symmetry": {"X1": x1, "X2": x2},
            "addition": {"X": x, "Y": y},
        })
        optimizer = torch.optim.Adam(combined.trainable_parameters)
        ```
    """

    def __init__(
        self,
        losses: dict[str, LogicLoss],
        weights: dict[str, float | nn.Parameter] | None = None,
        normalize: bool = False,
    ) -> None:
        """Initialize CombinedLoss.

        Args:
            losses: Dict mapping constraint name to LogicLoss.
            weights: Dict mapping constraint name to a static float or
                learnable nn.Parameter. Defaults to 1.0 for every
                constraint.
            normalize: If True, divide the weighted sum by the sum of
                weights.
        """
        if not losses:
            raise ValueError(
                "CombinedLoss requires at least one named LogicLoss."
            )

        if weights is not None and set(weights.keys()) != set(losses.keys()):
            raise ValueError(
                f"weights keys {sorted(weights.keys())} must exactly "
                f"match losses keys {sorted(losses.keys())}."
            )

        self._losses = dict(losses)
        self._weights: dict[str, float | nn.Parameter] = (
            dict(weights) if weights is not None
            else {name: 1.0 for name in losses}
        )
        self._normalize = normalize

    def __repr__(self) -> str:
        """Return string representation of CombinedLoss.

        Returns:
            Multi-line string showing constraint names, weights, and
            the normalize flag.
        """
        names = sorted(self._losses.keys())
        weights_repr = ", ".join(
            f"{name}={self._weights[name]!r}" for name in names
        )
        parts = ["CombinedLoss("]
        parts.append(f"  losses={{{", ".join(names)}}},")
        parts.append(f"  weights={{{weights_repr}}},")
        parts.append(f"  normalize={self._normalize}")
        parts.append(")")
        return "\n".join(parts)

    def loss(
        self,
        bindings: dict[str, dict[str, torch.Tensor]],
        quantify: Literal["forall", "exists", "none"] = "forall",
        reduction: Literal["mean", "sum", "none"] = "none",
        post_processing: (
            dict[str, str | Callable[[torch.Tensor], torch.Tensor] | None]
            | None
        ) = None,
    ) -> torch.Tensor:
        """Compute the weighted sum of every constraint's loss.

        Each constraint's loss is computed by calling its own
        `LogicLoss.loss()` with the given `quantify`/`reduction` (and,
        if provided, a per-constraint `post_processing` override), then
        multiplied by its weight and summed.

        Args:
            bindings: Dict mapping constraint name to that constraint's
                variable bindings (e.g. `{"a": {"X": x}}`). Keys must
                exactly match the `losses` passed to `__init__`.
            quantify: Batch quantification mode, forwarded to every
                sub-LogicLoss: 'forall', 'exists', or 'none'.
            reduction: Reduction mode, forwarded to every sub-LogicLoss:
                'mean', 'sum', or 'none'. Only meaningful with
                `quantify='none'`.
            post_processing: Optional dict mapping constraint name to a
                post-processing override for that constraint's call. If
                a name is absent, that constraint uses its own default.

        Returns:
            Scalar tensor: the (optionally normalized) weighted sum of
            every constraint's loss.

        Raises:
            ValueError: If `bindings` keys do not exactly match the
                constraint names, if `quantify='none'` and
                `reduction='none'` (each constraint's loss would not be
                a scalar), or if `post_processing` names an unknown
                constraint.

        Example:
            ```python
            loss = combined.loss({"a": {"X": x}, "b": {"Y": y}})
            ```
        """
        if set(bindings.keys()) != set(self._losses.keys()):
            raise ValueError(
                f"bindings keys {sorted(bindings.keys())} must exactly "
                f"match losses keys {sorted(self._losses.keys())}."
            )

        if quantify == "none" and reduction == "none":
            raise ValueError(
                "CombinedLoss always returns a scalar. With "
                "quantify='none', pass reduction='mean' or 'sum' to "
                "collapse each constraint's per-batch losses first."
            )

        if post_processing is not None:
            unknown = set(post_processing.keys()) - set(self._losses.keys())
            if unknown:
                raise ValueError(
                    f"post_processing names unknown constraint(s): "
                    f"{sorted(unknown)}. Known constraints: "
                    f"{sorted(self._losses.keys())}."
                )

        total: torch.Tensor | None = None
        for name, sub_loss in self._losses.items():
            pp = post_processing.get(name) if post_processing else None
            sub_result = sub_loss.loss(
                quantify=quantify,
                reduction=reduction,
                post_processing=pp,
                **bindings[name],
            )
            weighted = self._weights[name] * sub_result
            total = weighted if total is None else total + weighted

        assert total is not None  # losses is never empty

        if self._normalize:
            weight_sum = sum(self._weights.values())
            total = total / weight_sum

        return total

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        """All trainable parameters from sub-losses and learnable weights.

        Parameters are deduplicated by identity, so a predicate or
        model shared across constraints is not double-counted (which
        would otherwise make an optimizer step it more than once per
        `optimizer.step()`).

        Returns:
            List of torch.nn.Parameter objects.

        Example:
            ```python
            params = combined.trainable_parameters
            optimizer = torch.optim.Adam(params, lr=0.001)
            ```
        """
        params: list[nn.Parameter] = []
        seen: set[int] = set()

        for sub_loss in self._losses.values():
            for param in sub_loss.trainable_parameters:
                if id(param) not in seen:
                    seen.add(id(param))
                    params.append(param)

        for weight in self._weights.values():
            if isinstance(weight, nn.Parameter) and id(weight) not in seen:
                seen.add(id(weight))
                params.append(weight)

        return params
