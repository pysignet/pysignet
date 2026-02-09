"""Consistency metrics for neural models.

Provides ConsistencyReport, which accumulates satisfaction
results across batches to compute global and conditional
violation rates.

Based on Li et al. (2019), "Logic-Driven Context Extension and
Data Augmentation for Logical Reasoning of Text",
arXiv:1909.00126.

Metrics:
    Global violation (rho): Fraction of examples that violate
        the formula. rho = violated / total.
    Conditional violation (tau): Fraction of examples that
        violate the formula among those where the antecedent
        holds. Filters out vacuously true implications.

The antecedent is auto-detected: for Implies(A, B) it is A;
for any other formula it is sp.true (all examples count),
so tau equals rho.

Supports both single-expression and multi-constraint modes:
    Single: ConsistencyReport(expr, predicates)
    Multi:  ConsistencyReport({"name": expr, ...}, predicates)

Example:
    >>> from pysignet.eval import ConsistencyReport
    >>> report = ConsistencyReport(expr, predicates)
    >>> for x_batch in dataloader:
    ...     report.eval(X=x_batch)
    >>> print(report.global_violation())
"""

from typing import Any, Dict, List, Optional, Union

import sympy as sp
import torch

from pysignet.eval.checker import ConsistencyChecker
from pysignet.predicate import Predicate


class ConsistencyReport:
    """Accumulate consistency results across batches.

    Builds ConsistencyChecker(s) internally and provides
    aggregate metrics (global/conditional violation) after
    evaluating multiple batches.

    Supports single-expression (backward compatible) and
    multi-constraint modes. In multi-constraint mode, metrics
    are returned as dicts keyed by constraint name.

    The antecedent for conditional violation is auto-detected:
    - Implies(A, B): antecedent is A
    - Any other formula: antecedent is sp.true (all examples
      count, so conditional_violation equals global_violation)

    Args:
        expression: SymPy logic expression, or dict mapping
            constraint names to expressions.
        predicates: Dict mapping predicate names to Predicate
            objects.

    Example:
        >>> report = ConsistencyReport(expr, predicates)
        >>> for x_batch in dataloader:
        ...     report.eval(X=x_batch)
        >>> print(report.global_violation())
    """

    def __init__(
        self,
        expression: Union[
            sp.Basic, Dict[str, sp.Basic]
        ],
        predicates: Dict[str, Predicate],
    ) -> None:
        if isinstance(expression, dict):
            self._is_single = False
            self._names: List[str] = list(expression.keys())
            expr_dict = expression
        else:
            self._is_single = True
            self._names = ["default"]
            expr_dict = {"default": expression}

        self._checkers: Dict[str, ConsistencyChecker] = {}
        self._antecedents: Dict[
            str, Optional[ConsistencyChecker]
        ] = {}

        for name, expr in expr_dict.items():
            self._checkers[name] = ConsistencyChecker(
                expr, predicates
            )
            if isinstance(expr, sp.Implies):
                self._antecedents[name] = (
                    ConsistencyChecker(
                        expr.args[0], predicates
                    )
                )
            else:
                self._antecedents[name] = None

        # For backward compat (used by old code)
        if self._is_single:
            self.antecedent = self._antecedents["default"]

        self._satisfied_counts: Dict[str, int] = {
            n: 0 for n in self._names
        }
        self._total_count: int = 0
        self._antecedent_true_counts: Dict[str, int] = {
            n: 0 for n in self._names
        }
        self._antecedent_true_violated: Dict[str, int] = {
            n: 0 for n in self._names
        }
        self._last_satisfied: Dict[
            str, Optional[torch.Tensor]
        ] = {n: None for n in self._names}
        self._history: List[Dict[str, Any]] = []

    def eval(
        self, **variable_bindings: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluate one batch and accumulate results.

        Args:
            **variable_bindings: Variable bindings as keyword
                arguments (e.g., X=x_tensor).

        Returns:
            Single mode: Boolean tensor of shape (batch_size,).
            Multi mode: Dict mapping constraint names to
                boolean tensors.
        """
        results: Dict[str, torch.Tensor] = {}
        batch_size: Optional[int] = None

        for name in self._names:
            satisfied = self._checkers[name](
                **variable_bindings
            )
            results[name] = satisfied
            if batch_size is None:
                batch_size = satisfied.shape[0]

            self._satisfied_counts[name] += int(
                satisfied.sum().item()
            )
            self._last_satisfied[name] = satisfied

            ant_checker = self._antecedents[name]
            if ant_checker is not None:
                ant_true = ant_checker(**variable_bindings)
            else:
                ant_true = torch.ones(
                    satisfied.shape[0],
                    dtype=torch.bool,
                    device=satisfied.device,
                )

            self._antecedent_true_counts[name] += int(
                ant_true.sum().item()
            )
            violated = ant_true & (~satisfied)
            self._antecedent_true_violated[name] += int(
                violated.sum().item()
            )

        assert batch_size is not None
        self._total_count += batch_size

        # Build history entry
        self._history.append(
            self._build_history_entry(
                results, batch_size
            )
        )

        if self._is_single:
            return results["default"]
        return results

    def _build_history_entry(
        self,
        results: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> Dict[str, Any]:
        """Build a history entry for the current batch.

        Args:
            results: Dict of satisfied tensors per constraint.
            batch_size: Number of examples in this batch.

        Returns:
            History entry dict.
        """
        if self._is_single:
            satisfied = results["default"]
            sat_count = int(satisfied.sum().item())
            violated_count = batch_size - sat_count
            rho = violated_count / batch_size
            return {
                "batch_size": batch_size,
                "rho": rho,
                "tau": rho,
            }

        constraints: Dict[str, Dict[str, float]] = {}
        for name in self._names:
            satisfied = results[name]
            sat_count = int(satisfied.sum().item())
            violated_count = batch_size - sat_count
            rho = violated_count / batch_size
            constraints[name] = {"rho": rho, "tau": rho}

        return {
            "batch_size": batch_size,
            "constraints": constraints,
        }

    def global_violation(
        self,
    ) -> Union[float, Dict[str, float]]:
        """Global violation rate (rho).

        Fraction of all evaluated examples where the formula
        is violated.

        Returns:
            Single mode: Float in [0, 1].
            Multi mode: Dict of floats.
            Returns 0.0 if no examples evaluated.
        """
        if self._is_single:
            return self._global_violation_for("default")

        return {
            n: self._global_violation_for(n)
            for n in self._names
        }

    def _global_violation_for(self, name: str) -> float:
        if self._total_count == 0:
            return 0.0
        violated = (
            self._total_count - self._satisfied_counts[name]
        )
        return violated / self._total_count

    def global_consistency(
        self,
    ) -> Union[float, Dict[str, float]]:
        """Global consistency rate (1 - rho).

        Fraction of all evaluated examples where the formula
        is satisfied.

        Returns:
            Single mode: Float in [0, 1].
            Multi mode: Dict of floats.
            Returns 0.0 if no examples evaluated.
        """
        if self._is_single:
            return self._global_consistency_for("default")

        return {
            n: self._global_consistency_for(n)
            for n in self._names
        }

    def _global_consistency_for(self, name: str) -> float:
        if self._total_count == 0:
            return 0.0
        return self._satisfied_counts[name] / self._total_count

    def conditional_violation(
        self,
    ) -> Union[float, Dict[str, float]]:
        """Conditional violation rate (tau).

        Fraction of examples with antecedent=True where the
        formula is violated. For non-implication formulas the
        antecedent is sp.true, so tau equals rho.

        Returns:
            Single mode: Float in [0, 1].
            Multi mode: Dict of floats.
            Returns 0.0 if no examples have antecedent=True.
        """
        if self._is_single:
            return self._conditional_violation_for("default")

        return {
            n: self._conditional_violation_for(n)
            for n in self._names
        }

    def _conditional_violation_for(
        self, name: str
    ) -> float:
        if self._antecedent_true_counts[name] == 0:
            return 0.0
        return (
            self._antecedent_true_violated[name]
            / self._antecedent_true_counts[name]
        )

    def satisfaction_count(
        self,
    ) -> Union[int, Dict[str, int]]:
        """Number of examples where formula was satisfied.

        Returns:
            Single mode: Non-negative integer.
            Multi mode: Dict of non-negative integers.
        """
        if self._is_single:
            return self._satisfied_counts["default"]

        return dict(self._satisfied_counts)

    def total_count(self) -> int:
        """Total number of examples evaluated.

        Returns:
            Non-negative integer (shared across constraints).
        """
        return self._total_count

    def violated_indices(
        self,
    ) -> Union[
        torch.Tensor, Dict[str, torch.Tensor]
    ]:
        """Indices violated in the most recent eval() call.

        Returns indices where the formula was not satisfied
        in the last batch evaluated. Before any eval, returns
        an empty tensor.

        Returns:
            Single mode: 1-D int64 tensor of indices.
            Multi mode: Dict mapping constraint names to
                1-D int64 tensors.
        """
        if self._is_single:
            return self._violated_indices_for("default")

        return {
            n: self._violated_indices_for(n)
            for n in self._names
        }

    def _violated_indices_for(
        self, name: str
    ) -> torch.Tensor:
        last = self._last_satisfied[name]
        if last is None:
            return torch.tensor([], dtype=torch.int64)
        return torch.where(~last)[0]

    def history(self) -> List[Dict[str, Any]]:
        """Per-batch history of metrics.

        Each eval() call appends one entry. Reset clears all.

        Returns:
            List of dicts. Single mode entries have keys:
            rho, tau, batch_size. Multi mode entries have
            keys: batch_size, constraints (nested dict).
        """
        return list(self._history)

    def __repr__(self) -> str:
        if self._total_count == 0:
            return "ConsistencyReport(no data)"

        if self._is_single:
            return self._repr_single()
        return self._repr_multi()

    def _repr_single(self) -> str:
        rho = self._global_violation_for("default")
        tau = self._conditional_violation_for("default")
        sat = self._satisfied_counts["default"]
        lines = [
            f"ConsistencyReport("
            f"{sat}/{self._total_count} "
            f"satisfied)",
            f"  global_violation (rho): {rho:.4f}",
            f"  global_consistency:     "
            f"{1.0 - rho:.4f}",
        ]
        if self._antecedents["default"] is not None:
            lines.append(
                f"  conditional_violation (tau): "
                f"{tau:.4f}"
            )
        return "\n".join(lines)

    def _repr_multi(self) -> str:
        lines = [
            f"ConsistencyReport("
            f"{self._total_count} examples, "
            f"{len(self._names)} constraints)"
        ]
        for name in self._names:
            rho = self._global_violation_for(name)
            sat = self._satisfied_counts[name]
            lines.append(
                f"  {name}: "
                f"{sat}/{self._total_count} satisfied, "
                f"rho={rho:.4f}"
            )
        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        """Return metrics as a JSON-serializable dict.

        Returns:
            Single mode: Dict with keys: satisfied_count,
                total_count, global_violation,
                global_consistency, conditional_violation.
            Multi mode: Dict with keys: total_count,
                constraints (nested dict per constraint).
        """
        if self._is_single:
            return {
                "satisfied_count": (
                    self._satisfied_counts["default"]
                ),
                "total_count": self._total_count,
                "global_violation": (
                    self._global_violation_for("default")
                ),
                "global_consistency": (
                    self._global_consistency_for("default")
                ),
                "conditional_violation": (
                    self._conditional_violation_for(
                        "default"
                    )
                ),
            }

        constraints: Dict[str, Dict[str, Any]] = {}
        for name in self._names:
            constraints[name] = {
                "satisfied_count": (
                    self._satisfied_counts[name]
                ),
                "global_violation": (
                    self._global_violation_for(name)
                ),
                "global_consistency": (
                    self._global_consistency_for(name)
                ),
                "conditional_violation": (
                    self._conditional_violation_for(name)
                ),
            }
        return {
            "total_count": self._total_count,
            "constraints": constraints,
        }

    def reset(self) -> None:
        """Clear all accumulated state."""
        for name in self._names:
            self._satisfied_counts[name] = 0
            self._antecedent_true_counts[name] = 0
            self._antecedent_true_violated[name] = 0
            self._last_satisfied[name] = None
        self._total_count = 0
        self._history.clear()
