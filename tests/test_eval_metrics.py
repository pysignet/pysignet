"""Tests for ConsistencyReport evaluation metrics.

Tests global violation (rho), conditional violation (tau), and
accumulation across batches. Based on Li et al. (2019),
arXiv:1909.00126.
"""

# pylint: disable=invalid-name

from typing import Dict

import sympy as sp
import torch

from pysignet import (
    ConsistencyChecker,
    Predicate,
    Symbol,
    consistency_report,
)
from pysignet.eval import ConsistencyReport
from pysignet.logic import Variable


class TestGlobalMetrics:
    """Test global violation and consistency metrics."""

    def test_all_satisfied(self) -> None:
        """All examples satisfied -> rho=0, consistency=1."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor([True, True, True]),
                is_model=False,
            ),
            "Q": Predicate(
                lambda _x: torch.tensor([True, True, True]),
                is_model=False,
            ),
        }
        report = ConsistencyReport(expr, predicates)
        report.eval(X=torch.randn(3, 5))

        assert report.global_violation() == 0.0
        assert report.global_consistency() == 1.0

    def test_none_satisfied(self) -> None:
        """No examples satisfied -> rho=1, consistency=0."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor([True, True, True]),
                is_model=False,
            ),
            "Q": Predicate(
                lambda _x: torch.tensor(
                    [False, False, False]
                ),
                is_model=False,
            ),
        }
        report = ConsistencyReport(expr, predicates)
        report.eval(X=torch.randn(3, 5))

        assert report.global_violation() == 1.0
        assert report.global_consistency() == 0.0

    def test_half_satisfied(self) -> None:
        """Half satisfied -> rho=0.5, consistency=0.5."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, True, True, True]
                ),
                is_model=False,
            ),
            "Q": Predicate(
                lambda _x: torch.tensor(
                    [True, True, False, False]
                ),
                is_model=False,
            ),
        }
        report = ConsistencyReport(expr, predicates)
        report.eval(X=torch.randn(4, 5))

        assert report.global_violation() == 0.5
        assert report.global_consistency() == 0.5

    def test_accumulates_across_batches(self) -> None:
        """Metrics accumulate across multiple eval calls."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        call_count = [0]

        def switching_pred(
            _x: torch.Tensor,
        ) -> torch.Tensor:
            call_count[0] += 1
            if call_count[0] == 1:
                return torch.tensor([True, True])
            return torch.tensor([False, False])

        predicates = {
            "P": Predicate(switching_pred, is_model=False),
        }
        report = ConsistencyReport(expr, predicates)

        report.eval(X=torch.randn(2, 5))
        report.eval(X=torch.randn(2, 5))

        # 2 satisfied + 0 satisfied out of 4 total
        assert report.global_violation() == 0.5
        assert report.total_count() == 4

    def test_satisfaction_count(self) -> None:
        """Satisfaction and total counts are correct."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, True, True, False, False]
                ),
                is_model=False,
            ),
        }
        report = ConsistencyReport(expr, predicates)
        report.eval(X=torch.randn(5, 5))

        assert report.satisfaction_count() == 3
        assert report.total_count() == 5


class TestConditionalViolation:
    """Test conditional violation (tau) metric."""

    def test_filters_vacuous_truths(self) -> None:
        """Tau only counts examples where antecedent holds."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        # Implies -> antecedent P(X) auto-detected
        formula = sp.Implies(P(X), Q(X))

        # P: [T, T, F, F]  Q: [T, F, T, F]
        # Formula:  [T, F, T, T]  (implication)
        # Antecedent P(X): [T, T, F, F]
        # Among antecedent=True (indices 0,1):
        #   formula[0]=T (ok), formula[1]=F (violated)
        # tau = 1 / 2 = 0.5
        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, True, False, False]
                ),
                is_model=False,
            ),
            "Q": Predicate(
                lambda _x: torch.tensor(
                    [True, False, True, False]
                ),
                is_model=False,
            ),
        }
        report = ConsistencyReport(formula, predicates)
        report.eval(X=torch.randn(4, 5))

        assert report.conditional_violation() == 0.5

    def test_no_antecedent_true(self) -> None:
        """All antecedent False -> tau=0.0."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        formula = sp.Implies(P(X), Q(X))

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor([False, False]),
                is_model=False,
            ),
            "Q": Predicate(
                lambda _x: torch.tensor([False, False]),
                is_model=False,
            ),
        }
        report = ConsistencyReport(formula, predicates)
        report.eval(X=torch.randn(2, 5))

        assert report.conditional_violation() == 0.0

    def test_conditional_accumulates(self) -> None:
        """Conditional violation accumulates across batches."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        formula = sp.Implies(P(X), Q(X))

        call_count = [0]

        def p_pred(
            _x: torch.Tensor,
        ) -> torch.Tensor:
            call_count[0] += 1
            # Calls alternate: formula checker, antecedent
            # checker, for each batch.
            idx = (call_count[0] - 1) // 2
            if idx == 0:
                return torch.tensor([True, True])
            return torch.tensor([True, False])

        def q_pred(
            _x: torch.Tensor,
        ) -> torch.Tensor:
            return torch.tensor([True, False])

        predicates = {
            "P": Predicate(p_pred, is_model=False),
            "Q": Predicate(q_pred, is_model=False),
        }
        report = ConsistencyReport(formula, predicates)

        report.eval(X=torch.randn(2, 5))
        report.eval(X=torch.randn(2, 5))

        # Batch 1: P=[T,T], Q=[T,F]
        #   formula=[T,F], antecedent=[T,T]
        #   ant_true_violated=1, ant_true=2
        # Batch 2: P=[T,F], Q=[T,F]
        #   formula=[T,T], antecedent=[T,F]
        #   ant_true_violated=0, ant_true=1
        # Total: 1 violated / 3 antecedent-true = 1/3
        assert abs(
            report.conditional_violation() - 1.0 / 3.0
        ) < 1e-6

    def test_non_implies_tau_equals_rho(self) -> None:
        """For non-Implies formulas, tau equals rho."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, True, False]
                ),
                is_model=False,
            ),
            "Q": Predicate(
                lambda _x: torch.tensor(
                    [True, False, True]
                ),
                is_model=False,
            ),
        }
        report = ConsistencyReport(expr, predicates)
        report.eval(X=torch.randn(3, 5))

        # And: [T, F, F] -> 2 violated / 3
        assert abs(
            report.global_violation()
            - report.conditional_violation()
        ) < 1e-6


class TestReset:
    """Test reset functionality."""

    def test_reset_clears_state(self) -> None:
        """After reset, all counts are zero."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, False, True]
                ),
                is_model=False,
            ),
        }
        report = ConsistencyReport(expr, predicates)
        report.eval(X=torch.randn(3, 5))

        assert report.total_count() == 3
        assert report.satisfaction_count() == 2

        report.reset()

        assert report.total_count() == 0
        assert report.satisfaction_count() == 0
        assert report.global_violation() == 0.0
        assert report.global_consistency() == 0.0


class TestToJson:
    """Test to_json() output."""

    def test_to_json_no_data(self) -> None:
        """Before any eval, all values are zero."""
        P = Symbol("P")
        X = Variable("X")
        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor([True]),
                is_model=False,
            ),
        }
        report = ConsistencyReport(P(X), predicates)
        data = report.to_json()

        assert data == {
            "satisfied_count": 0,
            "total_count": 0,
            "global_violation": 0.0,
            "global_consistency": 0.0,
            "conditional_violation": 0.0,
        }

    def test_to_json_after_eval(self) -> None:
        """Values reflect accumulated metrics."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, True, False, False]
                ),
                is_model=False,
            ),
            "Q": Predicate(
                lambda _x: torch.tensor(
                    [True, False, True, False]
                ),
                is_model=False,
            ),
        }
        report = ConsistencyReport(
            sp.Implies(P(X), Q(X)), predicates
        )
        report.eval(X=torch.randn(4, 5))

        data = report.to_json()

        assert data["satisfied_count"] == 3
        assert data["total_count"] == 4
        assert data["global_violation"] == 0.25
        assert data["global_consistency"] == 0.75
        assert data["conditional_violation"] == 0.5

    def test_to_json_is_serializable(self) -> None:
        """Output is JSON-serializable."""
        import json

        P = Symbol("P")
        X = Variable("X")
        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor([True, False]),
                is_model=False,
            ),
        }
        report = ConsistencyReport(P(X), predicates)
        report.eval(X=torch.randn(2, 5))

        # Should not raise
        serialized = json.dumps(report.to_json())
        assert isinstance(serialized, str)


class TestRepr:
    """Test __repr__ output."""

    def test_repr_no_data(self) -> None:
        """Before any eval, repr says no data."""
        P = Symbol("P")
        X = Variable("X")
        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor([True]),
                is_model=False,
            ),
        }
        report = ConsistencyReport(P(X), predicates)
        assert repr(report) == "ConsistencyReport(no data)"

    def test_repr_global_only(self) -> None:
        """Non-Implies formula shows rho and consistency."""
        P = Symbol("P")
        X = Variable("X")
        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, True, False, False]
                ),
                is_model=False,
            ),
        }
        report = ConsistencyReport(P(X), predicates)
        report.eval(X=torch.randn(4, 5))

        text = repr(report)
        assert "2/4 satisfied" in text
        assert "global_violation (rho): 0.5000" in text
        assert "global_consistency:     0.5000" in text
        assert "conditional_violation" not in text

    def test_repr_with_antecedent(self) -> None:
        """Implies formula also shows tau."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, True, False, False]
                ),
                is_model=False,
            ),
            "Q": Predicate(
                lambda _x: torch.tensor(
                    [True, False, True, False]
                ),
                is_model=False,
            ),
        }
        report = ConsistencyReport(
            sp.Implies(P(X), Q(X)), predicates
        )
        report.eval(X=torch.randn(4, 5))

        text = repr(report)
        assert "3/4 satisfied" in text
        assert "conditional_violation (tau)" in text


class TestConvenienceHelper:
    """Test consistency_report() convenience function."""

    def test_consistency_report_with_predicates(self) -> None:
        """Works the same as constructing directly."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Implies(P(X), Q(X))

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, True, False]
                ),
                is_model=False,
            ),
            "Q": Predicate(
                lambda _x: torch.tensor(
                    [True, False, True]
                ),
                is_model=False,
            ),
        }
        report = consistency_report(expr, predicates)
        report.eval(X=torch.randn(3, 5))

        # Implies: [T, F, T] -> 1 violated / 3
        assert report.total_count() == 3
        assert abs(
            report.global_violation() - 1.0 / 3.0
        ) < 1e-6

    def test_auto_wraps_callables(self) -> None:
        """Raw callables are auto-wrapped in Predicate."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        # Pass raw lambda, not Predicate
        report = consistency_report(
            expr,
            {"P": lambda _x: torch.tensor([True, False])},
        )
        report.eval(X=torch.randn(2, 5))

        assert report.satisfaction_count() == 1
        assert report.total_count() == 2


class TestIntegrationWithConsistencyChecker:
    """Integration tests comparing report to manual calc."""

    def test_global_violation_matches_manual(self) -> None:
        """Report matches manual 1-mean(satisfied)."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Implies(P(X), Q(X))

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, True, False, True, False]
                ),
                is_model=False,
            ),
            "Q": Predicate(
                lambda _x: torch.tensor(
                    [True, False, True, True, False]
                ),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        x = torch.randn(5, 5)

        # Manual computation
        satisfied = checker(X=x)
        manual_violation = (
            1.0 - satisfied.float().mean().item()
        )

        # Report computation
        report = ConsistencyReport(expr, predicates)
        report.eval(X=x)

        assert abs(
            report.global_violation() - manual_violation
        ) < 1e-6

    def test_eval_returns_batch_result(self) -> None:
        """eval() returns the boolean tensor for the batch."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, False, True]
                ),
                is_model=False,
            ),
        }
        report = ConsistencyReport(expr, predicates)

        result = report.eval(X=torch.randn(3, 5))
        expected = torch.tensor([True, False, True])
        assert torch.equal(result, expected)

    def test_empty_no_eval(self) -> None:
        """Querying before any eval returns zero."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor([True]),
                is_model=False,
            ),
        }
        report = ConsistencyReport(expr, predicates)

        assert report.total_count() == 0
        assert report.satisfaction_count() == 0
        assert report.global_violation() == 0.0
        assert report.global_consistency() == 0.0


class TestMultiConstraint:
    """Test multi-constraint ConsistencyReport."""

    @staticmethod
    def _make_predicates() -> Dict[str, Predicate]:
        """Shared predicates for multi-constraint tests.

        P: [T, T, F, F], Q: [T, F, T, F]
        """
        return {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, True, False, False]
                ),
                is_model=False,
            ),
            "Q": Predicate(
                lambda _x: torch.tensor(
                    [True, False, True, False]
                ),
                is_model=False,
            ),
        }

    def test_multi_global_violation(self) -> None:
        """Dict of expressions returns dict of rho values."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        predicates = self._make_predicates()

        # And: [T,F,F,F] -> rho=0.75
        # Or:  [T,T,T,F] -> rho=0.25
        exprs = {
            "conj": sp.And(P(X), Q(X)),
            "disj": sp.Or(P(X), Q(X)),
        }
        report = ConsistencyReport(exprs, predicates)
        report.eval(X=torch.randn(4, 5))

        result = report.global_violation()
        assert isinstance(result, dict)
        assert result["conj"] == 0.75
        assert result["disj"] == 0.25

    def test_multi_conditional_violation(self) -> None:
        """Dict of tau values, one Implies and one And."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        predicates = self._make_predicates()

        # Implies(P,Q): formula=[T,F,T,T], ant P=[T,T,F,F]
        #   ant_true=2, violated_in_ant=1, tau=0.5
        # And(P,Q): non-Implies, tau==rho
        #   And=[T,F,F,F], rho=0.75, tau=0.75
        exprs = {
            "impl": sp.Implies(P(X), Q(X)),
            "conj": sp.And(P(X), Q(X)),
        }
        report = ConsistencyReport(exprs, predicates)
        report.eval(X=torch.randn(4, 5))

        result = report.conditional_violation()
        assert isinstance(result, dict)
        assert result["impl"] == 0.5
        assert result["conj"] == 0.75

    def test_multi_eval_returns_dict(self) -> None:
        """eval() returns dict of boolean tensors."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        predicates = self._make_predicates()

        exprs = {
            "conj": sp.And(P(X), Q(X)),
            "disj": sp.Or(P(X), Q(X)),
        }
        report = ConsistencyReport(exprs, predicates)
        result = report.eval(X=torch.randn(4, 5))

        assert isinstance(result, dict)
        assert torch.equal(
            result["conj"],
            torch.tensor([True, False, False, False]),
        )
        assert torch.equal(
            result["disj"],
            torch.tensor([True, True, True, False]),
        )

    def test_multi_satisfaction_count(self) -> None:
        """Dict of satisfaction counts per constraint."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        predicates = self._make_predicates()

        exprs = {
            "conj": sp.And(P(X), Q(X)),
            "disj": sp.Or(P(X), Q(X)),
        }
        report = ConsistencyReport(exprs, predicates)
        report.eval(X=torch.randn(4, 5))

        result = report.satisfaction_count()
        assert isinstance(result, dict)
        assert result["conj"] == 1
        assert result["disj"] == 3

    def test_multi_total_count(self) -> None:
        """total_count is shared across constraints (int)."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        predicates = self._make_predicates()

        exprs = {
            "conj": sp.And(P(X), Q(X)),
            "disj": sp.Or(P(X), Q(X)),
        }
        report = ConsistencyReport(exprs, predicates)
        report.eval(X=torch.randn(4, 5))

        assert report.total_count() == 4
        assert isinstance(report.total_count(), int)

    def test_multi_accumulates(self) -> None:
        """Metrics accumulate over two batches."""
        P = Symbol("P")
        X = Variable("X")

        call_count = [0]

        def switching_pred(
            _x: torch.Tensor,
        ) -> torch.Tensor:
            call_count[0] += 1
            if call_count[0] <= 2:
                # First batch: both checkers see [T, T]
                return torch.tensor([True, True])
            # Second batch: both checkers see [F, F]
            return torch.tensor([False, False])

        predicates = {
            "P": Predicate(switching_pred, is_model=False),
        }
        exprs = {
            "a": P(X),
            "b": P(X),
        }
        report = ConsistencyReport(exprs, predicates)

        report.eval(X=torch.randn(2, 5))
        report.eval(X=torch.randn(2, 5))

        assert report.total_count() == 4
        result = report.global_violation()
        assert isinstance(result, dict)
        # Both constraints use same pred, should both be 0.5
        assert result["a"] == 0.5
        assert result["b"] == 0.5

    def test_multi_repr(self) -> None:
        """Repr shows each constraint name."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        predicates = self._make_predicates()

        exprs = {
            "conj": sp.And(P(X), Q(X)),
            "disj": sp.Or(P(X), Q(X)),
        }
        report = ConsistencyReport(exprs, predicates)
        report.eval(X=torch.randn(4, 5))

        text = repr(report)
        assert "conj" in text
        assert "disj" in text

    def test_multi_to_json(self) -> None:
        """to_json has nested constraints dict."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        predicates = self._make_predicates()

        exprs = {
            "conj": sp.And(P(X), Q(X)),
            "disj": sp.Or(P(X), Q(X)),
        }
        report = ConsistencyReport(exprs, predicates)
        report.eval(X=torch.randn(4, 5))

        data = report.to_json()
        assert "total_count" in data
        assert data["total_count"] == 4
        assert "constraints" in data
        assert "conj" in data["constraints"]
        assert "disj" in data["constraints"]
        assert data["constraints"]["conj"]["satisfied_count"] == 1
        assert (
            data["constraints"]["conj"]["global_violation"]
            == 0.75
        )
        assert data["constraints"]["disj"]["satisfied_count"] == 3
        assert (
            data["constraints"]["disj"]["global_violation"]
            == 0.25
        )

    def test_multi_reset(self) -> None:
        """Reset clears all constraint state."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        predicates = self._make_predicates()

        exprs = {
            "conj": sp.And(P(X), Q(X)),
            "disj": sp.Or(P(X), Q(X)),
        }
        report = ConsistencyReport(exprs, predicates)
        report.eval(X=torch.randn(4, 5))
        report.reset()

        assert report.total_count() == 0
        result = report.global_violation()
        assert isinstance(result, dict)
        assert result["conj"] == 0.0
        assert result["disj"] == 0.0

    def test_multi_global_consistency(self) -> None:
        """Dict of global_consistency values."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        predicates = self._make_predicates()

        exprs = {
            "conj": sp.And(P(X), Q(X)),
            "disj": sp.Or(P(X), Q(X)),
        }
        report = ConsistencyReport(exprs, predicates)
        report.eval(X=torch.randn(4, 5))

        result = report.global_consistency()
        assert isinstance(result, dict)
        assert result["conj"] == 0.25
        assert result["disj"] == 0.75


class TestViolatedIndices:
    """Test violated_indices() method."""

    def test_violated_indices_single(self) -> None:
        """Returns indices where formula violated."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        # P: [T, F, T, F, F]
        # Violated at indices 1, 3, 4
        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, False, True, False, False]
                ),
                is_model=False,
            ),
        }
        report = ConsistencyReport(expr, predicates)
        report.eval(X=torch.randn(5, 5))

        indices = report.violated_indices()
        assert isinstance(indices, torch.Tensor)
        expected = torch.tensor([1, 3, 4])
        assert torch.equal(indices, expected)

    def test_violated_indices_multi(self) -> None:
        """Returns dict of indices per constraint."""
        P, Q = Symbol("P Q")
        X = Variable("X")

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, True, False, False]
                ),
                is_model=False,
            ),
            "Q": Predicate(
                lambda _x: torch.tensor(
                    [True, False, True, False]
                ),
                is_model=False,
            ),
        }
        # And: [T,F,F,F] -> violated at 1,2,3
        # Or:  [T,T,T,F] -> violated at 3
        exprs = {
            "conj": sp.And(P(X), Q(X)),
            "disj": sp.Or(P(X), Q(X)),
        }
        report = ConsistencyReport(exprs, predicates)
        report.eval(X=torch.randn(4, 5))

        result = report.violated_indices()
        assert isinstance(result, dict)
        assert torch.equal(
            result["conj"], torch.tensor([1, 2, 3])
        )
        assert torch.equal(
            result["disj"], torch.tensor([3])
        )

    def test_violated_indices_all_satisfied(self) -> None:
        """All satisfied returns empty tensor."""
        P = Symbol("P")
        X = Variable("X")

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor([True, True, True]),
                is_model=False,
            ),
        }
        report = ConsistencyReport(P(X), predicates)
        report.eval(X=torch.randn(3, 5))

        indices = report.violated_indices()
        assert isinstance(indices, torch.Tensor)
        assert indices.numel() == 0

    def test_violated_indices_before_eval(self) -> None:
        """Before any eval, returns empty tensor."""
        P = Symbol("P")
        X = Variable("X")

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor([True]),
                is_model=False,
            ),
        }
        report = ConsistencyReport(P(X), predicates)

        indices = report.violated_indices()
        assert isinstance(indices, torch.Tensor)
        assert indices.numel() == 0


class TestHistory:
    """Test per-batch history tracking."""

    def test_history_records_per_batch(self) -> None:
        """History length grows with each eval call."""
        P = Symbol("P")
        X = Variable("X")

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor([True, False]),
                is_model=False,
            ),
        }
        report = ConsistencyReport(P(X), predicates)

        report.eval(X=torch.randn(2, 5))
        assert len(report.history()) == 1

        report.eval(X=torch.randn(2, 5))
        assert len(report.history()) == 2

    def test_history_single_values(self) -> None:
        """Single-constraint history has rho, tau, batch_size."""
        P = Symbol("P")
        X = Variable("X")

        # P: [T, F, T] -> 1 violated out of 3
        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, False, True]
                ),
                is_model=False,
            ),
        }
        report = ConsistencyReport(P(X), predicates)
        report.eval(X=torch.randn(3, 5))

        hist = report.history()
        assert len(hist) == 1
        entry = hist[0]
        assert entry["batch_size"] == 3
        assert abs(entry["rho"] - 1.0 / 3.0) < 1e-6
        # Non-Implies: tau == rho
        assert abs(entry["tau"] - 1.0 / 3.0) < 1e-6

    def test_history_multi_values(self) -> None:
        """Multi-constraint history has nested per-constraint."""
        P, Q = Symbol("P Q")
        X = Variable("X")

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, True, False, False]
                ),
                is_model=False,
            ),
            "Q": Predicate(
                lambda _x: torch.tensor(
                    [True, False, True, False]
                ),
                is_model=False,
            ),
        }
        exprs = {
            "conj": sp.And(P(X), Q(X)),
            "disj": sp.Or(P(X), Q(X)),
        }
        report = ConsistencyReport(exprs, predicates)
        report.eval(X=torch.randn(4, 5))

        hist = report.history()
        assert len(hist) == 1
        entry = hist[0]
        assert entry["batch_size"] == 4
        assert "constraints" in entry
        assert "conj" in entry["constraints"]
        assert "disj" in entry["constraints"]
        assert entry["constraints"]["conj"]["rho"] == 0.75
        assert entry["constraints"]["disj"]["rho"] == 0.25

    def test_history_cleared_on_reset(self) -> None:
        """Reset clears history."""
        P = Symbol("P")
        X = Variable("X")

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor([True, False]),
                is_model=False,
            ),
        }
        report = ConsistencyReport(P(X), predicates)
        report.eval(X=torch.randn(2, 5))
        assert len(report.history()) == 1

        report.reset()
        assert len(report.history()) == 0


class TestMultiConvenienceHelper:
    """Test consistency_report() with multi-constraint dict."""

    def test_consistency_report_multi(self) -> None:
        """Dict arg works with convenience function."""
        P, Q = Symbol("P Q")
        X = Variable("X")

        predicates = {
            "P": Predicate(
                lambda _x: torch.tensor(
                    [True, True, False]
                ),
                is_model=False,
            ),
            "Q": Predicate(
                lambda _x: torch.tensor(
                    [True, False, True]
                ),
                is_model=False,
            ),
        }
        exprs = {
            "conj": sp.And(P(X), Q(X)),
            "disj": sp.Or(P(X), Q(X)),
        }
        report = consistency_report(exprs, predicates)
        report.eval(X=torch.randn(3, 5))

        result = report.global_violation()
        assert isinstance(result, dict)
        assert report.total_count() == 3
