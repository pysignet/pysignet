"""Tests for the ConsistencyChecker class.

This module tests hard (boolean) formula evaluation and consistency
measurement using the eval module's ConsistencyChecker.
"""

# pylint: disable=invalid-name

import pytest
import sympy as sp
import torch

from pysignet import ConsistencyChecker, Symbol, Predicate
from pysignet.logic import Variable


class TestBasicConsistencyChecking:
    """Test basic consistency checking with boolean predicates."""

    def test_simple_and_satisfied(self) -> None:
        """Test AND formula that is satisfied."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(
                lambda x: torch.tensor([True]), is_model=False
            ),
            "Q": Predicate(
                lambda x: torch.tensor([True]), is_model=False
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))
        assert result[0].item() is True

    def test_simple_and_violated(self) -> None:
        """Test AND formula that is violated."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(
                lambda x: torch.tensor([True]), is_model=False
            ),
            "Q": Predicate(
                lambda x: torch.tensor([False]),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))
        assert result[0].item() is False

    def test_simple_or_satisfied(self) -> None:
        """Test OR formula that is satisfied."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Or(P(X), Q(X))

        predicates = {
            "P": Predicate(
                lambda x: torch.tensor([False]),
                is_model=False,
            ),
            "Q": Predicate(
                lambda x: torch.tensor([True]), is_model=False
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))
        assert result[0].item() is True

    def test_simple_not(self) -> None:
        """Test NOT formula."""
        P = Symbol("P")
        X = Variable("X")
        expr = sp.Not(P(X))

        # Test with P=False
        predicates = {
            "P": Predicate(
                lambda x: torch.tensor([False]),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))
        assert result[0].item() is True

        # Test with P=True
        predicates = {
            "P": Predicate(
                lambda x: torch.tensor([True]), is_model=False
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))
        assert result[0].item() is False

    def test_implication_satisfied(self) -> None:
        """Test implication that is satisfied."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Implies(P(X), Q(X))

        # True -> True = True
        predicates = {
            "P": Predicate(
                lambda x: torch.tensor([True]), is_model=False
            ),
            "Q": Predicate(
                lambda x: torch.tensor([True]), is_model=False
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))
        assert result[0].item() is True

        # False -> True = True
        predicates = {
            "P": Predicate(
                lambda x: torch.tensor([False]),
                is_model=False,
            ),
            "Q": Predicate(
                lambda x: torch.tensor([True]), is_model=False
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))
        assert result[0].item() is True

        # False -> False = True
        predicates = {
            "P": Predicate(
                lambda x: torch.tensor([False]),
                is_model=False,
            ),
            "Q": Predicate(
                lambda x: torch.tensor([False]),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))
        assert result[0].item() is True

    def test_implication_violated(self) -> None:
        """Test implication that is violated."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Implies(P(X), Q(X))

        # True -> False = False
        predicates = {
            "P": Predicate(
                lambda x: torch.tensor([True]), is_model=False
            ),
            "Q": Predicate(
                lambda x: torch.tensor([False]),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))
        assert result[0].item() is False

    def test_equivalence(self) -> None:
        """Test equivalence (biconditional)."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Equivalent(P(X), Q(X))

        # Both true = satisfied
        predicates = {
            "P": Predicate(
                lambda x: torch.tensor([True]), is_model=False
            ),
            "Q": Predicate(
                lambda x: torch.tensor([True]), is_model=False
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))
        assert result[0].item() is True

        # Both false = satisfied
        predicates = {
            "P": Predicate(
                lambda x: torch.tensor([False]),
                is_model=False,
            ),
            "Q": Predicate(
                lambda x: torch.tensor([False]),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))
        assert result[0].item() is True

        # Different values = violated
        predicates = {
            "P": Predicate(
                lambda x: torch.tensor([True]), is_model=False
            ),
            "Q": Predicate(
                lambda x: torch.tensor([False]),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))
        assert result[0].item() is False


class TestTransitivityConstraint:
    """Test transitivity constraint from NLI paper."""

    def test_transitivity_entailment(self) -> None:
        """Test: E(P,H) AND E(H,Z) -> E(P,Z)."""
        E = Symbol("E")
        constraint = sp.Implies(
            sp.And(E("P", "H"), E("H", "Z")), E("P", "Z")
        )

        # Satisfied: all True
        predicates = {
            "E": Predicate(
                lambda p, h: torch.tensor([True]),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(constraint, predicates)
        result = checker()
        assert result[0].item() is True

        # Violated: E(P,Z) = False
        def e_violated(p, h):
            if p == "P" and h == "H":
                return torch.tensor([True])
            if p == "H" and h == "Z":
                return torch.tensor([True])
            if p == "P" and h == "Z":
                return torch.tensor([False])
            return torch.tensor([False])

        predicates = {
            "E": Predicate(e_violated, is_model=False)
        }
        checker = ConsistencyChecker(constraint, predicates)
        result = checker()
        assert result[0].item() is False

        # Vacuously true: E(P,H) = False
        def e_vacuous(p, h):
            if p == "P" and h == "H":
                return torch.tensor([False])
            if p == "H" and h == "Z":
                return torch.tensor([True])
            if p == "P" and h == "Z":
                return torch.tensor([False])
            return torch.tensor([False])

        predicates = {
            "E": Predicate(e_vacuous, is_model=False)
        }
        checker = ConsistencyChecker(constraint, predicates)
        result = checker()
        assert result[0].item() is True

    def test_transitivity_contradiction(self) -> None:
        """Test: E(P,H) AND C(H,Z) -> C(P,Z)."""
        E, C = Symbol("E C")
        constraint = sp.Implies(
            sp.And(E("P", "H"), C("H", "Z")), C("P", "Z")
        )

        def e_func(p, h):
            if p == "P" and h == "H":
                return torch.tensor([True])
            return torch.tensor([False])

        def c_func(p, h):
            if p == "H" and h == "Z":
                return torch.tensor([True])
            if p == "P" and h == "Z":
                return torch.tensor([False])
            return torch.tensor([False])

        predicates = {
            "E": Predicate(e_func, is_model=False),
            "C": Predicate(c_func, is_model=False),
        }
        checker = ConsistencyChecker(constraint, predicates)
        result = checker()
        assert result[0].item() is False


class TestSymmetryConstraint:
    """Test symmetry constraint from NLI paper."""

    def test_symmetry_equivalence(self) -> None:
        """Test: C(P,H) <-> C(H,P)."""
        C = Symbol("C")
        constraint = sp.Equivalent(C("P", "H"), C("H", "P"))

        # Both true
        predicates = {
            "C": Predicate(
                lambda p, h: torch.tensor([True]),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(constraint, predicates)
        result = checker()
        assert result[0].item() is True

        # Both false
        predicates = {
            "C": Predicate(
                lambda p, h: torch.tensor([False]),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(constraint, predicates)
        result = checker()
        assert result[0].item() is True

        # Mismatch
        def c_mismatch(p, h):
            if p == "P" and h == "H":
                return torch.tensor([True])
            if p == "H" and h == "P":
                return torch.tensor([False])
            return torch.tensor([False])

        predicates = {
            "C": Predicate(c_mismatch, is_model=False)
        }
        checker = ConsistencyChecker(constraint, predicates)
        result = checker()
        assert result[0].item() is False


class TestBatchConsistencyChecking:
    """Test consistency checking with batches."""

    def test_batch_and_formula(self) -> None:
        """Test AND formula with batch of decisions."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(
                lambda x: torch.tensor(
                    [True, True, False, True]
                ),
                is_model=False,
            ),
            "Q": Predicate(
                lambda x: torch.tensor(
                    [True, False, True, False]
                ),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))
        expected = torch.tensor([True, False, False, False])
        assert torch.equal(result, expected)

    def test_batch_implication(self) -> None:
        """Test implication with batch of decisions."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Implies(P(X), Q(X))

        predicates = {
            "P": Predicate(
                lambda x: torch.tensor(
                    [True, False, True, False]
                ),
                is_model=False,
            ),
            "Q": Predicate(
                lambda x: torch.tensor(
                    [True, True, False, False]
                ),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))
        expected = torch.tensor([True, True, False, True])
        assert torch.equal(result, expected)

    def test_consistency_counting(self) -> None:
        """Test counting satisfied examples in a batch."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(
                lambda x: torch.tensor(
                    [True, True, False, True, True]
                ),
                is_model=False,
            ),
            "Q": Predicate(
                lambda x: torch.tensor(
                    [True, False, True, False, True]
                ),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        satisfied = checker(X=torch.randn(1, 10))

        num_satisfied = satisfied.sum().item()
        assert num_satisfied == 2

        fraction = satisfied.float().mean().item()
        assert abs(fraction - 0.4) < 1e-6


class TestPredicatesWithModels:
    """Test predicates that use actual model outputs."""

    def test_argmax_predicates(self) -> None:
        """Test predicates based on argmax of model outputs."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Implies(P(X), Q(X))

        model1_outputs = torch.tensor([
            [0.9, 0.1, 0.0],  # argmax = 0
            [0.1, 0.8, 0.1],  # argmax = 1
            [0.8, 0.1, 0.1],  # argmax = 0
        ])

        model2_outputs = torch.tensor([
            [0.2, 0.8],  # argmax = 1
            [0.9, 0.1],  # argmax = 0
            [0.1, 0.9],  # argmax = 1
        ])

        predicates = {
            "P": Predicate(
                lambda x: (
                    model1_outputs.argmax(dim=-1) == 0
                ),
                is_model=False,
            ),
            "Q": Predicate(
                lambda x: (
                    model2_outputs.argmax(dim=-1) == 1
                ),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))

        # P: [True, False, True]
        # Q: [True, False, True]
        # P -> Q: [True, True, True]
        expected = torch.tensor([True, True, True])
        assert torch.equal(result, expected)


class TestBooleanConstants:
    """Test boolean constants in formulas."""

    def test_true_constant(self) -> None:
        """Test true constant."""
        P = Symbol("P")
        X = Variable("X")
        expr = sp.And(sp.true, P(X))

        predicates = {
            "P": Predicate(
                lambda x: torch.tensor([True, False, True]),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))
        expected = torch.tensor([True, False, True])
        assert torch.equal(result, expected)

    def test_false_constant(self) -> None:
        """Test false constant."""
        P = Symbol("P")
        X = Variable("X")
        expr = sp.Or(sp.false, P(X))

        predicates = {
            "P": Predicate(
                lambda x: torch.tensor([True, False, True]),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))
        expected = torch.tensor([True, False, True])
        assert torch.equal(result, expected)


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_missing_predicate_error(self) -> None:
        """Test error when predicate is missing."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": Predicate(
                lambda x: torch.tensor([True]),
                is_model=False,
            ),
        }

        with pytest.raises(
            ValueError, match="Missing predicates"
        ):
            ConsistencyChecker(expr, predicates)

    def test_non_boolean_tensor_conversion(self) -> None:
        """Test automatic conversion of float to boolean."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        # Predicate returns float tensor -- threshold at 0.5
        predicates = {
            "P": Predicate(
                lambda x: torch.tensor([1.0, 0.0, 1.0]),
                is_model=False,
            ),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(X=torch.randn(1, 10))

        expected = torch.tensor([True, False, True])
        assert torch.equal(result, expected)
