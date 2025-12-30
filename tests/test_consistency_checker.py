"""Tests for the ConsistencyChecker class.

This module tests hard (boolean) formula evaluation and consistency
measurement as described in the new simplified interface.
"""

import pytest
import sympy as sp
import torch

from pysignet import ConsistencyChecker


class TestBasicConsistencyChecking:
    """Test basic consistency checking with boolean predicates."""

    def test_simple_and_satisfied(self) -> None:
        """Test AND formula that is satisfied."""
        P, Q = sp.symbols("P Q")
        expr = sp.And(P, Q)

        predicates = {
            "P": lambda x: torch.tensor([True]),
            "Q": lambda x: torch.tensor([True]),
        }
        checker = ConsistencyChecker(expr, predicates)

        x = torch.randn(1, 10)
        result = checker(x)
        assert result[0].item() is True

    def test_simple_and_violated(self) -> None:
        """Test AND formula that is violated."""
        P, Q = sp.symbols("P Q")
        expr = sp.And(P, Q)

        predicates = {
            "P": lambda x: torch.tensor([True]),
            "Q": lambda x: torch.tensor([False]),
        }
        checker = ConsistencyChecker(expr, predicates)

        x = torch.randn(1, 10)
        result = checker(x)
        assert result[0].item() is False

    def test_simple_or_satisfied(self) -> None:
        """Test OR formula that is satisfied."""
        P, Q = sp.symbols("P Q")
        expr = sp.Or(P, Q)

        predicates = {
            "P": lambda x: torch.tensor([False]),
            "Q": lambda x: torch.tensor([True]),
        }
        checker = ConsistencyChecker(expr, predicates)

        x = torch.randn(1, 10)
        result = checker(x)
        assert result[0].item() is True

    def test_simple_not(self) -> None:
        """Test NOT formula."""
        P = sp.symbols("P")
        expr = sp.Not(P)

        # Test with P=False
        predicates = {"P": lambda x: torch.tensor([False])}
        checker = ConsistencyChecker(expr, predicates)
        x = torch.randn(1, 10)
        result = checker(x)
        assert result[0].item() is True

        # Test with P=True
        predicates = {"P": lambda x: torch.tensor([True])}
        checker = ConsistencyChecker(expr, predicates)
        result = checker(x)
        assert result[0].item() is False

    def test_implication_satisfied(self) -> None:
        """Test implication that is satisfied."""
        P, Q = sp.symbols("P Q")
        expr = sp.Implies(P, Q)

        # True → True = True
        predicates = {
            "P": lambda x: torch.tensor([True]),
            "Q": lambda x: torch.tensor([True]),
        }
        checker = ConsistencyChecker(expr, predicates)
        x = torch.randn(1, 10)
        result = checker(x)
        assert result[0].item() is True

        # False → True = True
        predicates = {
            "P": lambda x: torch.tensor([False]),
            "Q": lambda x: torch.tensor([True]),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(x)
        assert result[0].item() is True

        # False → False = True
        predicates = {
            "P": lambda x: torch.tensor([False]),
            "Q": lambda x: torch.tensor([False]),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(x)
        assert result[0].item() is True

    def test_implication_violated(self) -> None:
        """Test implication that is violated."""
        P, Q = sp.symbols("P Q")
        expr = sp.Implies(P, Q)

        # True → False = False
        predicates = {
            "P": lambda x: torch.tensor([True]),
            "Q": lambda x: torch.tensor([False]),
        }
        checker = ConsistencyChecker(expr, predicates)

        x = torch.randn(1, 10)
        result = checker(x)
        assert result[0].item() is False

    def test_equivalence(self) -> None:
        """Test equivalence (biconditional)."""
        P, Q = sp.symbols("P Q")
        expr = sp.Equivalent(P, Q)

        # Both true = satisfied
        predicates = {
            "P": lambda x: torch.tensor([True]),
            "Q": lambda x: torch.tensor([True]),
        }
        checker = ConsistencyChecker(expr, predicates)
        x = torch.randn(1, 10)
        result = checker(x)
        assert result[0].item() is True

        # Both false = satisfied
        predicates = {
            "P": lambda x: torch.tensor([False]),
            "Q": lambda x: torch.tensor([False]),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(x)
        assert result[0].item() is True

        # Different values = violated
        predicates = {
            "P": lambda x: torch.tensor([True]),
            "Q": lambda x: torch.tensor([False]),
        }
        checker = ConsistencyChecker(expr, predicates)
        result = checker(x)
        assert result[0].item() is False


class TestTransitivityConstraint:
    """Test transitivity constraint from NLI paper."""

    def test_transitivity_entailment(self) -> None:
        """Test: E(P,H) ∧ E(H,Z) → E(P,Z)."""
        E_PH, E_HZ, E_PZ = sp.symbols("E_PH E_HZ E_PZ")
        constraint = sp.Implies(sp.And(E_PH, E_HZ), E_PZ)

        # Satisfied: antecedent true, consequent true
        predicates = {
            "E_PH": lambda x: torch.tensor([True]),
            "E_HZ": lambda x: torch.tensor([True]),
            "E_PZ": lambda x: torch.tensor([True]),
        }
        checker = ConsistencyChecker(constraint, predicates)
        x = torch.randn(1, 10)
        result = checker(x)
        assert result[0].item() is True

        # Violated: antecedent true, consequent false
        predicates = {
            "E_PH": lambda x: torch.tensor([True]),
            "E_HZ": lambda x: torch.tensor([True]),
            "E_PZ": lambda x: torch.tensor([False]),
        }
        checker = ConsistencyChecker(constraint, predicates)
        result = checker(x)
        assert result[0].item() is False

        # Satisfied: antecedent false (vacuously true)
        predicates = {
            "E_PH": lambda x: torch.tensor([False]),
            "E_HZ": lambda x: torch.tensor([True]),
            "E_PZ": lambda x: torch.tensor([False]),
        }
        checker = ConsistencyChecker(constraint, predicates)
        result = checker(x)
        assert result[0].item() is True

    def test_transitivity_contradiction(self) -> None:
        """Test: E(P,H) ∧ C(H,Z) → C(P,Z)."""
        E_PH, C_HZ, C_PZ = sp.symbols("E_PH C_HZ C_PZ")
        constraint = sp.Implies(sp.And(E_PH, C_HZ), C_PZ)

        # Example from paper: should be violated
        predicates = {
            "E_PH": lambda x: torch.tensor([True]),
            "C_HZ": lambda x: torch.tensor([True]),
            "C_PZ": lambda x: torch.tensor([False]),
        }
        checker = ConsistencyChecker(constraint, predicates)
        x = torch.randn(1, 10)
        result = checker(x)
        assert result[0].item() is False


class TestSymmetryConstraint:
    """Test symmetry constraint from NLI paper."""

    def test_symmetry_equivalence(self) -> None:
        """Test: C(P,H) ↔ C(H,P)."""
        C_PH, C_HP = sp.symbols("C_PH C_HP")
        constraint = sp.Equivalent(C_PH, C_HP)

        # Both true = satisfied
        predicates = {
            "C_PH": lambda x: torch.tensor([True]),
            "C_HP": lambda x: torch.tensor([True]),
        }
        checker = ConsistencyChecker(constraint, predicates)
        x = torch.randn(1, 10)
        result = checker(x)
        assert result[0].item() is True

        # Both false = satisfied
        predicates = {
            "C_PH": lambda x: torch.tensor([False]),
            "C_HP": lambda x: torch.tensor([False]),
        }
        checker = ConsistencyChecker(constraint, predicates)
        result = checker(x)
        assert result[0].item() is True

        # Mismatch = violated
        predicates = {
            "C_PH": lambda x: torch.tensor([True]),
            "C_HP": lambda x: torch.tensor([False]),
        }
        checker = ConsistencyChecker(constraint, predicates)
        x = torch.randn(1, 10)
        result = checker(x)
        assert result[0].item() is False


class TestBatchConsistencyChecking:
    """Test consistency checking with batches (tensors)."""

    def test_batch_and_formula(self) -> None:
        """Test AND formula with batch of decisions."""
        P, Q = sp.symbols("P Q")
        expr = sp.And(P, Q)

        predicates = {
            "P": lambda x: torch.tensor([True, True, False, True]),
            "Q": lambda x: torch.tensor([True, False, True, False]),
        }
        checker = ConsistencyChecker(expr, predicates)

        x = torch.randn(4, 10)
        result = checker(x)
        expected = torch.tensor([True, False, False, False])
        assert torch.equal(result, expected)

    def test_batch_implication(self) -> None:
        """Test implication with batch of decisions."""
        P, Q = sp.symbols("P Q")
        expr = sp.Implies(P, Q)

        predicates = {
            "P": lambda x: torch.tensor([True, False, True, False]),
            "Q": lambda x: torch.tensor([True, True, False, False]),
        }
        checker = ConsistencyChecker(expr, predicates)

        x = torch.randn(4, 10)
        # True→True, False→True, True→False, False→False
        result = checker(x)
        expected = torch.tensor([True, True, False, True])
        assert torch.equal(result, expected)

    def test_consistency_counting(self) -> None:
        """Test counting satisfied examples in a batch."""
        P, Q = sp.symbols("P Q")
        expr = sp.And(P, Q)

        predicates = {
            "P": lambda x: torch.tensor([True, True, False, True, True]),
            "Q": lambda x: torch.tensor([True, False, True, False, True]),
        }
        checker = ConsistencyChecker(expr, predicates)

        x = torch.randn(5, 10)
        satisfied = checker(x)

        # Count satisfied
        num_satisfied = satisfied.sum().item()
        assert num_satisfied == 2  # Only examples 0 and 4 satisfy P AND Q

        # Fraction satisfied
        fraction_satisfied = satisfied.float().mean().item()
        assert abs(fraction_satisfied - 0.4) < 1e-6  # 2/5


class TestPredicatesWithModels:
    """Test predicates that use actual model outputs."""

    def test_argmax_predicates(self) -> None:
        """Test predicates based on argmax of model outputs."""
        P, Q = sp.symbols("P Q")
        expr = sp.Implies(P, Q)

        # Simulate model outputs
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
            "P": lambda x: model1_outputs.argmax(dim=-1) == 0,
            "Q": lambda x: model2_outputs.argmax(dim=-1) == 1,
        }
        checker = ConsistencyChecker(expr, predicates)

        x = torch.randn(3, 10)
        result = checker(x)

        # P: [True, False, True]
        # Q: [True, False, True]
        # P → Q: [True, True, True]
        expected = torch.tensor([True, True, True])
        assert torch.equal(result, expected)


class TestBooleanConstants:
    """Test boolean constants in formulas."""

    def test_true_constant(self) -> None:
        """Test true constant."""
        P = sp.symbols("P")
        expr = sp.And(sp.true, P)

        predicates = {
            "P": lambda x: torch.tensor([True, False, True]),
        }
        checker = ConsistencyChecker(expr, predicates)

        x = torch.randn(3, 10)
        result = checker(x)
        expected = torch.tensor([True, False, True])
        assert torch.equal(result, expected)

    def test_false_constant(self) -> None:
        """Test false constant."""
        P = sp.symbols("P")
        expr = sp.Or(sp.false, P)

        predicates = {
            "P": lambda x: torch.tensor([True, False, True]),
        }
        checker = ConsistencyChecker(expr, predicates)

        x = torch.randn(3, 10)
        result = checker(x)
        expected = torch.tensor([True, False, True])
        assert torch.equal(result, expected)


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_missing_predicate_error(self) -> None:
        """Test error when predicate is missing."""
        P, Q = sp.symbols("P Q")
        expr = sp.And(P, Q)

        predicates = {"P": lambda x: torch.tensor([True])}

        with pytest.raises(ValueError, match="Missing predicates for symbols"):
            ConsistencyChecker(expr, predicates)

    def test_non_boolean_tensor_conversion(self) -> None:
        """Test automatic conversion to boolean tensor."""
        P = sp.symbols("P")
        expr = P

        # Predicate returns float tensor
        predicates = {
            "P": lambda x: torch.tensor([1.0, 0.0, 1.0]),
        }
        checker = ConsistencyChecker(expr, predicates)

        x = torch.randn(3, 10)
        result = checker(x)

        # Should convert to boolean
        expected = torch.tensor([True, False, True])
        assert torch.equal(result, expected)
