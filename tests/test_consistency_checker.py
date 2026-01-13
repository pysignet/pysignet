"""Tests for the ConsistencyChecker class.

This module tests hard (boolean) formula evaluation and consistency
measurement as described in the new simplified interface.
"""

import pytest
import sympy as sp
import torch

from pysignet import ConsistencyChecker, Symbol
from pysignet.logic import Variable


class TestBasicConsistencyChecking:
    """Test basic consistency checking with boolean predicates."""

    def test_simple_and_satisfied(self) -> None:
        """Test AND formula that is satisfied."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))

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
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))

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
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Or(P(X), Q(X))

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
        P = Symbol("P")
        X = Variable("X")
        expr = sp.Not(P(X))

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
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Implies(P(X), Q(X))

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
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Implies(P(X), Q(X))

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
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Equivalent(P(X), Q(X))

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
        E = Symbol("E")
        constraint = sp.Implies(sp.And(E("P", "H"), E("H", "Z")), E("P", "Z"))

        # Satisfied: antecedent true, consequent true
        predicates = {
            "E": lambda p, h: torch.tensor([True]),
        }
        checker = ConsistencyChecker(constraint, predicates)
        x = torch.randn(1, 10)
        result = checker(x)
        assert result[0].item() is True

        # Violated: antecedent true, consequent false
        def e_violated(p, h):
            # E(P,H) = True, E(H,Z) = True, E(P,Z) = False
            if p == "P" and h == "H":
                return torch.tensor([True])
            elif p == "H" and h == "Z":
                return torch.tensor([True])
            elif p == "P" and h == "Z":
                return torch.tensor([False])
            return torch.tensor([False])

        predicates = {"E": e_violated}
        checker = ConsistencyChecker(constraint, predicates)
        result = checker(x)
        assert result[0].item() is False

        # Satisfied: antecedent false (vacuously true)
        def e_vacuous(p, h):
            # E(P,H) = False, E(H,Z) = True, E(P,Z) = False
            if p == "P" and h == "H":
                return torch.tensor([False])
            elif p == "H" and h == "Z":
                return torch.tensor([True])
            elif p == "P" and h == "Z":
                return torch.tensor([False])
            return torch.tensor([False])

        predicates = {"E": e_vacuous}
        checker = ConsistencyChecker(constraint, predicates)
        result = checker(x)
        assert result[0].item() is True

    def test_transitivity_contradiction(self) -> None:
        """Test: E(P,H) ∧ C(H,Z) → C(P,Z)."""
        E, C = Symbol("E C")
        constraint = sp.Implies(sp.And(E("P", "H"), C("H", "Z")), C("P", "Z"))

        # Example from paper: should be violated
        def e_func(p, h):
            if p == "P" and h == "H":
                return torch.tensor([True])
            return torch.tensor([False])

        def c_func(p, h):
            if p == "H" and h == "Z":
                return torch.tensor([True])
            elif p == "P" and h == "Z":
                return torch.tensor([False])
            return torch.tensor([False])

        predicates = {
            "E": e_func,
            "C": c_func,
        }
        checker = ConsistencyChecker(constraint, predicates)
        x = torch.randn(1, 10)
        result = checker(x)
        assert result[0].item() is False


class TestSymmetryConstraint:
    """Test symmetry constraint from NLI paper."""

    def test_symmetry_equivalence(self) -> None:
        """Test: C(P,H) ↔ C(H,P)."""
        C = Symbol("C")
        constraint = sp.Equivalent(C("P", "H"), C("H", "P"))

        # Both true = satisfied
        predicates = {
            "C": lambda p, h: torch.tensor([True]),
        }
        checker = ConsistencyChecker(constraint, predicates)
        x = torch.randn(1, 10)
        result = checker(x)
        assert result[0].item() is True

        # Both false = satisfied
        predicates = {
            "C": lambda p, h: torch.tensor([False]),
        }
        checker = ConsistencyChecker(constraint, predicates)
        result = checker(x)
        assert result[0].item() is True

        # Mismatch = violated
        def c_mismatch(p, h):
            if p == "P" and h == "H":
                return torch.tensor([True])
            elif p == "H" and h == "P":
                return torch.tensor([False])
            return torch.tensor([False])

        predicates = {
            "C": c_mismatch,
        }
        checker = ConsistencyChecker(constraint, predicates)
        x = torch.randn(1, 10)
        result = checker(x)
        assert result[0].item() is False


class TestBatchConsistencyChecking:
    """Test consistency checking with batches (tensors)."""

    def test_batch_and_formula(self) -> None:
        """Test AND formula with batch of decisions."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))

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
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Implies(P(X), Q(X))

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
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))

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
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Implies(P(X), Q(X))

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
        P = Symbol("P")
        X = Variable("X")
        expr = sp.And(sp.true, P(X))

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
        P = Symbol("P")
        X = Variable("X")
        expr = sp.Or(sp.false, P(X))

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
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))

        predicates = {"P": lambda x: torch.tensor([True])}

        with pytest.raises(ValueError, match="Missing predicates for symbols"):
            ConsistencyChecker(expr, predicates)

    def test_non_boolean_tensor_conversion(self) -> None:
        """Test automatic conversion to boolean tensor."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

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

    def test_consistency_with_dict_input_specific_predicate(self):
        """ConsistencyChecker works with dict input specifying predicate names."""
        X = Variable("X")
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))

        predicates = {
            "P": lambda x: torch.tensor([True, False, True]),
            "Q": lambda x: torch.tensor([True, True, False]),
        }
        checker = ConsistencyChecker(expr, predicates)

        # Dict input with specific predicate names
        inputs = {
            "P": torch.randn(3, 10),
            "Q": torch.randn(3, 10),
        }
        result = checker(inputs)

        # P AND Q
        expected = torch.tensor([True, False, False])
        assert torch.equal(result, expected)

    def test_consistency_with_true_constant(self):
        """ConsistencyChecker handles sp.true constant."""
        X = Variable("X")
        P = Symbol("P")
        X = Variable("X")
        # Expression: P(X) AND true
        expr = sp.And(P(X), sp.true)

        predicates = {
            "P": lambda x: torch.tensor([True, False, True]),
        }
        checker = ConsistencyChecker(expr, predicates)

        x = torch.randn(3, 10)
        result = checker({"P": x})

        # P AND true = P
        expected = torch.tensor([True, False, True])
        assert torch.equal(result, expected)

    def test_consistency_with_false_constant(self):
        """ConsistencyChecker handles sp.false constant."""
        X = Variable("X")
        P = Symbol("P")
        X = Variable("X")
        # Expression: P(X) OR false
        expr = sp.Or(P(X), sp.false)

        predicates = {
            "P": lambda x: torch.tensor([True, False, True]),
        }
        checker = ConsistencyChecker(expr, predicates)

        x = torch.randn(3, 10)
        result = checker({"P": x})

        # P OR false = P
        expected = torch.tensor([True, False, True])
        assert torch.equal(result, expected)

    def test_consistency_unsupported_expression_type(self):
        """ConsistencyChecker raises ValueError for unsupported expressions."""
        import sympy
        X = Variable("X")

        # Create an unsupported SymPy expression (e.g., Add for arithmetic)
        unsupported_expr = sympy.Add(X, X)

        predicates = {}
        checker = ConsistencyChecker(unsupported_expr, predicates)

        x = torch.randn(3, 10)
        with pytest.raises(ValueError, match="Unsupported expression type"):
            checker(x)
