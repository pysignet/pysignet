"""Tests for the ConsistencyChecker class.

This module tests hard (boolean) formula evaluation and consistency
measurement using the eval module's ConsistencyChecker.
"""

# pylint: disable=invalid-name

import warnings

import pytest
import sympy as sp
import torch
import torch.nn as nn

from pysignet import ConsistencyChecker, Predicate, Symbol
from pysignet.api import consistency_report
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


class TestMulticlassModulePredicates:
    """Test consistency checker with multiclass nn.Module predicates.

    These tests verify that the checker correctly splits variables
    into model inputs vs output indices for nn.Module predicates,
    mirroring the logic in the compilation path (base.py).
    """

    def test_multiclass_variable_index(self) -> None:
        """Test Digit(X, Y) with multiclass model and variable Y.

        This reproduces the MNIST notebook scenario: a model that
        takes one input and produces 10-class output, where Y is
        a per-element class label variable.
        """
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")
        expr = Digit(X, Y)

        # Simple 10-class model: input -> 10 logits
        model = nn.Sequential(
            nn.Linear(4, 10),
        )

        predicates = {"Digit": Predicate(model)}
        checker = ConsistencyChecker(expr, predicates)

        # Batch of 3 inputs
        torch.manual_seed(42)
        x = torch.randn(3, 4)

        # Get model predictions to determine expected results
        with torch.no_grad():
            logits = model(x)
            predicted = logits.argmax(dim=-1)

        # Y matches predicted classes -> should be satisfied
        result = checker(X=x, Y=predicted)
        assert result.all(), (
            "Should be satisfied when Y matches argmax"
        )

        # Y is wrong for all -> should be violated
        wrong_y = (predicted + 1) % 10
        result = checker(X=x, Y=wrong_y)
        assert not result.any(), (
            "Should be violated when Y never matches argmax"
        )

    def test_multiclass_constant_index(self) -> None:
        """Test Digit(X, 3) with constant class index."""
        Digit = Symbol("Digit")
        X = Variable("X")
        expr = Digit(X, 3)

        model = nn.Sequential(
            nn.Linear(4, 10),
        )

        predicates = {"Digit": Predicate(model)}
        checker = ConsistencyChecker(expr, predicates)

        torch.manual_seed(42)
        x = torch.randn(3, 4)

        with torch.no_grad():
            logits = model(x)
            predicted = logits.argmax(dim=-1)

        result = checker(X=x)
        expected = predicted == 3
        assert torch.equal(result, expected)

    def test_multiclass_mixed_satisfied_violated(self) -> None:
        """Test that variable index gives correct per-element results.

        Some examples match argmax, some do not.
        """
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")
        expr = Digit(X, Y)

        model = nn.Sequential(
            nn.Linear(4, 10),
        )

        predicates = {"Digit": Predicate(model)}
        checker = ConsistencyChecker(expr, predicates)

        torch.manual_seed(42)
        x = torch.randn(4, 4)

        with torch.no_grad():
            logits = model(x)
            predicted = logits.argmax(dim=-1)

        # Mix correct and incorrect labels
        y = predicted.clone()
        y[1] = (y[1] + 1) % 10  # Make second wrong
        y[3] = (y[3] + 1) % 10  # Make fourth wrong

        result = checker(X=x, Y=y)
        expected = torch.tensor([True, False, True, False])
        assert torch.equal(result, expected)

    def test_multiclass_in_formula(self) -> None:
        """Test multiclass predicate inside a logical formula.

        Exists(Y, range(10), Digit(X, Y)) should always be True
        since argmax always picks some class.
        """
        from pysignet.logic import Exists

        Digit = Symbol("Digit")
        X, Y = Variable("X Y")
        expr = Exists(Y, range(10), Digit(X, Y))

        model = nn.Sequential(
            nn.Linear(4, 10),
        )

        predicates = {"Digit": Predicate(model)}
        checker = ConsistencyChecker(expr, predicates)

        torch.manual_seed(42)
        x = torch.randn(3, 4)

        result = checker(X=x)
        assert result.all(), (
            "Exists over all classes should always be True"
        )

    def test_custom_module_activation_configured(self) -> None:
        """Test that custom (non-Sequential) modules get activation.

        Custom nn.Modules cannot auto-detect activation from
        structure. The checker must configure activation using
        expression-context arity so that softmax is applied
        before argmax, not clamping (which corrupts argmax
        when logits exceed 1.0).
        """

        class CustomClassifier(nn.Module):
            """Custom module that returns raw logits."""

            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        Digit = Symbol("Digit")
        X, Y = Variable("X Y")
        expr = Digit(X, Y)

        torch.manual_seed(42)
        model = CustomClassifier()

        predicates = {"Digit": Predicate(model)}
        checker = ConsistencyChecker(expr, predicates)

        x = torch.randn(8, 4)

        # Ground truth: argmax of raw logits (softmax is
        # monotonic so argmax is the same on logits)
        with torch.no_grad():
            logits = model(x)
            predicted = logits.argmax(dim=-1)

        # Consistency with correct labels should equal accuracy
        result = checker(X=x, Y=predicted)
        assert result.all(), (
            "All examples should be satisfied when Y matches "
            "the model's argmax prediction"
        )

    def test_consistency_matches_accuracy_via_api(self) -> None:
        """Test that consistency equals accuracy through the API.

        Uses consistency_report (the public API) with a custom
        module. Verifies no activation warning is emitted and
        that consistency matches accuracy exactly.
        """

        class Classifier(nn.Module):
            """Custom module returning raw logits."""

            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        Digit = Symbol("Digit")
        X, Y = Variable("X Y")
        expr = Digit(X, Y)

        torch.manual_seed(42)
        model = Classifier()
        x = torch.randn(50, 4)

        with torch.no_grad():
            predicted = model(x).argmax(dim=-1)

        # Make some labels wrong so accuracy < 1
        y = predicted.clone()
        y[:10] = (y[:10] + 1) % 10
        accuracy = (predicted == y).float().mean().item()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            report = consistency_report(
                expr, {"Digit": model}
            )
            report.eval(X=x, Y=y)

            # No activation warning should be emitted
            activation_warnings = [
                x for x in w
                if "Could not determine activation" in str(
                    x.message
                )
            ]
            assert len(activation_warnings) == 0, (
                "No activation warning should be emitted"
            )

        consistency = report.global_consistency()
        assert abs(accuracy - consistency) < 1e-6, (
            f"Consistency {consistency:.4f} should match "
            f"accuracy {accuracy:.4f}"
        )

    def test_binary_model_not_affected(self) -> None:
        """Test that binary models (single output) still work.

        P(X) with a binary model should not be affected by the
        variable index splitting logic.
        """
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        model = nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

        predicates = {"P": Predicate(model)}
        checker = ConsistencyChecker(expr, predicates)

        torch.manual_seed(42)
        x = torch.randn(3, 4)

        with torch.no_grad():
            out = model(x).squeeze(-1)
            expected = out > 0.5

        result = checker(X=x)
        assert torch.equal(result, expected)
