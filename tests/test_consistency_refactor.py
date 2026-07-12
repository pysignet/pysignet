"""Tests for the refactored ConsistencyChecker in pysignet.eval.

Tests verify that ConsistencyChecker:
- Accepts Predicate objects (not just raw callables)
- Handles ForAll/Exists quantifiers internally
- Uses correct boolean conversion rules:
  - Binary (sigmoid): threshold at 0.5
  - Multiclass (softmax): argmax == class_idx
  - Others: threshold at 0.5
- Produces boolean decisions matching standard classification rules
  (so accuracy computed via checker == accuracy computed manually)
"""

# pylint: disable=invalid-name

import sympy as sp
import torch
import torch.nn as nn

from pysignet import Predicate, Symbol, Variable, compile_logic
from pysignet.eval import ConsistencyChecker, to_boolean
from pysignet.logic import Exists, ForAll


class TestAccuracyEqualsConsistency:
    """Core invariant: checker decisions match standard hard decisions.

    For binary: checker(P(X)) == (model(x) > 0.5)
    For multiclass: checker(Digit(X, k)) == (model(x).argmax() == k)
    For multi-input: checker(P(X, Y)) == (func(x, y) > 0.5)
    """

    def test_binary_accuracy_equals_consistency(self) -> None:
        """Binary model: boolean decisions match threshold at 0.5."""
        model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        x = torch.randn(50, 10)

        # Manual hard decisions
        with torch.no_grad():
            soft = model(x).squeeze(-1)
        hard_manual = soft > 0.5

        # Via ConsistencyChecker
        checker = ConsistencyChecker(
            expr, {"P": Predicate(model)}
        )
        hard_checker = checker(X=x)

        assert hard_checker.dtype == torch.bool
        assert torch.equal(hard_manual, hard_checker)

        # Therefore accuracy matches for any labels
        labels = torch.randint(0, 2, (50,)).bool()
        acc_manual = (hard_manual == labels).float().mean()
        acc_checker = (hard_checker == labels).float().mean()
        assert acc_manual == acc_checker

    def test_multiclass_accuracy_equals_consistency(self) -> None:
        """Multiclass model: boolean decisions match argmax."""
        model = nn.Sequential(
            nn.Linear(10, 5), nn.Softmax(dim=-1)
        )
        Digit = Symbol("Digit")
        X = Variable("X")

        x = torch.randn(50, 10)

        # Manual hard decisions via argmax
        with torch.no_grad():
            probs = model(x)
        preds = probs.argmax(dim=-1)

        # Via ConsistencyChecker for each class
        for k in range(5):
            expr = Digit(X, k)
            checker = ConsistencyChecker(
                expr, {"Digit": Predicate(model)}
            )
            result = checker(X=x)
            expected = preds == k
            assert torch.equal(result, expected), (
                f"Class {k}: checker disagrees with argmax"
            )

    def test_multi_input_accuracy_equals_consistency(self) -> None:
        """Multi-input function: decisions match threshold."""
        P = Symbol("P")
        X, Y = Variable("X Y")
        expr = P(X, Y)

        def similarity(x, y):
            return torch.sigmoid(torch.sum(x * y, dim=-1))

        pred = Predicate(similarity, is_model=False)

        x = torch.randn(50, 10)
        y = torch.randn(50, 10)

        # Manual hard decisions
        with torch.no_grad():
            soft = similarity(x, y)
        hard_manual = soft > 0.5

        # Via ConsistencyChecker
        checker = ConsistencyChecker(expr, {"P": pred})
        hard_checker = checker(X=x, Y=y)

        assert hard_checker.dtype == torch.bool
        assert torch.equal(hard_manual, hard_checker)


class TestMulticlassArgmaxVsThreshold:
    """Verify argmax is used for model predicates, not threshold."""

    def test_argmax_for_model(self) -> None:
        """Model with argmax: class with highest prob wins.

        Even when all probs are below 0.5, the argmax class is True.
        """
        Digit = Symbol("Digit")
        X = Variable("X")

        # Model that outputs specific probabilities
        class FixedModel(nn.Module):
            def forward(self, x):
                batch = x.shape[0]
                # [0.3, 0.4, 0.3] -- class 1 is argmax (< 0.5)
                return torch.tensor(
                    [[0.3, 0.4, 0.3]] * batch
                )

        model = FixedModel()

        # Digit(X, 1) should be True (argmax == 1)
        checker1 = ConsistencyChecker(
            Digit(X, 1), {"Digit": Predicate(model)}
        )
        result1 = checker1(X=torch.randn(4, 10))
        assert result1.all(), (
            "Class 1 is argmax, should be True"
        )

        # Digit(X, 0) should be False (argmax != 0)
        checker0 = ConsistencyChecker(
            Digit(X, 0), {"Digit": Predicate(model)}
        )
        result0 = checker0(X=torch.randn(4, 10))
        assert not result0.any(), (
            "Class 0 is not argmax, should be False"
        )

    def test_threshold_for_function(self) -> None:
        """Function predicate: always threshold at 0.5.

        Same values but via function -- uses threshold, not argmax.
        """
        P = Symbol("P")
        X = Variable("X")

        # Function that returns 0.4 for any class
        def p_func(x, _class_idx):
            batch = x.shape[0]
            return torch.ones(batch) * 0.4

        pred = Predicate(p_func, is_model=False)

        # P(X, 1) should be False (0.4 < 0.5)
        checker = ConsistencyChecker(
            P(X, 1), {"P": pred}
        )
        result = checker(X=torch.randn(4, 10))
        assert not result.any(), (
            "0.4 < 0.5 threshold, should be False"
        )


class TestConsistencyWithPredicateObjects:
    """Test ConsistencyChecker accepts Predicate objects."""

    def test_binary_predicate_object(self) -> None:
        """Predicate wrapping binary model (sigmoid)."""
        model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
        P = Symbol("P")
        X = Variable("X")

        checker = ConsistencyChecker(
            P(X), {"P": Predicate(model)}
        )
        result = checker(X=torch.randn(8, 5))

        assert result.dtype == torch.bool
        assert result.shape == (8,)

    def test_multiclass_predicate_argmax(self) -> None:
        """Predicate wrapping multiclass model (softmax)."""
        model = nn.Sequential(
            nn.Linear(5, 3), nn.Softmax(dim=-1)
        )
        Digit = Symbol("Digit")
        X = Variable("X")

        checker = ConsistencyChecker(
            Digit(X, 0), {"Digit": Predicate(model)}
        )
        result = checker(X=torch.randn(8, 5))

        assert result.dtype == torch.bool
        assert result.shape == (8,)

    def test_function_predicate_object(self) -> None:
        """Predicate wrapping a function (threshold at 0.5)."""
        def p_func(x):
            return torch.sigmoid(x.sum(dim=-1))

        pred = Predicate(p_func, is_model=False)
        P = Symbol("P")
        X = Variable("X")

        checker = ConsistencyChecker(P(X), {"P": pred})
        result = checker(X=torch.randn(8, 5))

        assert result.dtype == torch.bool
        assert result.shape == (8,)


class TestConsistencyWithQuantifiers:
    """Test that ConsistencyChecker handles ForAll/Exists directly."""

    def test_forall_direct(self) -> None:
        """ForAll passed directly -- no manual expansion needed."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        # ForAll Y in {0,1,2}: Digit(X, Y)
        expr = ForAll(Y, [0, 1, 2], Digit(X, Y))

        # Function where all classes > 0.5
        def digit_func(x, _class_idx):
            batch = x.shape[0]
            return torch.ones(batch) * 0.9

        pred = Predicate(digit_func, is_model=False)
        checker = ConsistencyChecker(expr, {"Digit": pred})
        result = checker(X=torch.randn(4, 10))

        assert result.dtype == torch.bool
        assert result.all()

    def test_exists_direct(self) -> None:
        """Exists passed directly -- no manual expansion needed."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        # Exists Y in {0,1,2}: Digit(X, Y)
        expr = Exists(Y, [0, 1, 2], Digit(X, Y))

        # Only class 1 exceeds 0.5
        def digit_func(x, class_idx):
            batch = x.shape[0]
            if class_idx == 1:
                return torch.ones(batch) * 0.9
            return torch.ones(batch) * 0.1

        pred = Predicate(digit_func, is_model=False)
        checker = ConsistencyChecker(expr, {"Digit": pred})
        result = checker(X=torch.randn(4, 10))

        assert result.dtype == torch.bool
        assert result.all()

    def test_exactly_one_with_model(self) -> None:
        """Exactly-one constraint with nn.Module Predicate.

        Uses argmax for multiclass: exactly one class wins.
        """
        Digit = Symbol("Digit")
        X, Y, I, J = Variable("X Y I J")

        n_classes = 3
        at_least_one = Exists(
            Y, range(n_classes), Digit(X, Y)
        )
        all_pairs = [
            (i, j) for i in range(n_classes)
            for j in range(i + 1, n_classes)
        ]
        at_most_one = ForAll(
            [I, J], all_pairs,
            sp.Not(sp.And(Digit(X, I), Digit(X, J)))
        )
        exactly_one = sp.And(at_least_one, at_most_one)

        # Multiclass model: argmax always picks exactly one
        model = nn.Sequential(
            nn.Linear(10, n_classes), nn.Softmax(dim=-1)
        )
        checker = ConsistencyChecker(
            exactly_one, {"Digit": Predicate(model)}
        )
        result = checker(X=torch.randn(8, 10))

        # Argmax always produces exactly one winner
        assert result.all()


class TestConsistencyWithVariableBindings:
    """Test keyword variable bindings."""

    def test_kwarg_bindings(self) -> None:
        """Single variable via keyword arg."""
        P = Symbol("P")
        X = Variable("X")

        def p_func(x):
            return torch.sigmoid(x.sum(dim=-1))

        checker = ConsistencyChecker(
            P(X), {"P": Predicate(p_func, is_model=False)}
        )
        result = checker(X=torch.randn(4, 5))

        assert result.dtype == torch.bool
        assert result.shape == (4,)

    def test_multi_variable_kwargs(self) -> None:
        """Multiple variables via keyword args."""
        P = Symbol("P")
        X, Y = Variable("X Y")

        def p_func(x, y):
            return torch.sigmoid(
                torch.sum(x * y, dim=-1)
            )

        checker = ConsistencyChecker(
            P(X, Y),
            {"P": Predicate(p_func, is_model=False)},
        )
        result = checker(
            X=torch.randn(4, 5), Y=torch.randn(4, 5)
        )

        assert result.dtype == torch.bool
        assert result.shape == (4,)


class TestCompiledExpressionThinWrapper:
    """Test that compiled(return_boolean=True) uses new checker."""

    def test_return_boolean_still_works(self) -> None:
        """compiled(X=x, return_boolean=True) still works."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        def p_func(x):
            return torch.sigmoid(x.sum(dim=-1))

        compiled = compile_logic(
            expr, {"P": p_func}
        )
        x = torch.randn(8, 5)
        result = compiled(X=x, return_boolean=True)

        assert result.dtype == torch.bool
        assert result.shape == (8,)

    def test_return_boolean_multiclass_argmax(self) -> None:
        """return_boolean with multiclass model uses argmax."""
        Digit = Symbol("Digit")
        X = Variable("X")

        class FixedModel(nn.Module):
            def forward(self, x):
                batch = x.shape[0]
                # Class 2 is argmax (0.45 > 0.3, 0.25)
                return torch.tensor(
                    [[0.3, 0.25, 0.45]] * batch
                )

        expr = Digit(X, 2)
        compiled = compile_logic(
            expr, {"Digit": Predicate(FixedModel())}
        )
        x = torch.randn(4, 10)
        result = compiled(X=x, return_boolean=True)

        # Class 2 is argmax -> True for all
        assert result.all()


class TestToBooleanFunction:
    """Test the standalone to_boolean conversion function."""

    def test_already_boolean(self) -> None:
        """Boolean input returned as-is."""
        x = torch.tensor([True, False, True])
        result = to_boolean(x)
        assert torch.equal(result, x)

    def test_binary_threshold(self) -> None:
        """1D float tensor thresholded at 0.5."""
        x = torch.tensor([0.3, 0.6, 0.5, 0.51])
        result = to_boolean(x)
        expected = torch.tensor([False, True, False, True])
        assert torch.equal(result, expected)

    def test_multiclass_argmax_with_class_idx(self) -> None:
        """2D float tensor with class_idx uses argmax."""
        # (batch=2, classes=3)
        x = torch.tensor([
            [0.3, 0.4, 0.3],  # argmax = 1
            [0.5, 0.3, 0.2],  # argmax = 0
        ])
        # Class 1: True for batch 0, False for batch 1
        result = to_boolean(x, class_idx=1)
        expected = torch.tensor([True, False])
        assert torch.equal(result, expected)

    def test_multiclass_argmax_no_class_idx(self) -> None:
        """2D float without class_idx: max > 0.5."""
        x = torch.tensor([
            [0.3, 0.4, 0.3],  # max = 0.4 < 0.5
            [0.1, 0.8, 0.1],  # max = 0.8 > 0.5
        ])
        result = to_boolean(x)
        expected = torch.tensor([False, True])
        assert torch.equal(result, expected)

    def test_squeeze_batch_1(self) -> None:
        """(batch, 1) tensor gets squeezed then thresholded."""
        x = torch.tensor([[0.3], [0.7]])
        result = to_boolean(x)
        expected = torch.tensor([False, True])
        assert torch.equal(result, expected)
