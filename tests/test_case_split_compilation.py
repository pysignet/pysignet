"""Tests for case-split compilation (SEMANTIC_LOSS_DESIGN.md Section 10).

Detects ForAll(S, domain, Implies(Cond(S), Body(S))) subtrees -- anywhere
in the expression, not just at the top level -- whose atoms are disjoint
from the rest of the containing formula, and replaces each with a
synthetic ground atom compiled through the ordinary, unmodified
SemanticLossCompiler pipeline. Nesting under any connective, multiple
independent subtrees, and the degenerate whole-formula case are all the
SAME mechanism, not separate code paths.

Cond represents observed per-example hard evidence (e.g. a training
label), not a soft/uncertain prediction -- verified at every batch call,
since (per the design doc's corrected derivation) the decomposition is
only exact for hard 0/1 values. A violation raises ValueError rather
than silently computing a different, wrong number.

Value-correctness tests are checked against direct hand computation
(WMC of Body(s_true) for whichever s is the true per-example label,
using the module's own existing brute-force-style helper for Body's own
internal structure) -- valid because Cond is hard evidence, so the
And-of-Implies collapses deterministically per example.
"""

import itertools

import pytest
import sympy as sp
import torch
import torch.nn as nn

from pysignet import Symbol, Variable, compile_logic, logic_to_loss
from pysignet.logic import ForAll


def _const_pred(value: float):
    def pred(x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), value, dtype=torch.float64)

    return pred


def _hard_label_pred(true_labels: torch.Tensor):
    """A predicate representing observed per-example hard evidence:
    Cat(labels, s) is 1.0 where labels == s, else 0.0."""

    def pred(labels: torch.Tensor, s: int) -> torch.Tensor:
        return (labels == s).to(torch.float64)

    return pred


def _brute_force_wmc(bool_fn, probs: dict[str, float]) -> float:
    names = list(probs)
    total = 0.0
    for bits in itertools.product([False, True], repeat=len(names)):
        assignment = dict(zip(names, bits, strict=True))
        if bool_fn(**assignment):
            weight = 1.0
            for name in names:
                weight *= (
                    probs[name] if assignment[name] else (1 - probs[name])
                )
            total += weight
    return total


class TestDegenerateTopLevelCase:
    """The whole compiled expression IS the case-split subtree."""

    def test_matches_body_of_true_label_per_example(self):
        X, Labels = Variable("X Labels")
        Digit, Cat = Symbol("Digit Cat")
        S = Variable("S")
        domain = [0, 1, 2]
        expr = ForAll(S, domain, sp.Implies(Cat(Labels, S), Digit(X, S)))

        digit_probs = [0.2, 0.6, 0.9]  # Digit(X, s) for s = 0, 1, 2

        def digit_pred_indexed(x: torch.Tensor, s: int) -> torch.Tensor:
            return torch.full(
                (x.shape[0],), digit_probs[s], dtype=torch.float64
            )

        true_labels = torch.tensor([0, 1, 2, 1], dtype=torch.float64)
        x = torch.zeros(4, 1, dtype=torch.float64)

        compiled = compile_logic(
            expr,
            {
                "Cat": _hard_label_pred(true_labels),
                "Digit": digit_pred_indexed,
            },
            mode="semantic",
        )
        result = compiled(X=x, Labels=true_labels)

        expected = torch.tensor(
            [digit_probs[s] for s in [0, 1, 2, 1]], dtype=torch.float64
        )
        assert torch.allclose(result, expected, atol=1e-9)


class TestCrossBranchAtomSharing:
    """Body(s) sharing atoms ACROSS branches (e.g. Digit(X1,I) recurring
    over many Sum values in the real MNIST Addition constraint) must
    not affect correctness -- only sharing with content OUTSIDE the
    case-split subtree is disallowed."""

    def test_common_atom_shared_across_all_branches(self):
        X, Labels = Variable("X Labels")
        Digit, Common, Cat = Symbol("Digit Common Cat")
        S = Variable("S")
        domain = [0, 1, 2]
        expr = ForAll(
            S, domain, sp.Implies(Cat(Labels, S), sp.And(Digit(X, S), Common(X)))
        )

        digit_probs = [0.3, 0.5, 0.8]
        common_p = 0.4

        def digit_pred_indexed(x: torch.Tensor, s: int) -> torch.Tensor:
            return torch.full(
                (x.shape[0],), digit_probs[s], dtype=torch.float64
            )

        true_labels = torch.tensor([0, 2], dtype=torch.float64)
        x = torch.zeros(2, 1, dtype=torch.float64)

        compiled = compile_logic(
            expr,
            {
                "Cat": _hard_label_pred(true_labels),
                "Digit": digit_pred_indexed,
                "Common": _const_pred(common_p),
            },
            mode="semantic",
        )
        result = compiled(X=x, Labels=true_labels)

        expected = torch.tensor(
            [digit_probs[0] * common_p, digit_probs[2] * common_p],
            dtype=torch.float64,
        )
        assert torch.allclose(result, expected, atol=1e-9)


def _case_split_formula(digit_probs, common_p, cat_name="Cat", var_suffix=""):
    """Build ForAll(S, domain, Implies(Cat(Labels,S), And(Digit(X,S),
    Common(X)))) with unique symbol/variable names per suffix, so
    multiple independent instances can be combined in one expression
    without accidental atom collisions."""
    X = Variable(f"X{var_suffix}")
    Labels = Variable(f"Labels{var_suffix}")
    Digit = Symbol(f"Digit{var_suffix}")
    Common = Symbol(f"Common{var_suffix}")
    Cat = Symbol(cat_name)
    S = Variable(f"S{var_suffix}")
    domain = list(range(len(digit_probs)))
    expr = ForAll(
        S, domain, sp.Implies(Cat(Labels, S), sp.And(Digit(X, S), Common(X)))
    )

    def digit_pred_indexed(x: torch.Tensor, s: int) -> torch.Tensor:
        return torch.full((x.shape[0],), digit_probs[s], dtype=torch.float64)

    return expr, {
        "digit_pred": digit_pred_indexed,
        "common_pred": _const_pred(common_p),
        "digit_name": f"Digit{var_suffix}",
        "common_name": f"Common{var_suffix}",
        "cat_name": cat_name,
        "x_name": f"X{var_suffix}",
        "labels_name": f"Labels{var_suffix}",
    }


class TestNestedUnderConnectives:
    """The same mechanism handles nesting under any connective, since
    the case-split subtree becomes an ordinary synthetic ground atom
    combined via PySDD's own already-correct &/|/~/.equiv()."""

    def _build(self, true_label, digit_probs=(0.3, 0.7), common_p=0.6):
        expr, info = _case_split_formula(list(digit_probs), common_p)
        Outer = Symbol("Outer")
        Y = Variable("Y")
        outer_p = 0.9
        true_labels = torch.tensor([float(true_label)], dtype=torch.float64)
        x = torch.zeros(1, 1, dtype=torch.float64)
        y = torch.zeros(1, 1, dtype=torch.float64)
        predicates = {
            info["cat_name"]: _hard_label_pred(true_labels),
            info["digit_name"]: info["digit_pred"],
            info["common_name"]: info["common_pred"],
            "Outer": _const_pred(outer_p),
        }
        case_split_value = digit_probs[true_label] * common_p
        bindings = {
            info["x_name"]: x,
            info["labels_name"]: true_labels,
            "Y": y,
        }
        return expr, Outer, Y, predicates, bindings, case_split_value, outer_p

    def test_and_with_outer_atom(self):
        expr, Outer, Y, predicates, bindings, csv, outer_p = self._build(1)
        full_expr = sp.And(expr, Outer(Y))
        compiled = compile_logic(full_expr, predicates, mode="semantic")
        result = compiled(**bindings)
        assert torch.allclose(
            result, torch.tensor([csv * outer_p], dtype=torch.float64),
            atol=1e-9,
        )

    def test_or_with_outer_atom(self):
        expr, Outer, Y, predicates, bindings, csv, outer_p = self._build(0)
        full_expr = sp.Or(expr, Outer(Y))
        compiled = compile_logic(full_expr, predicates, mode="semantic")
        result = compiled(**bindings)
        expected = csv + outer_p - csv * outer_p
        assert torch.allclose(
            result, torch.tensor([expected], dtype=torch.float64), atol=1e-9
        )

    def test_implies_case_split_as_antecedent(self):
        expr, Outer, Y, predicates, bindings, csv, outer_p = self._build(1)
        full_expr = sp.Implies(expr, Outer(Y))
        compiled = compile_logic(full_expr, predicates, mode="semantic")
        result = compiled(**bindings)
        expected = (1 - csv) + csv * outer_p
        assert torch.allclose(
            result, torch.tensor([expected], dtype=torch.float64), atol=1e-9
        )

    def test_implies_case_split_as_consequent(self):
        expr, Outer, Y, predicates, bindings, csv, outer_p = self._build(0)
        full_expr = sp.Implies(Outer(Y), expr)
        compiled = compile_logic(full_expr, predicates, mode="semantic")
        result = compiled(**bindings)
        expected = (1 - outer_p) + outer_p * csv
        assert torch.allclose(
            result, torch.tensor([expected], dtype=torch.float64), atol=1e-9
        )

    def test_equivalent(self):
        expr, Outer, Y, predicates, bindings, csv, outer_p = self._build(1)
        full_expr = sp.Equivalent(expr, Outer(Y))
        compiled = compile_logic(full_expr, predicates, mode="semantic")
        result = compiled(**bindings)
        expected = csv * outer_p + (1 - csv) * (1 - outer_p)
        assert torch.allclose(
            result, torch.tensor([expected], dtype=torch.float64), atol=1e-9
        )

    def test_not_of_case_split(self):
        expr, _, _, predicates, bindings, csv, _ = self._build(0)
        full_expr = sp.Not(expr)
        del bindings["Y"]
        compiled = compile_logic(full_expr, predicates, mode="semantic")
        result = compiled(**bindings)
        assert torch.allclose(
            result, torch.tensor([1 - csv], dtype=torch.float64), atol=1e-9
        )


class TestMultipleIndependentCaseSplits:
    def test_two_independent_case_splits_combined_via_and(self):
        expr1, info1 = _case_split_formula(
            [0.2, 0.8], 0.5, cat_name="Cat1", var_suffix="A"
        )
        expr2, info2 = _case_split_formula(
            [0.9, 0.1], 0.3, cat_name="Cat2", var_suffix="B"
        )
        full_expr = sp.And(expr1, expr2)

        labels_a = torch.tensor([1.0], dtype=torch.float64)
        labels_b = torch.tensor([0.0], dtype=torch.float64)
        predicates = {
            info1["cat_name"]: _hard_label_pred(labels_a),
            info1["digit_name"]: info1["digit_pred"],
            info1["common_name"]: info1["common_pred"],
            info2["cat_name"]: _hard_label_pred(labels_b),
            info2["digit_name"]: info2["digit_pred"],
            info2["common_name"]: info2["common_pred"],
        }
        bindings = {
            info1["x_name"]: torch.zeros(1, 1, dtype=torch.float64),
            info1["labels_name"]: labels_a,
            info2["x_name"]: torch.zeros(1, 1, dtype=torch.float64),
            info2["labels_name"]: labels_b,
        }
        compiled = compile_logic(full_expr, predicates, mode="semantic")
        result = compiled(**bindings)

        expected = (0.8 * 0.5) * (0.9 * 0.3)
        assert torch.allclose(
            result, torch.tensor([expected], dtype=torch.float64), atol=1e-9
        )


class TestDisjointnessViolationFallback:
    """When a candidate's atoms recur outside its own subtree, it must
    NOT be decomposed -- ordinary (always-correct) compilation must
    take over for the whole formula, verified against the true joint
    WMC (standard independent-atom semantics), not the decomposition's
    (would-be-wrong) value."""

    def test_shared_atom_with_outer_formula_falls_back_correctly(self):
        X, Labels = Variable("X Labels")
        Digit, Common, Cat = Symbol("Digit Common Cat")
        S = Variable("S")
        domain = [0, 1]
        case_split = ForAll(
            S, domain, sp.Implies(Cat(Labels, S), sp.And(Digit(X, S), Common(X)))
        )
        # Common(X) is reused OUTSIDE the case-split subtree -- disjointness
        # violated, must fall back to ordinary compilation.
        full_expr = sp.And(case_split, Common(X))

        digit_probs = [0.3, 0.6]
        common_p = 0.5
        cat_p = 0.5  # soft, independent per-branch Cond values

        def digit_pred_indexed(x, s):
            return torch.full((x.shape[0],), digit_probs[s], dtype=torch.float64)

        def cat_pred(labels, s):
            del labels
            return torch.full((1,), cat_p, dtype=torch.float64)

        predicates = {
            "Cat": cat_pred,
            "Digit": digit_pred_indexed,
            "Common": _const_pred(common_p),
        }
        labels = torch.zeros(1, dtype=torch.float64)
        x = torch.zeros(1, 1, dtype=torch.float64)
        compiled = compile_logic(full_expr, predicates, mode="semantic")
        result = compiled(X=x, Labels=labels)

        # True joint WMC: standard independent-atom semantics over
        # Cat0, Cat1, Digit0, Digit1, Common (5 atoms), no case-split
        # shortcut applied (Common is shared, so disjointness fails and
        # the raw And-of-Implies-conjoined-with-Common(X) is compiled
        # as an ordinary formula).
        def bool_fn(cat0, cat1, d0, d1, common):
            case_split_holds = (not cat0 or (d0 and common)) and (
                not cat1 or (d1 and common)
            )
            return case_split_holds and common

        expected = _brute_force_wmc(
            bool_fn,
            {
                "cat0": cat_p,
                "cat1": cat_p,
                "d0": digit_probs[0],
                "d1": digit_probs[1],
                "common": common_p,
            },
        )
        assert torch.allclose(
            result, torch.tensor([expected], dtype=torch.float64), atol=1e-6
        )
        # Sanity: this must NOT equal the (wrong, if applied) decomposition.
        wrong_decomposition = (
            cat_p * digit_probs[0] * common_p
            + cat_p * digit_probs[1] * common_p
        ) * common_p
        assert abs(expected - wrong_decomposition) > 1e-6


class TestHardEvidenceViolation:
    """A candidate that structurally qualifies (shape + disjointness)
    but whose Cond values turn out NOT to be hard 0/1 at evaluation
    time must raise, not silently compute a wrong number."""

    def test_soft_cond_values_raise(self):
        X, Labels = Variable("X Labels")
        Digit, Cat = Symbol("Digit Cat")
        S = Variable("S")
        domain = [0, 1]
        expr = ForAll(S, domain, sp.Implies(Cat(Labels, S), Digit(X, S)))

        def soft_cat_pred(labels, s):
            del labels
            # Sums to 1 but is NOT hard -- exactly the case the design
            # doc's correction found unsound to decompose.
            return torch.full((1,), 0.5, dtype=torch.float64)

        def digit_pred_indexed(x, s):
            return torch.full((x.shape[0],), 0.5, dtype=torch.float64)

        compiled = compile_logic(
            expr,
            {"Cat": soft_cat_pred, "Digit": digit_pred_indexed},
            mode="semantic",
        )
        labels = torch.zeros(1, dtype=torch.float64)
        x = torch.zeros(1, 1, dtype=torch.float64)
        with pytest.raises(ValueError, match="hard evidence"):
            compiled(X=x, Labels=labels)

    def test_cond_not_summing_to_one_raises(self):
        X, Labels = Variable("X Labels")
        Digit, Cat = Symbol("Digit Cat")
        S = Variable("S")
        domain = [0, 1]
        expr = ForAll(S, domain, sp.Implies(Cat(Labels, S), Digit(X, S)))

        def bad_cat_pred(labels, s):
            del labels
            # Both hard 1 -- violates "exactly one true".
            return torch.ones((1,), dtype=torch.float64)

        def digit_pred_indexed(x, s):
            return torch.full((x.shape[0],), 0.5, dtype=torch.float64)

        compiled = compile_logic(
            expr,
            {"Cat": bad_cat_pred, "Digit": digit_pred_indexed},
            mode="semantic",
        )
        labels = torch.zeros(1, dtype=torch.float64)
        x = torch.zeros(1, 1, dtype=torch.float64)
        with pytest.raises(ValueError, match="hard evidence"):
            compiled(X=x, Labels=labels)


class TestGradientFlow:
    def test_gradients_flow_through_body_predicates(self):
        X, Labels = Variable("X Labels")
        Digit, Cat = Symbol("Digit Cat")
        S = Variable("S")
        domain = [0, 1]
        expr = ForAll(S, domain, sp.Implies(Cat(Labels, S), Digit(X, S)))

        model0 = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())
        model1 = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())

        def digit_pred(x, s):
            model = model0 if s == 0 else model1
            return model(x).squeeze(-1)

        true_labels = torch.tensor([0.0, 1.0], dtype=torch.float32)
        logic_loss = logic_to_loss(
            expr,
            {"Cat": _hard_label_pred(true_labels), "Digit": digit_pred},
            mode="semantic",
        )
        x = torch.randn(2, 3)
        loss = logic_loss.loss(X=x, Labels=true_labels)
        loss.backward()

        for model in (model0, model1):
            for param in model.parameters():
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


class TestMnistAdditionTractability:
    """The real motivating constraint: 19 possible sums, hard-evidence
    Sum label, must compile and evaluate fast (contrasted with the
    840K-node joint circuit from the earlier tractability spikes)."""

    def test_mnist_addition_shape_compiles_and_evaluates_quickly(self):
        import time

        from pysignet.logic import Exists

        X1, X2, Labels = Variable("X1 X2 Labels")
        Digit1, Digit2, Sum = Symbol("Digit1 Digit2 Sum")
        S, I = Variable("S I")  # noqa: E741
        sum_domain = list(range(19))
        digit_domain = list(range(10))

        # The real MNIST Addition constraint's own shape: Body(S) is a
        # single template with S embedded arithmetically (S - I), not
        # structurally different terms per branch.
        expr = ForAll(
            S,
            sum_domain,
            sp.Implies(
                Sum(Labels, S),
                Exists(
                    I,
                    digit_domain,
                    sp.And(Digit1(X1, I), Digit2(X2, S - I)),
                ),
            ),
        )

        digit1_probs = [0.05] * 10
        digit2_probs = [0.05] * 10
        digit1_probs[3] = 0.9
        digit2_probs[4] = 0.9

        def digit1_pred(x, i):
            return torch.full(
                (x.shape[0],), digit1_probs[i], dtype=torch.float64
            )

        def digit2_pred(x, j):
            # Out-of-range indices (S - I outside [0, 9]) are always
            # false, matching the established pattern for this
            # constraint elsewhere in the project.
            if not 0 <= j <= 9:
                return torch.zeros((x.shape[0],), dtype=torch.float64)
            return torch.full(
                (x.shape[0],), digit2_probs[j], dtype=torch.float64
            )

        true_sum = 7
        labels = torch.tensor([float(true_sum)], dtype=torch.float64)
        predicates = {
            "Sum": _hard_label_pred(labels),
            "Digit1": digit1_pred,
            "Digit2": digit2_pred,
        }
        x1 = torch.zeros(1, 1, dtype=torch.float64)
        x2 = torch.zeros(1, 1, dtype=torch.float64)

        start = time.perf_counter()
        compiled = compile_logic(expr, predicates, mode="semantic")
        result = compiled(X1=x1, X2=x2, Labels=labels)
        elapsed = time.perf_counter() - start

        pairs = [
            (i, true_sum - i)
            for i in range(10)
            if 0 <= true_sum - i <= 9
        ]
        expected = _brute_force_wmc(
            lambda **kw: any(
                kw[f"d1_{i}"] and kw[f"d2_{j}"] for i, j in pairs
            ),
            {
                **{f"d1_{i}": digit1_probs[i] for i, _ in pairs},
                **{f"d2_{j}": digit2_probs[j] for _, j in pairs},
            },
        )
        assert torch.allclose(
            result, torch.tensor([expected], dtype=torch.float64), atol=1e-6
        )
        assert elapsed < 5.0, (
            f"MNIST Addition case-split compile+eval took {elapsed:.2f}s, "
            f"expected well under 5s (vs. hundreds of ms/call for the "
            f"840K-node joint circuit from the earlier spike)"
        )


class TestShapeMatchingEdgeCases:
    """Structural rejections that must fall through to ordinary
    (still-correct) compilation, not error -- these exercise
    find_case_split_candidates' shape-matching rules directly."""

    def test_multi_variable_forall_is_not_a_candidate(self):
        X, Labels = Variable("X Labels")
        Digit, Cat = Symbol("Digit Cat")
        S, T = Variable("S T")
        # A multi-variable ForAll (v1 scope: single-variable only) that
        # otherwise looks case-split-shaped must not be treated as one.
        expr = ForAll(
            [S, T], [(0, 0), (1, 1)], sp.Implies(Cat(Labels, S), Digit(X, S))
        )
        p = 0.4

        def digit_pred(x, s):
            return torch.full((x.shape[0],), p, dtype=torch.float64)

        def cat_pred(labels, s):
            del labels
            return torch.ones((1,), dtype=torch.float64)

        compiled = compile_logic(
            expr, {"Cat": cat_pred, "Digit": digit_pred}, mode="semantic"
        )
        labels = torch.zeros(1, dtype=torch.float64)
        x = torch.zeros(1, 1, dtype=torch.float64)
        # Should compile via the ordinary path without error (even
        # though Cat is hard-evidence-shaped, it's never checked since
        # this was never treated as a case-split candidate).
        result = compiled(X=x, Labels=labels)
        assert result.shape == (1,)

    def test_non_predicate_antecedent_is_not_a_candidate(self):
        X = Variable("X")
        Digit = Symbol("Digit")
        S = Variable("S")
        # Antecedent is Eq(S, 0), not a PredicateApplication -- not a
        # case-split candidate; ForAll still expands and compiles
        # normally. (sp.Implies(sp.true, ...) would simplify away at
        # construction time before ever reaching this check, so a
        # genuinely non-constant, non-PredicateApplication antecedent
        # is needed to actually exercise this rejection branch.)
        expr = ForAll(S, [0, 1], sp.Implies(sp.Eq(S, 0), Digit(X, S)))
        p = 0.3

        def digit_pred(x, s):
            return torch.full((x.shape[0],), p, dtype=torch.float64)

        compiled = compile_logic(
            expr, {"Digit": digit_pred}, mode="semantic"
        )
        x = torch.zeros(1, 1, dtype=torch.float64)
        result = compiled(X=x)
        # Implies(Eq(0,0)=True, Digit(0)) AND Implies(Eq(1,0)=False, Digit(1))
        # = Digit(0) AND True = Digit(0) = p
        assert torch.allclose(
            result, torch.tensor([p], dtype=torch.float64), atol=1e-9
        )

    def test_case_split_nested_inside_ordinary_forall(self):
        """A case-split candidate nested inside an outer ForAll that is
        NOT itself case-split-shaped -- exercises the "recurse into a
        non-matching Quantifier's body" path in both detection and
        substitution."""
        X, Labels, Y = Variable("X Labels Y")
        Digit, Cat, Other = Symbol("Digit Cat Other")
        S, T = Variable("S T")

        case_split = ForAll(
            S, [0, 1], sp.Implies(Cat(Labels, S), Digit(X, S))
        )
        # Outer ForAll over T is NOT case-split-shaped (its own body is
        # just an ordinary conjunct, not Implies(Cond(T), Body(T))),
        # but CONTAINS the real case split in its body.
        expr = ForAll(T, [0, 1], sp.And(case_split, Other(Y, T)))

        digit_probs = [0.25, 0.75]
        other_p = 0.6

        def digit_pred(x, s):
            return torch.full(
                (x.shape[0],), digit_probs[s], dtype=torch.float64
            )

        def other_pred(y, t):
            del t
            return torch.full((y.shape[0],), other_p, dtype=torch.float64)

        true_label = 1
        labels = torch.tensor([float(true_label)], dtype=torch.float64)
        x = torch.zeros(1, 1, dtype=torch.float64)
        y = torch.zeros(1, 1, dtype=torch.float64)

        compiled = compile_logic(
            expr,
            {"Cat": _hard_label_pred(labels), "Digit": digit_pred, "Other": other_pred},
            mode="semantic",
        )
        result = compiled(X=x, Labels=labels, Y=y)

        # ForAll(T, [0,1], And(case_split, Other(Y,T))) expands to
        # And(case_split & Other(Y,0), case_split & Other(Y,1))
        # = case_split_value * other_p * other_p (case_split reused
        # across both T branches -- disjoint from Other, still eligible)
        case_split_value = digit_probs[true_label]
        expected = case_split_value * other_p * other_p
        assert torch.allclose(
            result, torch.tensor([expected], dtype=torch.float64), atol=1e-9
        )


class TestBranchSizeGuard:
    def test_branch_exceeding_max_atoms_raises(self):
        X, Labels = Variable("X Labels")
        Digit, Cat = Symbol("Digit Cat")
        S = Variable("S")
        I = Variable("I")  # noqa: E741
        from pysignet.logic import ForAll as _ForAll

        # Body(S) itself expands to a large ForAll (25 atoms), forcing
        # that single branch's own compilation to exceed max_atoms=20,
        # independent of the outer/other-branches' sizes.
        expr = ForAll(
            S,
            [0, 1],
            sp.Implies(
                Cat(Labels, S), _ForAll(I, list(range(25)), Digit(X, I))
            ),
        )

        def digit_pred(x, i):
            return torch.full((x.shape[0],), 0.5, dtype=torch.float64)

        def cat_pred(labels, s):
            del labels
            return torch.tensor(
                [1.0 if s == 0 else 0.0], dtype=torch.float64
            )

        with pytest.raises(ValueError, match="max_atoms"):
            compile_logic(
                expr, {"Cat": cat_pred, "Digit": digit_pred}, mode="semantic"
            )
