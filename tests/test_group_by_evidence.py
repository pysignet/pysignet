"""Tests for group_by_evidence= (TODO.md 2.22).

Compiler-agnostic hard-evidence branch grouping: for
ForAll(S, domain, Implies(Cond(S), Body(S))) where Cond is observed
per-example evidence (not a soft prediction), groups batch elements by
which branch is actually live in THIS batch and evaluates each group
only against its own compiled Body(s) -- unlike SemanticLossCompiler's
case-split (which evaluates every branch, weighted by an indicator, to
avoid one huge SDD), this is a genuine compute reduction: branches with
no examples in the current batch are never evaluated at all.

Lives at the compile_logic()/logic_to_loss() level (the "logic to
loss" entry points), not inside the compiler classes -- see TODO.md
2.22 for why. v1 scope: the case-split candidate must be the entire
compiled expression (no nested support yet).
"""

import pytest
import sympy as sp
import torch
import torch.nn as nn

from pysignet import Symbol, Variable, compile_logic, logic_to_loss
from pysignet.logic import ForAll


def _hard_label_pred(true_labels: torch.Tensor):
    def pred(labels: torch.Tensor, s: int) -> torch.Tensor:
        return (labels == s).to(torch.float64)

    return pred


def _counting_pred(values: list[float]):
    """A predicate that records which s values it was actually called
    with, to verify dead branches are never evaluated."""
    calls: list[int] = []

    def pred(x: torch.Tensor, s: int) -> torch.Tensor:
        calls.append(s)
        return torch.full((x.shape[0],), values[s], dtype=torch.float64)

    pred.calls = calls  # type: ignore[attr-defined]
    return pred


class TestBasicCorrectness:
    def test_tnorm_matches_body_of_true_label(self):
        X, Labels = Variable("X Labels")
        Digit, Cat = Symbol("Digit Cat")
        S = Variable("S")
        domain = [0, 1, 2]
        expr = ForAll(S, domain, sp.Implies(Cat(Labels, S), Digit(X, S)))

        digit_probs = [0.2, 0.6, 0.9]

        def digit_pred(x, s):
            return torch.full(
                (x.shape[0],), digit_probs[s], dtype=torch.float64
            )

        true_labels = torch.tensor([0.0, 1.0, 2.0, 1.0])
        x = torch.zeros(4, 1, dtype=torch.float64)

        compiled = compile_logic(
            expr,
            {"Cat": _hard_label_pred(true_labels), "Digit": digit_pred},
            mode="tnorm",
            group_by_evidence=True,
        )
        result = compiled(X=x, Labels=true_labels)

        expected = torch.tensor(
            [digit_probs[s] for s in [0, 1, 2, 1]], dtype=torch.float64
        )
        assert torch.allclose(result, expected, atol=1e-9)

    def test_ltu_mode_also_supported(self):
        X, Labels = Variable("X Labels")
        Digit, Cat = Symbol("Digit Cat")
        S = Variable("S")
        domain = [0, 1]
        expr = ForAll(S, domain, sp.Implies(Cat(Labels, S), Digit(X, S)))

        digit_probs = [0.3, 0.8]

        def digit_pred(x, s):
            return torch.full(
                (x.shape[0],), digit_probs[s], dtype=torch.float64
            )

        true_labels = torch.tensor([1.0])
        x = torch.zeros(1, 1, dtype=torch.float64)

        compiled = compile_logic(
            expr,
            {"Cat": _hard_label_pred(true_labels), "Digit": digit_pred},
            mode="ltu",
            group_by_evidence=True,
        )
        result = compiled(X=x, Labels=true_labels)
        # LTU with default alpha applies a sigmoid threshold, not a raw
        # passthrough -- just confirm it runs and matches compiling
        # Body(1) directly via the same (LTU) mode, not the tnorm value.
        direct = compile_logic(
            Digit(X, 1), {"Digit": digit_pred}, mode="ltu"
        )
        assert torch.allclose(result, direct(X=x), atol=1e-9)


class TestDeadBranchesNeverEvaluated:
    def test_branches_absent_from_batch_are_skipped(self):
        X, Labels = Variable("X Labels")
        Digit, Cat = Symbol("Digit Cat")
        S = Variable("S")
        domain = [0, 1, 2, 3, 4]
        expr = ForAll(S, domain, sp.Implies(Cat(Labels, S), Digit(X, S)))

        digit_pred = _counting_pred([0.1, 0.2, 0.3, 0.4, 0.5])
        true_labels = torch.tensor([1.0, 1.0, 3.0])  # only s=1,3 present
        x = torch.zeros(3, 1, dtype=torch.float64)

        compiled = compile_logic(
            expr,
            {"Cat": _hard_label_pred(true_labels), "Digit": digit_pred},
            mode="tnorm",
            group_by_evidence=True,
        )
        compiled(X=x, Labels=true_labels)

        assert set(digit_pred.calls) == {1, 3}  # type: ignore[attr-defined]


class TestHardEvidenceViolation:
    def test_soft_cond_raises(self):
        X, Labels = Variable("X Labels")
        Digit, Cat = Symbol("Digit Cat")
        S = Variable("S")
        domain = [0, 1]
        expr = ForAll(S, domain, sp.Implies(Cat(Labels, S), Digit(X, S)))

        def soft_cat(labels, s):
            del labels
            return torch.full((1,), 0.5, dtype=torch.float64)

        def digit_pred(x, s):
            return torch.full((x.shape[0],), 0.5, dtype=torch.float64)

        compiled = compile_logic(
            expr,
            {"Cat": soft_cat, "Digit": digit_pred},
            mode="tnorm",
            group_by_evidence=True,
        )
        labels = torch.zeros(1, dtype=torch.float64)
        x = torch.zeros(1, 1, dtype=torch.float64)
        with pytest.raises(ValueError, match="hard evidence"):
            compiled(X=x, Labels=labels)


class TestScopeBoundaries:
    def test_non_case_split_shape_raises_clear_error(self):
        X = Variable("X")
        A = Symbol("A")
        expr = A(X)
        with pytest.raises(ValueError, match="group_by_evidence"):
            compile_logic(
                expr, {"A": lambda x: x.squeeze(-1)},
                mode="tnorm", group_by_evidence=True,
            )

    def test_nested_case_split_not_yet_supported(self):
        X, Labels, Y = Variable("X Labels Y")
        Digit, Cat, Other = Symbol("Digit Cat Other")
        S = Variable("S")
        case_split_expr = ForAll(
            S, [0, 1], sp.Implies(Cat(Labels, S), Digit(X, S))
        )
        full_expr = sp.And(case_split_expr, Other(Y))
        with pytest.raises(ValueError, match="group_by_evidence"):
            compile_logic(
                full_expr,
                {
                    "Cat": lambda labels, s: torch.zeros(1, dtype=torch.float64),
                    "Digit": lambda x, s: torch.zeros(x.shape[0], dtype=torch.float64),
                    "Other": lambda y: torch.zeros(y.shape[0], dtype=torch.float64),
                },
                mode="tnorm",
                group_by_evidence=True,
            )


class TestGradientFlow:
    def test_gradients_flow_only_through_exercised_branches(self):
        X, Labels = Variable("X Labels")
        Digit, Cat = Symbol("Digit Cat")
        S = Variable("S")
        domain = [0, 1, 2]
        expr = ForAll(S, domain, sp.Implies(Cat(Labels, S), Digit(X, S)))

        models = [nn.Sequential(nn.Linear(3, 1), nn.Sigmoid()) for _ in range(3)]

        def digit_pred(x, s):
            return models[s](x).squeeze(-1)

        true_labels = torch.tensor([0.0, 2.0])  # s=1 never appears
        logic_loss = logic_to_loss(
            expr,
            {"Cat": _hard_label_pred(true_labels), "Digit": digit_pred},
            mode="tnorm",
            group_by_evidence=True,
        )
        x = torch.randn(2, 3)
        loss = logic_loss.loss(X=x, Labels=true_labels)
        loss.backward()

        for i, model in enumerate(models):
            grads_exist = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters()
            )
            if i == 1:
                assert not grads_exist, "branch 1 was never in the batch"
            else:
                assert grads_exist, f"branch {i} should have received grads"


class TestEmptyBatch:
    def test_empty_batch_returns_empty_tensor_without_crashing(self):
        X, Labels = Variable("X Labels")
        Digit, Cat = Symbol("Digit Cat")
        S = Variable("S")
        domain = [0, 1]
        expr = ForAll(S, domain, sp.Implies(Cat(Labels, S), Digit(X, S)))

        def digit_pred(x, s):
            return torch.full((x.shape[0],), 0.5, dtype=torch.float64)

        def cat_pred(labels, s):
            return (labels == s).to(torch.float64)

        compiled = compile_logic(
            expr,
            {"Cat": cat_pred, "Digit": digit_pred},
            mode="tnorm",
            group_by_evidence=True,
        )
        labels = torch.zeros(0, dtype=torch.float64)
        x = torch.zeros(0, 1, dtype=torch.float64)
        result = compiled(X=x, Labels=labels)
        assert result.shape == (0,)
