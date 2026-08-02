"""Tests for SemanticLossCompiler (mode='semantic').

Semantic loss is defined directly over the formula's models (Xu et al.,
ICML 2018), so its defining correctness property is that satisfaction
depends only on a formula's *meaning*, not its syntax. Value-correctness
tests below therefore compare against brute-force enumeration of
Definition 1, not just against the compiled circuit, and semantic
equivalence is tested explicitly. See SEMANTIC_LOSS_DESIGN.md.
"""

import itertools
import sys
from collections.abc import Callable
from unittest import mock

import pytest
import sympy as sp
import torch
import torch.nn as nn

from pysignet import Symbol, Variable, compile_logic, logic_to_loss
from pysignet.compilation import SemanticLossCompiler


def _const_pred(value: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """A predicate that ignores its input and returns a fixed probability."""

    def pred(x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), value, dtype=torch.float64)

    return pred


def _brute_force_wmc(
    bool_fn: Callable[..., bool], probs: dict[str, float]
) -> float:
    """Definition 1: enumerate every assignment and sum satisfying weights."""
    names = list(probs)
    total = 0.0
    for bits in itertools.product([False, True], repeat=len(names)):
        assignment = dict(zip(names, bits, strict=True))
        if bool_fn(**assignment):
            weight = 1.0
            for name in names:
                weight *= probs[name] if assignment[name] else (
                    1 - probs[name]
                )
            total += weight
    return total


class TestValueCorrectness:
    """Semantic loss values vs. brute-force Definition 1 enumeration."""

    def test_and_value(self):
        X = Variable("X")
        A, B = Symbol("A B")
        expr = sp.And(A(X), B(X))
        p_a, p_b = 0.3, 0.7

        compiled = compile_logic(
            expr,
            {"A": _const_pred(p_a), "B": _const_pred(p_b)},
            mode="semantic",
        )
        x = torch.zeros(1, 1, dtype=torch.float64)
        result = compiled(X=x)

        expected = _brute_force_wmc(
            lambda A, B: A and B, {"A": p_a, "B": p_b}
        )
        assert torch.allclose(
            result, torch.tensor([expected], dtype=torch.float64), atol=1e-9
        )

    def test_or_value(self):
        X = Variable("X")
        A, B = Symbol("A B")
        expr = sp.Or(A(X), B(X))
        p_a, p_b = 0.2, 0.6

        compiled = compile_logic(
            expr,
            {"A": _const_pred(p_a), "B": _const_pred(p_b)},
            mode="semantic",
        )
        x = torch.zeros(1, 1, dtype=torch.float64)
        result = compiled(X=x)

        expected = _brute_force_wmc(
            lambda A, B: A or B, {"A": p_a, "B": p_b}
        )
        assert torch.allclose(
            result, torch.tensor([expected], dtype=torch.float64), atol=1e-9
        )

    def test_not_value(self):
        X = Variable("X")
        A = Symbol("A")
        expr = sp.Not(A(X))
        p_a = 0.35

        compiled = compile_logic(
            expr, {"A": _const_pred(p_a)}, mode="semantic"
        )
        x = torch.zeros(1, 1, dtype=torch.float64)
        result = compiled(X=x)

        expected = _brute_force_wmc(lambda A: not A, {"A": p_a})
        assert torch.allclose(
            result, torch.tensor([expected], dtype=torch.float64), atol=1e-9
        )

    def test_implies_value(self):
        X = Variable("X")
        A, B = Symbol("A B")
        expr = sp.Implies(A(X), B(X))
        p_a, p_b = 0.4, 0.6

        compiled = compile_logic(
            expr,
            {"A": _const_pred(p_a), "B": _const_pred(p_b)},
            mode="semantic",
        )
        x = torch.zeros(1, 1, dtype=torch.float64)
        result = compiled(X=x)

        expected = _brute_force_wmc(
            lambda A, B: (not A) or B, {"A": p_a, "B": p_b}
        )
        assert torch.allclose(
            result, torch.tensor([expected], dtype=torch.float64), atol=1e-9
        )

    def test_equivalent_value(self):
        X = Variable("X")
        A, B = Symbol("A B")
        expr = sp.Equivalent(A(X), B(X))
        p_a, p_b = 0.45, 0.55

        compiled = compile_logic(
            expr,
            {"A": _const_pred(p_a), "B": _const_pred(p_b)},
            mode="semantic",
        )
        x = torch.zeros(1, 1, dtype=torch.float64)
        result = compiled(X=x)

        expected = _brute_force_wmc(
            lambda A, B: A == B, {"A": p_a, "B": p_b}
        )
        assert torch.allclose(
            result, torch.tensor([expected], dtype=torch.float64), atol=1e-9
        )


class TestSemanticEquivalence:
    """Proposition 2: logically equivalent formulas get identical loss.

    This is the core property t-norms fail (a t-norm's value depends on
    how a formula happens to be written) and semantic loss is supposed
    to guarantee.
    """

    def test_de_morgan_equivalence(self):
        X = Variable("X")
        A, B = Symbol("A B")
        expr1 = sp.And(A(X), B(X))
        expr2 = sp.Not(sp.Or(sp.Not(A(X)), sp.Not(B(X))))
        preds = {"A": _const_pred(0.3), "B": _const_pred(0.8)}

        c1 = compile_logic(expr1, preds, mode="semantic")
        c2 = compile_logic(expr2, preds, mode="semantic")
        x = torch.zeros(1, 1, dtype=torch.float64)

        assert torch.allclose(c1(X=x), c2(X=x), atol=1e-9)

    def test_absorption_law_equivalence_repeated_atom(self):
        """And(A, Or(A, B)) == A (absorption); also exercises a ground
        atom appearing at two different tree positions, which must
        collapse to a single SDD variable (Section 3.2)."""
        X = Variable("X")
        A, B = Symbol("A B")
        expr1 = sp.And(A(X), sp.Or(A(X), B(X)))
        expr2 = A(X)
        preds = {"A": _const_pred(0.42), "B": _const_pred(0.9)}

        c1 = compile_logic(expr1, preds, mode="semantic")
        c2 = compile_logic(expr2, preds, mode="semantic")
        x = torch.zeros(1, 1, dtype=torch.float64)

        assert torch.allclose(c1(X=x), c2(X=x), atol=1e-9)
        assert torch.allclose(
            c1(X=x), torch.tensor([0.42], dtype=torch.float64), atol=1e-9
        )


class TestGradients:
    """Gradient flow through the differentiable WMC walk."""

    def test_gradients_flow_to_model(self):
        X = Variable("X")
        A, B = Symbol("A B")
        expr = sp.Implies(A(X), B(X))

        model_a = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
        model_b = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())

        def pred_a(x):
            return model_a(x).squeeze(-1)

        def pred_b(x):
            return model_b(x).squeeze(-1)

        logic_loss = logic_to_loss(
            expr, {"A": pred_a, "B": pred_b}, mode="semantic"
        )
        x = torch.randn(8, 4)
        loss = logic_loss.loss(X=x)
        loss.backward()

        for model in (model_a, model_b):
            for param in model.parameters():
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

    def test_gradcheck_small_formula(self):
        X1, X2 = Variable("X1 X2")
        A, B = Symbol("A B")
        expr = sp.And(A(X1), B(X2))

        compiled = compile_logic(
            expr,
            {
                "A": lambda x: x.squeeze(-1),
                "B": lambda x: x.squeeze(-1),
            },
            mode="semantic",
        )

        def fn(a_val, b_val):
            return compiled(X1=a_val, X2=b_val)

        a_val = torch.tensor(
            [[0.3]], dtype=torch.float64, requires_grad=True
        )
        b_val = torch.tensor(
            [[0.7]], dtype=torch.float64, requires_grad=True
        )
        assert torch.autograd.gradcheck(fn, (a_val, b_val))


class TestBatchReduction:
    """conjunction()/disjunction() are used ONLY for i.i.d. batch
    reduction (forall/exists), never to evaluate the formula itself --
    see SemanticLossCompiler's class docstring."""

    def test_forall_is_product_of_batch_elements(self):
        X = Variable("X")
        A = Symbol("A")
        expr = A(X)

        logic_loss = logic_to_loss(
            expr, {"A": _const_pred(0.5)}, mode="semantic"
        )
        x = torch.zeros(3, 1, dtype=torch.float64)
        result = logic_loss.satisfaction(X=x, quantify="forall")
        assert torch.allclose(
            result, torch.tensor(0.125, dtype=torch.float64), atol=1e-9
        )

    def test_exists_is_probabilistic_sum_of_batch_elements(self):
        X = Variable("X")
        A = Symbol("A")
        expr = A(X)

        logic_loss = logic_to_loss(
            expr, {"A": _const_pred(0.5)}, mode="semantic"
        )
        x = torch.zeros(3, 1, dtype=torch.float64)
        result = logic_loss.satisfaction(X=x, quantify="exists")
        # 1 - (1-0.5)^3 = 0.875
        assert torch.allclose(
            result, torch.tensor(0.875, dtype=torch.float64), atol=1e-9
        )


class TestModeWiring:
    """mode='semantic' wiring in compile_logic/logic_to_loss."""

    def test_compile_logic_mode_semantic_uses_compiler(self):
        X = Variable("X")
        A = Symbol("A")
        expr = A(X)

        compiled = compile_logic(
            expr, {"A": _const_pred(0.5)}, mode="semantic"
        )
        assert isinstance(compiled.compiler, SemanticLossCompiler)

    def test_logic_to_loss_default_postprocessing_is_log(self):
        X = Variable("X")
        A = Symbol("A")
        expr = A(X)

        logic_loss = logic_to_loss(
            expr, {"A": _const_pred(0.5)}, mode="semantic"
        )
        assert logic_loss.default_post_processing == "log"

    def test_tnorm_with_mode_semantic_raises(self):
        from pysignet.tnorms import LukasiewiczTNorm

        X = Variable("X")
        A = Symbol("A")
        expr = A(X)

        with pytest.raises(ValueError, match="tnorm= is only valid"):
            compile_logic(
                expr,
                {"A": _const_pred(0.5)},
                mode="semantic",
                tnorm=LukasiewiczTNorm(),
            )

    def test_max_atoms_passthrough(self):
        X = Variable("X")
        A, B = Symbol("A B")
        expr = sp.And(A(X), B(X))

        with pytest.raises(ValueError, match="max_atoms"):
            compile_logic(
                expr,
                {"A": _const_pred(0.5), "B": _const_pred(0.5)},
                mode="semantic",
                max_atoms=1,
            )


class TestEdgeCases:
    """Boolean constants, single atom, and repeated atoms."""

    def test_true_constant(self):
        # A bare sp.true/sp.false (not embedded in And/Or -- SymPy
        # simplifies And(a, True) to just `a` at construction time, so
        # embedding would never actually reach the constant-node branch
        # of _expr_to_sdd) is the only way to exercise it directly,
        # matching the existing TNormCompiler pattern in
        # tests/test_constants.py.
        X = Variable("X")
        A = Symbol("A")
        expr = sp.true

        compiled = compile_logic(
            expr, {"A": _const_pred(0.6)}, mode="semantic"
        )
        x = torch.zeros(1, 1, dtype=torch.float64)
        result = compiled(X=x)
        assert torch.allclose(
            result, torch.tensor([1.0], dtype=torch.float64), atol=1e-9
        )

    def test_false_constant(self):
        X = Variable("X")
        A = Symbol("A")
        expr = sp.false

        compiled = compile_logic(
            expr, {"A": _const_pred(0.6)}, mode="semantic"
        )
        x = torch.zeros(1, 1, dtype=torch.float64)
        result = compiled(X=x)
        assert torch.allclose(
            result, torch.tensor([0.0], dtype=torch.float64), atol=1e-9
        )

    def test_constant_with_no_inputs_raises(self):
        """A constant expression with truly no inputs at all (not even
        an unrelated variable to infer batch size from) cannot
        determine a batch size."""
        expr = sp.true

        compiled = compile_logic(expr, {}, mode="semantic")
        with pytest.raises(ValueError, match="Inputs dict cannot be empty"):
            compiled()

    def test_single_atom_formula(self):
        X = Variable("X")
        A = Symbol("A")
        expr = A(X)

        compiled = compile_logic(
            expr, {"A": _const_pred(0.77)}, mode="semantic"
        )
        x = torch.zeros(1, 1, dtype=torch.float64)
        result = compiled(X=x)
        assert torch.allclose(
            result, torch.tensor([0.77], dtype=torch.float64), atol=1e-9
        )


class TestWmcCrossCheck:
    """Promoted from the Step 0 feasibility spike (TODO.md 2.8): the
    hand-written differentiable _wmc() walk must match PySDD's own
    (non-differentiable, scalar) WmcManager.propagate() exactly."""

    def test_exactly_one_matches_wmcmanager(self):
        from pysdd.sdd import SddManager, Vtree

        n = 5
        X = Variable("X")
        preds = {f"A{i}": Symbol(f"A{i}") for i in range(n)}
        probs = [0.1, 0.2, 0.15, 0.05, 0.3]

        atoms = [preds[f"A{i}"](X) for i in range(n)]
        at_least_one = sp.Or(*atoms)
        mutex = sp.And(
            *[
                sp.Not(sp.And(atoms[i], atoms[j]))
                for i in range(n)
                for j in range(i + 1, n)
            ]
        )
        expr = sp.And(at_least_one, mutex)

        predicates = {
            f"A{i}": _const_pred(probs[i]) for i in range(n)
        }
        compiled = compile_logic(expr, predicates, mode="semantic")
        x = torch.zeros(1, 1, dtype=torch.float64)
        result = compiled(X=x)

        # Independent ground-truth reference via PySDD's own WMC, built
        # directly (not via SemanticLossCompiler internals).
        vtree = Vtree(var_count=1)
        mgr = SddManager.from_vtree(vtree)
        for _ in range(n - 1):
            mgr.add_var_after_last()
        lits = [mgr.literal(i + 1) for i in range(n)]
        import functools
        import operator

        at_least_one_sdd = functools.reduce(operator.or_, lits)
        mutex_sdd = functools.reduce(
            operator.and_,
            [
                (~lits[i]) | (~lits[j])
                for i in range(n)
                for j in range(i + 1, n)
            ],
        )
        formula = at_least_one_sdd & mutex_sdd
        wmc = formula.wmc(log_mode=False)
        for i in range(n):
            wmc.set_literal_weight(mgr.literal(i + 1), probs[i])
            wmc.set_literal_weight(mgr.literal(-(i + 1)), 1 - probs[i])
        reference = wmc.propagate()

        assert torch.allclose(
            result, torch.tensor([reference], dtype=torch.float64),
            atol=1e-9,
        )

        expected = _brute_force_wmc(
            lambda **assignment: (
                any(assignment.values())
                and sum(assignment.values()) == 1
            ),
            {f"A{i}": probs[i] for i in range(n)},
        )
        assert abs(reference - expected) < 1e-9


class TestSizeGuard:
    """Compiling too many unique ground atoms raises a clear error."""

    def test_too_many_atoms_raises_value_error(self):
        from pysignet.logic import ForAll

        X, Y = Variable("X Y")
        Digit = Symbol("Digit")
        expr = ForAll(Y, range(25), Digit(X, Y))

        model = nn.Sequential(nn.Linear(4, 25), nn.Softmax(dim=-1))
        with pytest.raises(ValueError, match="max_atoms"):
            compile_logic(expr, {"Digit": model}, mode="semantic")

    def test_custom_max_atoms_allows_larger_formula(self):
        from pysignet.logic import ForAll

        X, Y = Variable("X Y")
        Digit = Symbol("Digit")
        expr = ForAll(Y, range(25), Digit(X, Y))

        model = nn.Sequential(nn.Linear(4, 25), nn.Softmax(dim=-1))
        compiled = compile_logic(
            expr, {"Digit": model}, mode="semantic", max_atoms=25
        )
        x = torch.randn(2, 4, dtype=torch.float32)
        result = compiled(X=x)
        assert result.shape == (2,)


class TestMissingDependency:
    """ImportError with an install hint when PySDD is unavailable."""

    def test_missing_pysdd_raises_helpful_error(self, monkeypatch):
        X = Variable("X")
        A = Symbol("A")
        expr = A(X)

        monkeypatch.setitem(sys.modules, "pysdd", None)
        monkeypatch.setitem(sys.modules, "pysdd.sdd", None)

        with pytest.raises(ImportError, match="pip install pysignet"):
            compile_logic(
                expr, {"A": _const_pred(0.5)}, mode="semantic"
            )


class TestLinearizedProgram:
    """SEMANTIC_LOSS_DESIGN.md Optimization Considerations (2026-08-01):
    the compiled SDD is linearized into a flat program once at
    compile() time -- mirroring pypsdd's own weighted_model_count()/
    generate_tf_ac(), which the authors' own comment describes as
    "faster than recursive traversal" -- so _wmc() iterates a plain
    list on every call instead of recursing over the SDD's own API
    with a fresh memo dict each time."""

    def test_program_built_once_across_multiple_calls(self):
        X = Variable("X")
        A, B = Symbol("A B")
        expr = sp.And(A(X), B(X))

        from pysignet.compilation import semantic_compiler as sc

        with mock.patch.object(
            sc, "_compile_wmc_program", wraps=sc._compile_wmc_program
        ) as spy:
            compiled = compile_logic(
                expr,
                {"A": _const_pred(0.3), "B": _const_pred(0.7)},
                mode="semantic",
            )
            assert spy.call_count == 1

            x = torch.zeros(1, 1, dtype=torch.float64)
            compiled(X=x)
            compiled(X=x)
            compiled(X=x)
            # Still only compiled once, even after three forward calls.
            assert spy.call_count == 1

    def test_repeated_calls_are_independent_and_correct(self):
        """No stale state should leak between calls now that _wmc() no
        longer takes a per-call memo dict -- each call must be
        evaluated fresh against its own inputs."""
        X = Variable("X")
        A, B = Symbol("A B")
        expr = sp.And(A(X), sp.Or(A(X), B(X)))  # absorption -> A

        compiled = compile_logic(
            expr,
            {"A": _const_pred(0.2), "B": _const_pred(0.9)},
            mode="semantic",
        )
        x = torch.zeros(1, 1, dtype=torch.float64)
        first = compiled(X=x)
        second = compiled(X=x)
        assert torch.allclose(first, second, atol=1e-9)
        assert torch.allclose(
            first, torch.tensor([0.2], dtype=torch.float64), atol=1e-9
        )

    def test_program_structure_for_small_and(self):
        """White-box: lock in the flat program's shape for a tiny known
        SDD (A & B), as a contract for _wmc()'s consumer."""
        from pysdd.sdd import SddManager, Vtree

        from pysignet.compilation.semantic_compiler import (
            _compile_wmc_program,
            _wmc,
        )

        vtree = Vtree(var_count=1)
        mgr = SddManager.from_vtree(vtree)
        mgr.add_var_after_last()
        a = mgr.literal(1)
        b = mgr.literal(2)
        sdd = a & b

        program = _compile_wmc_program(sdd)
        kinds = [op.kind for op in program]
        # Exact node count/shape is an SDD compilation detail (e.g. a
        # decomposition may include a "false" branch to cover the
        # negated-prime case) -- only the position-indexed contract
        # matters here: literals precede the decision(s), and the
        # program is post-order (root last).
        assert kinds.count("literal") >= 2
        assert kinds[-1] == "decision"
        for i, op in enumerate(program):
            if op.kind == "decision":
                for prime_pos, sub_pos in op.children:
                    assert prime_pos < i
                    assert sub_pos < i

        result = _wmc(
            program,
            literal_weights={
                1: torch.tensor([0.4], dtype=torch.float64),
                2: torch.tensor([0.5], dtype=torch.float64),
            },
            batch_shape=torch.Size((1,)),
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        assert torch.allclose(
            result, torch.tensor([0.2], dtype=torch.float64), atol=1e-9
        )
