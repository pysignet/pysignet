"""Tests for the opt-in torch.compile JIT path on LogicCompiler.

Phase 1 (TODO.md 2.21): jit=False by default (opt-in). These tests exercise
the jit=True path explicitly and confirm it produces identical values and
gradients to the default eager path, and that small formulas skip
torch.compile entirely (size threshold).
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from pysignet import (
    Equivalent,
    Implies,
    LogicLoss,
    Not,
    Or,
    Predicate,
    Symbol,
    TNormCompiler,
    Variable,
)
from pysignet.compilation.ltu_compiler import LinearThresholdUnitCompiler


def _and_expr_with_n_atoms(n: int):
    """Build And(P0(X), P1(X), ..., P{n-1}(X)) with n distinct atoms."""
    X = Variable("X")  # pylint: disable=invalid-name
    names = " ".join(f"P{i}" for i in range(n))
    symbols = Symbol(names)
    apps = [symbols[i](X) for i in range(n)]
    expr = apps[0]
    for app in apps[1:]:
        expr = expr & app
    return expr, X, symbols


def _make_constant_predicate(value: float):
    def predicate(x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0]) * value

    return predicate


def _constant_predicates(symbols, values):
    return {
        str(sym): Predicate(_make_constant_predicate(value))
        for sym, value in zip(symbols, values, strict=True)
    }


class TestJitDisabledByDefault:
    """Phase 1: jit=False is the default, and torch.compile is never
    called unless explicitly requested."""

    def test_tnorm_default_never_compiles(self) -> None:
        expr, x_var, symbols = _and_expr_with_n_atoms(12)
        predicates = _constant_predicates(
            symbols, [0.05 * (i + 1) for i in range(12)]
        )

        compiler = TNormCompiler()
        assert compiler.jit is False

        with patch(
            "pysignet.compilation.base.torch.compile", wraps=torch.compile
        ) as mock_compile:
            compiled = compiler.compile(expr, predicates)
            compiled(**{x_var.name: torch.randn(4, 3)})

        mock_compile.assert_not_called()

    def test_ltu_default_never_compiles(self) -> None:
        expr, x_var, symbols = _and_expr_with_n_atoms(12)
        predicates = _constant_predicates(
            symbols, [0.05 * (i + 1) for i in range(12)]
        )

        compiler = LinearThresholdUnitCompiler()
        assert compiler.jit is False

        with patch(
            "pysignet.compilation.base.torch.compile", wraps=torch.compile
        ) as mock_compile:
            compiled = compiler.compile(expr, predicates)
            compiled(**{x_var.name: torch.randn(4, 3)})

        mock_compile.assert_not_called()


class TestJitSizeThreshold:
    """jit=True only triggers torch.compile for formulas at/above the
    configured size threshold; smaller formulas silently use the eager
    path (avoiding tracing overhead for trivial cases)."""

    def test_small_formula_skips_torch_compile(self) -> None:
        n = 3
        assert n < TNormCompiler.JIT_SIZE_THRESHOLD
        expr, x_var, symbols = _and_expr_with_n_atoms(n)
        predicates = _constant_predicates(symbols, [0.3, 0.5, 0.7])

        compiler = TNormCompiler(jit=True)

        with patch(
            "pysignet.compilation.base.torch.compile", wraps=torch.compile
        ) as mock_compile:
            compiled = compiler.compile(expr, predicates)
            compiled(**{x_var.name: torch.randn(4, 3)})

        mock_compile.assert_not_called()

    def test_large_formula_triggers_torch_compile(self) -> None:
        n = 12
        assert n >= TNormCompiler.JIT_SIZE_THRESHOLD
        expr, x_var, symbols = _and_expr_with_n_atoms(n)
        predicates = _constant_predicates(
            symbols, [0.05 * (i + 1) for i in range(n)]
        )

        compiler = TNormCompiler(jit=True)

        with patch(
            "pysignet.compilation.base.torch.compile", wraps=torch.compile
        ) as mock_compile:
            compiled = compiler.compile(expr, predicates)
            compiled(**{x_var.name: torch.randn(4, 3)})

        mock_compile.assert_called_once()


class TestJitParity:
    """jit=True must produce identical values and gradients to jit=False."""

    def test_tnorm_large_formula_values_match(self) -> None:
        n = 12
        expr, x_var, symbols = _and_expr_with_n_atoms(n)
        predicates = _constant_predicates(
            symbols, [0.05 * (i + 1) for i in range(n)]
        )

        eager = TNormCompiler(jit=False).compile(expr, predicates)
        jitted = TNormCompiler(jit=True).compile(expr, predicates)

        x = torch.randn(8, 3)
        eager_out = eager(**{x_var.name: x})
        jit_out = jitted(**{x_var.name: x})

        assert torch.allclose(eager_out, jit_out, atol=1e-6)

    def test_ltu_large_formula_values_match(self) -> None:
        n = 12
        expr, x_var, symbols = _and_expr_with_n_atoms(n)
        predicates = _constant_predicates(
            symbols, [0.05 * (i + 1) for i in range(n)]
        )

        eager = LinearThresholdUnitCompiler(jit=False).compile(
            expr, predicates
        )
        jitted = LinearThresholdUnitCompiler(jit=True).compile(
            expr, predicates
        )

        x = torch.randn(8, 3)
        eager_out = eager(**{x_var.name: x})
        jit_out = jitted(**{x_var.name: x})

        assert torch.allclose(eager_out, jit_out, atol=1e-6)

    def test_gradients_match(self) -> None:
        n = 12
        expr, x_var, symbols = _and_expr_with_n_atoms(n)

        class ConstModel(nn.Module):
            def __init__(self, init: float) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([init]))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sigmoid(self.weight).expand(x.shape[0])

        models_eager = [ConstModel(0.1 * (i + 1)) for i in range(n)]
        models_jit = [ConstModel(0.1 * (i + 1)) for i in range(n)]

        predicates_eager = {
            str(s): Predicate(m) for s, m in zip(symbols, models_eager, strict=True)
        }
        predicates_jit = {
            str(s): Predicate(m) for s, m in zip(symbols, models_jit, strict=True)
        }

        eager = TNormCompiler(jit=False).compile(expr, predicates_eager)
        jitted = TNormCompiler(jit=True).compile(expr, predicates_jit)

        x = torch.randn(8, 3)
        eager_loss = LogicLoss(eager, post_processing="linear")
        jit_loss = LogicLoss(jitted, post_processing="linear")

        eager_loss.loss(**{x_var.name: x}).backward()
        jit_loss.loss(**{x_var.name: x}).backward()

        for m_eager, m_jit in zip(models_eager, models_jit, strict=True):
            assert m_eager.weight.grad is not None
            assert m_jit.weight.grad is not None
            assert torch.allclose(
                m_eager.weight.grad, m_jit.weight.grad, atol=1e-6
            )

    def test_mixed_operators_values_match(self) -> None:
        """Exercise Or, Not, Implies, and Equivalent (not just And) in a
        single formula large enough to trigger the jit path."""
        n = 12
        X = Variable("X")  # pylint: disable=invalid-name
        names = " ".join(f"P{i}" for i in range(n))
        symbols = Symbol(names)
        apps = [symbols[i](X) for i in range(n)]
        predicates = _constant_predicates(
            symbols, [0.05 * (i + 1) for i in range(n)]
        )

        expr = (
            Or(*apps[0:4])
            & Not(apps[4])
            & Implies(apps[5], apps[6])
            & Equivalent(apps[7], apps[8])
            & apps[9]
            & apps[10]
            & apps[11]
        )

        eager = TNormCompiler(jit=False).compile(expr, predicates)
        jitted = TNormCompiler(jit=True).compile(expr, predicates)

        x = torch.randn(8, 3)
        eager_out = eager(**{X.name: x})
        jit_out = jitted(**{X.name: x})

        assert torch.allclose(eager_out, jit_out, atol=1e-6)

    def test_predicate_with_data_dependent_control_flow(self) -> None:
        """Predicates are evaluated eagerly before the jitted combine step,
        so arbitrary Python control flow inside a predicate must work fine
        under jit=True -- it is never traced."""
        n = 12
        expr, x_var, symbols = _and_expr_with_n_atoms(n)

        def weird_predicate(x: torch.Tensor) -> torch.Tensor:
            if x.mean().item() > 0:
                return torch.sigmoid(x.mean(dim=-1))
            return torch.zeros(x.shape[0])

        predicates = _constant_predicates(
            symbols[:-1], [0.05 * (i + 1) for i in range(n - 1)]
        )
        predicates[str(symbols[-1])] = Predicate(weird_predicate)

        eager = TNormCompiler(jit=False).compile(expr, predicates)
        jitted = TNormCompiler(jit=True).compile(expr, predicates)

        x = torch.randn(8, 3)
        eager_out = eager(**{x_var.name: x})
        jit_out = jitted(**{x_var.name: x})

        assert torch.allclose(eager_out, jit_out, atol=1e-6)


class TestJitBareSymbol:
    """A bare (unapplied) predicate symbol is rejected by arity
    validation at compile() time, before either the eager or jit
    evaluation path ever runs -- so jit=True must raise the same error
    as jit=False, unchanged."""

    def test_bare_symbol_raises_same_error_regardless_of_jit(self) -> None:
        n = 9
        expr, _, symbols = _and_expr_with_n_atoms(n)
        # Replace the last conjunct with a bare (unapplied) symbol.
        bare = Symbol("Bare")
        expr = expr.func(*expr.args[:-1], bare)

        predicates = _constant_predicates(
            symbols, [0.05 * (i + 1) for i in range(n)]
        )
        predicates["Bare"] = Predicate(_make_constant_predicate(0.5))

        for jit in (False, True):
            compiler = TNormCompiler(jit=jit)
            with pytest.raises(ValueError, match="without arguments"):
                compiler.compile(expr, predicates)
