"""Tests for Jupyter/IPython display of pysignet objects.

Covers _pretty (SymPy pretty printer) and _latex (SymPy LaTeX printer)
for PredicateApplication and Quantifiers, plus __repr__ / _repr_pretty_
for CompiledExpression and LogicLoss.

These prevent regressions where nested custom SymPy nodes render as
bare class names (e.g. "PredicateApplication<=>PredicateApplication")
instead of human-readable strings.
"""

import pytest
import sympy as sp
import torch
from sympy.printing.latex import latex as sympy_latex
from sympy.printing.pretty import pretty as sympy_pretty
from sympy.printing.pretty.stringpict import prettyForm

import pysignet as psn
from pysignet.logic import Variable
from pysignet.logic.quantifier import Exists, ForAll
from pysignet.symbols import PredicateApplication

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_predicates():
    """Return simple lambda predicates that always return 0.5."""
    return {
        "P": lambda x: torch.full((x.shape[0],), 0.5),
        "Q": lambda x: torch.full((x.shape[0],), 0.5),
        "Similar": lambda x1, x2: torch.full((x1.shape[0],), 0.5),
    }


# ---------------------------------------------------------------------------
# PredicateApplication display
# ---------------------------------------------------------------------------

class TestPredicateApplicationDisplay:
    """Test _pretty and _latex for PredicateApplication."""

    def test_pretty_returns_prettyform(self):
        """_pretty should return a prettyForm instance."""
        X = Variable("X")
        P = psn.Symbol("P")
        app = P(X)
        result = app._pretty(printer=None)
        assert isinstance(result, prettyForm)

    def test_pretty_contains_predicate_name(self):
        """_pretty output should include the predicate name."""
        X = Variable("X")
        P = psn.Symbol("MyPred")
        app = P(X)
        result = app._pretty(printer=None)
        assert "MyPred" in str(result)

    def test_pretty_contains_arguments(self):
        """_pretty output should include the arguments."""
        X = Variable("X")
        P = psn.Symbol("Digit")
        app = P(X, 3)
        result = app._pretty(printer=None)
        text = str(result)
        assert "X" in text
        assert "3" in text

    def test_latex_returns_string(self):
        """_latex should return a string."""
        X = Variable("X")
        P = psn.Symbol("P")
        app = P(X)
        result = app._latex(printer=None)
        assert isinstance(result, str)

    def test_latex_contains_predicate_name(self):
        """_latex output should include the predicate name."""
        X = Variable("X")
        P = psn.Symbol("Similar")
        app = P(X)
        result = app._latex(printer=None)
        assert "Similar" in result

    def test_latex_contains_arguments(self):
        """_latex output should contain argument strings."""
        X = Variable("X")
        P = psn.Symbol("Digit")
        app = P(X, 5)
        result = app._latex(printer=None)
        assert "5" in result

    def test_sympy_pretty_nested_in_equivalent(self):
        """SymPy pretty printer should show predicate names, not class names."""
        Similar = psn.Symbol("Similar")
        X1, X2 = Variable("X1 X2")
        expr = sp.Equivalent(Similar(X1, X2), Similar(X2, X1))
        output = sympy_pretty(expr)
        assert "PredicateApplication" not in output
        assert "Similar" in output

    def test_sympy_pretty_nested_in_implies(self):
        """Pretty-printed Implies should show predicate names."""
        P, Q = psn.Symbol("P Q")
        X = Variable("X")
        expr = sp.Implies(P(X), Q(X))
        output = sympy_pretty(expr)
        assert "PredicateApplication" not in output
        assert "P" in output
        assert "Q" in output

    def test_sympy_latex_nested_in_equivalent(self):
        """LaTeX rendering of nested predicates should not use class names."""
        Similar = psn.Symbol("Similar")
        X1, X2 = Variable("X1 X2")
        expr = sp.Equivalent(Similar(X1, X2), Similar(X2, X1))
        output = sympy_latex(expr)
        assert "PredicateApplication" not in output
        assert "Similar" in output

    def test_sympy_latex_nested_in_and(self):
        """LaTeX rendering of And should show predicate names."""
        P, Q = psn.Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))
        output = sympy_latex(expr)
        assert "PredicateApplication" not in output

    def test_pretty_unary_predicate(self):
        """Single-argument predicate displays correctly."""
        X = Variable("X")
        P = psn.Symbol("IsDigit")
        app = P(X)
        result = str(app._pretty(printer=None))
        assert result == "IsDigit(X)"

    def test_pretty_binary_predicate(self):
        """Two-argument predicate includes both args."""
        X1, X2 = Variable("X1 X2")
        S = psn.Symbol("Similar")
        app = S(X1, X2)
        result = str(app._pretty(printer=None))
        assert result == "Similar(X1, X2)"

    def test_latex_uses_text_command(self):
        """LaTeX output uses \\text{} to render the predicate name."""
        X = Variable("X")
        P = psn.Symbol("P")
        app = P(X)
        result = app._latex(printer=None)
        assert r"\text{P}" in result


# ---------------------------------------------------------------------------
# ForAll / Exists display
# ---------------------------------------------------------------------------

class TestQuantifierDisplay:
    """Test _pretty and _latex for ForAll and Exists."""

    def test_forall_pretty_returns_prettyform(self):
        """ForAll._pretty should return a prettyForm."""
        X = Variable("X")
        P = psn.Symbol("P")
        expr = ForAll(X, [0, 1, 2], P(X))
        result = expr._pretty(printer=None)
        assert isinstance(result, prettyForm)

    def test_forall_pretty_contains_forall(self):
        """ForAll pretty output should contain 'ForAll'."""
        X = Variable("X")
        P = psn.Symbol("P")
        expr = ForAll(X, [0, 1, 2], P(X))
        result = str(expr._pretty(printer=None))
        assert "ForAll" in result

    def test_exists_pretty_returns_prettyform(self):
        """Exists._pretty should return a prettyForm."""
        X = Variable("X")
        P = psn.Symbol("P")
        expr = Exists(X, [0, 1, 2], P(X))
        result = expr._pretty(printer=None)
        assert isinstance(result, prettyForm)

    def test_exists_pretty_contains_exists(self):
        """Exists pretty output should contain 'Exists'."""
        X = Variable("X")
        P = psn.Symbol("P")
        expr = Exists(X, range(3), P(X))
        result = str(expr._pretty(printer=None))
        assert "Exists" in result

    def test_forall_latex_returns_string(self):
        """ForAll._latex should return a string."""
        X = Variable("X")
        P = psn.Symbol("P")
        expr = ForAll(X, [0, 1], P(X))
        result = expr._latex(printer=None)
        assert isinstance(result, str)

    def test_forall_latex_contains_forall(self):
        """ForAll LaTeX output should reference 'ForAll'."""
        X = Variable("X")
        P = psn.Symbol("P")
        expr = ForAll(X, [0, 1], P(X))
        result = expr._latex(printer=None)
        assert "ForAll" in result

    def test_exists_latex_contains_exists(self):
        """Exists LaTeX output should reference 'Exists'."""
        X = Variable("X")
        P = psn.Symbol("P")
        expr = Exists(X, [0, 1], P(X))
        result = expr._latex(printer=None)
        assert "Exists" in result

    def test_forall_pretty_nested_in_implies(self):
        """ForAll nested in Implies should not show class name."""
        X = Variable("X")
        P, Q = psn.Symbol("P Q")
        forall = ForAll(X, [0, 1], P(X))
        expr = sp.Implies(forall, Q(X))
        output = sympy_pretty(expr)
        assert "Quantifier" not in output
        assert "ForAll" in output

    def test_exists_pretty_nested_in_and(self):
        """Exists nested in And should not show class name."""
        X = Variable("X")
        P, Q = psn.Symbol("P Q")
        exists = Exists(X, range(3), P(X))
        expr = sp.And(exists, Q(X))
        output = sympy_pretty(expr)
        assert "Quantifier" not in output
        assert "Exists" in output

    def test_forall_large_domain_truncated(self):
        """ForAll pretty/latex should truncate large domains."""
        X = Variable("X")
        P = psn.Symbol("P")
        expr = ForAll(X, list(range(20)), P(X))
        pretty_out = str(expr._pretty(printer=None))
        # Should not list all 20 elements
        assert "..." in pretty_out

    def test_exists_large_domain_truncated(self):
        """Exists pretty/latex should truncate large domains."""
        X = Variable("X")
        P = psn.Symbol("P")
        expr = Exists(X, list(range(20)), P(X))
        pretty_out = str(expr._pretty(printer=None))
        assert "..." in pretty_out


# ---------------------------------------------------------------------------
# CompiledExpression display
# ---------------------------------------------------------------------------

class TestCompiledExpressionDisplay:
    """Test __repr__ and _repr_pretty_ for CompiledExpression."""

    def test_repr_contains_expr(self):
        """CompiledExpression repr should include the expression."""
        X = Variable("X")
        P, Q = psn.Symbol("P Q")
        expr = sp.Implies(P(X), Q(X))
        compiled = psn.compile_logic(expr, _make_simple_predicates())
        r = repr(compiled)
        assert "CompiledExpression" in r
        assert "P" in r or "Q" in r

    def test_repr_contains_free_variables(self):
        """CompiledExpression repr should list free variables."""
        X = Variable("X")
        P = psn.Symbol("P")
        compiled = psn.compile_logic(P(X), _make_simple_predicates())
        r = repr(compiled)
        assert "X" in r

    def test_repr_pretty_matches_repr(self):
        """_repr_pretty_ should produce the same text as repr."""

        class FakePrinter:
            def __init__(self):
                self.captured = ""

            def text(self, s: str) -> None:
                self.captured = s

        X = Variable("X")
        P = psn.Symbol("P")
        compiled = psn.compile_logic(P(X), _make_simple_predicates())
        printer = FakePrinter()
        compiled._repr_pretty_(printer, cycle=False)
        assert printer.captured == repr(compiled)

    def test_repr_shows_predicates(self):
        """CompiledExpression repr should list predicate names."""
        X = Variable("X")
        P, Q = psn.Symbol("P Q")
        expr = sp.And(P(X), Q(X))
        compiled = psn.compile_logic(expr, _make_simple_predicates())
        r = repr(compiled)
        assert "P" in r
        assert "Q" in r


# ---------------------------------------------------------------------------
# LogicLoss display
# ---------------------------------------------------------------------------

class TestLogicLossDisplay:
    """Test __repr__ and _repr_pretty_ for LogicLoss."""

    def test_repr_is_string(self):
        """LogicLoss repr should return a string."""
        X = Variable("X")
        P = psn.Symbol("P")
        expr = P(X)
        ll = psn.logic_to_loss(expr, _make_simple_predicates())
        assert isinstance(repr(ll), str)

    def test_repr_contains_logicLoss(self):
        """LogicLoss repr should contain 'LogicLoss'."""
        X = Variable("X")
        P = psn.Symbol("P")
        ll = psn.logic_to_loss(P(X), _make_simple_predicates())
        assert "LogicLoss" in repr(ll)

    def test_repr_contains_free_variables(self):
        """LogicLoss repr should list free variables."""
        X = Variable("X")
        P = psn.Symbol("P")
        ll = psn.logic_to_loss(P(X), _make_simple_predicates())
        r = repr(ll)
        assert "X" in r

    def test_repr_contains_post_processing(self):
        """LogicLoss repr should show post_processing mode."""
        X = Variable("X")
        P = psn.Symbol("P")
        ll = psn.logic_to_loss(P(X), _make_simple_predicates())
        r = repr(ll)
        assert "post_processing" in r

    def test_repr_contains_compiler(self):
        """LogicLoss repr should include the compiler class name."""
        X = Variable("X")
        P = psn.Symbol("P")
        ll = psn.logic_to_loss(P(X), _make_simple_predicates())
        r = repr(ll)
        assert "Compiler" in r

    def test_repr_contains_expr(self):
        """LogicLoss repr should include the expression when available."""
        X = Variable("X")
        P, Q = psn.Symbol("P Q")
        expr = sp.Implies(P(X), Q(X))
        ll = psn.logic_to_loss(expr, _make_simple_predicates())
        r = repr(ll)
        assert "P" in r or "Q" in r

    def test_repr_pretty_matches_repr(self):
        """_repr_pretty_ should produce the same text as repr."""

        class FakePrinter:
            def __init__(self):
                self.captured = ""

            def text(self, s: str) -> None:
                self.captured = s

        X = Variable("X")
        P = psn.Symbol("P")
        ll = psn.logic_to_loss(P(X), _make_simple_predicates())
        printer = FakePrinter()
        ll._repr_pretty_(printer, cycle=False)
        assert printer.captured == repr(ll)

    def test_repr_not_object_address(self):
        """LogicLoss repr should not be the default object address repr."""
        X = Variable("X")
        P = psn.Symbol("P")
        ll = psn.logic_to_loss(P(X), _make_simple_predicates())
        r = repr(ll)
        assert "object at 0x" not in r
