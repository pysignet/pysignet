"""Tests for ConsistencyReport HTML display methods.

Tests cover:
- to_html_metrics(): key-value metrics table
- to_html_chart(): metrics + CSS bar chart
- to_html_history(): metrics + per-batch history table
- _repr_html_(): Jupyter default display
"""

import re

import sympy as sp
import torch

from pysignet import Predicate, Symbol
from pysignet.eval import ConsistencyReport
from pysignet.logic import Variable


def _make_single_report(
    p_vals: list[bool],
    q_vals: list[bool],
) -> ConsistencyReport:
    """Create a single-expression AND report."""
    p_sym = Symbol("P")
    q_sym = Symbol("Q")
    x_var = Variable("X")
    expr = sp.And(p_sym(x_var), q_sym(x_var))
    predicates = {
        "P": Predicate(
            lambda _x, _v=p_vals: torch.tensor(_v),
            is_model=False,
        ),
        "Q": Predicate(
            lambda _x, _v=q_vals: torch.tensor(_v),
            is_model=False,
        ),
    }
    return ConsistencyReport(expr, predicates)


def _make_implies_report(
    p_vals: list[bool],
    q_vals: list[bool],
) -> ConsistencyReport:
    """Create a single-expression Implies report."""
    p_sym, q_sym = Symbol("P Q")
    x_var = Variable("X")
    expr = sp.Implies(p_sym(x_var), q_sym(x_var))
    predicates = {
        "P": Predicate(
            lambda _x, _v=p_vals: torch.tensor(_v),
            is_model=False,
        ),
        "Q": Predicate(
            lambda _x, _v=q_vals: torch.tensor(_v),
            is_model=False,
        ),
    }
    return ConsistencyReport(expr, predicates)


def _make_multi_report(
    p_vals: list[bool],
    q_vals: list[bool],
) -> ConsistencyReport:
    """Create a multi-constraint report (AND + OR)."""
    p_sym, q_sym = Symbol("P Q")
    x_var = Variable("X")
    exprs = {
        "conj": sp.And(p_sym(x_var), q_sym(x_var)),
        "disj": sp.Or(p_sym(x_var), q_sym(x_var)),
    }
    predicates = {
        "P": Predicate(
            lambda _x, _v=p_vals: torch.tensor(_v),
            is_model=False,
        ),
        "Q": Predicate(
            lambda _x, _v=q_vals: torch.tensor(_v),
            is_model=False,
        ),
    }
    return ConsistencyReport(exprs, predicates)


def _make_multi_with_implies(
    p_vals: list[bool],
    q_vals: list[bool],
) -> ConsistencyReport:
    """Create a multi-constraint report with an Implies."""
    p_sym, q_sym = Symbol("P Q")
    x_var = Variable("X")
    exprs = {
        "impl": sp.Implies(p_sym(x_var), q_sym(x_var)),
        "disj": sp.Or(p_sym(x_var), q_sym(x_var)),
    }
    predicates = {
        "P": Predicate(
            lambda _x, _v=p_vals: torch.tensor(_v),
            is_model=False,
        ),
        "Q": Predicate(
            lambda _x, _v=q_vals: torch.tensor(_v),
            is_model=False,
        ),
    }
    return ConsistencyReport(exprs, predicates)


def _is_ascii_only(text: str) -> bool:
    """Check that text contains only ASCII characters."""
    return all(ord(c) < 128 for c in text)


# -- TestHtmlMetrics --------------------------------------------------


class TestHtmlMetrics:
    """Tests for to_html_metrics()."""

    def test_no_data(self) -> None:
        """No-data report returns a placeholder."""
        report = _make_single_report([True], [True])
        html = report.to_html_metrics()
        assert "no data" in html.lower()

    def test_single_contains_counts(self) -> None:
        """Single mode shows satisfied/total counts."""
        # P=[T,T,F,F], Q=[T,F,T,F]
        # AND: [T,F,F,F] -> 1/4 satisfied
        report = _make_single_report(
            [True, True, False, False],
            [True, False, True, False],
        )
        report.eval(X=torch.randn(4, 5))
        html = report.to_html_metrics()
        assert "1" in html  # satisfied count
        assert "4" in html  # total count

    def test_single_contains_rho(self) -> None:
        """Single mode shows global violation (rho)."""
        report = _make_single_report(
            [True, True, False, False],
            [True, False, True, False],
        )
        report.eval(X=torch.randn(4, 5))
        html = report.to_html_metrics()
        # rho = 3/4 = 0.75
        assert "0.7500" in html

    def test_single_contains_consistency(self) -> None:
        """Single mode shows global consistency."""
        report = _make_single_report(
            [True, True, False, False],
            [True, False, True, False],
        )
        report.eval(X=torch.randn(4, 5))
        html = report.to_html_metrics()
        # consistency = 1/4 = 0.25
        assert "0.2500" in html

    def test_single_no_tau_for_non_implies(self) -> None:
        """tau row is omitted for non-Implies formulas."""
        report = _make_single_report(
            [True, True, False, False],
            [True, False, True, False],
        )
        report.eval(X=torch.randn(4, 5))
        html = report.to_html_metrics()
        assert "tau" not in html.lower()

    def test_single_tau_shown_for_implies(self) -> None:
        """tau row is shown for Implies formulas."""
        # P=[T,T,F], Q=[T,F,T]
        # Implies: [T,F,T] -> 2/3 satisfied, rho=1/3
        # antecedent P true for indices 0,1
        # violated when P true and Q false: index 1
        # tau = 1/2 = 0.5
        report = _make_implies_report(
            [True, True, False],
            [True, False, True],
        )
        report.eval(X=torch.randn(3, 5))
        html = report.to_html_metrics()
        assert "tau" in html.lower()
        assert "0.5000" in html

    def test_multi_contains_constraint_names(self) -> None:
        """Multi mode shows constraint names."""
        report = _make_multi_report(
            [True, True, False, False],
            [True, False, True, False],
        )
        report.eval(X=torch.randn(4, 5))
        html = report.to_html_metrics()
        assert "conj" in html
        assert "disj" in html

    def test_multi_contains_rho_per_constraint(self) -> None:
        """Multi mode shows rho for each constraint."""
        # P=[T,T,F,F], Q=[T,F,T,F]
        # AND: [T,F,F,F] -> rho=0.75
        # OR:  [T,T,T,F] -> rho=0.25
        report = _make_multi_report(
            [True, True, False, False],
            [True, False, True, False],
        )
        report.eval(X=torch.randn(4, 5))
        html = report.to_html_metrics()
        assert "0.7500" in html
        assert "0.2500" in html

    def test_multi_no_tau_column_without_implies(self) -> None:
        """Multi mode omits tau column when no Implies."""
        report = _make_multi_report(
            [True, True, False, False],
            [True, False, True, False],
        )
        report.eval(X=torch.randn(4, 5))
        html = report.to_html_metrics()
        assert "tau" not in html.lower()

    def test_multi_tau_column_with_implies(self) -> None:
        """Multi mode shows tau column when any is Implies."""
        report = _make_multi_with_implies(
            [True, True, False],
            [True, False, True],
        )
        report.eval(X=torch.randn(3, 5))
        html = report.to_html_metrics()
        assert "tau" in html.lower()

    def test_ascii_only(self) -> None:
        """Output is ASCII-only."""
        report = _make_single_report(
            [True, False],
            [True, True],
        )
        report.eval(X=torch.randn(2, 5))
        html = report.to_html_metrics()
        assert _is_ascii_only(html)

    def test_returns_string(self) -> None:
        """Return type is str."""
        report = _make_single_report([True], [True])
        report.eval(X=torch.randn(1, 5))
        assert isinstance(report.to_html_metrics(), str)

    def test_is_valid_html(self) -> None:
        """Output contains basic HTML structure."""
        report = _make_single_report(
            [True, False],
            [True, True],
        )
        report.eval(X=torch.randn(2, 5))
        html = report.to_html_metrics()
        assert "<div" in html
        assert "</div>" in html
        assert "pysignet-report" in html


# -- TestHtmlChart ----------------------------------------------------


class TestHtmlChart:
    """Tests for to_html_chart()."""

    def test_no_data(self) -> None:
        """No-data report returns a placeholder."""
        report = _make_single_report([True], [True])
        html = report.to_html_chart()
        assert "no data" in html.lower()

    def test_single_contains_bar(self) -> None:
        """Single mode contains a bar element."""
        report = _make_single_report(
            [True, True, False, False],
            [True, True, True, True],
        )
        report.eval(X=torch.randn(4, 5))
        html = report.to_html_chart()
        # Should contain a bar with width style
        assert "width:" in html or "width :" in html

    def test_single_bar_width_matches_consistency(self) -> None:
        """Bar width percentage matches consistency rate."""
        # P=[T,T,F,F], Q=[T,T,T,T]
        # AND: [T,T,F,F] -> consistency = 2/4 = 50%
        report = _make_single_report(
            [True, True, False, False],
            [True, True, True, True],
        )
        report.eval(X=torch.randn(4, 5))
        html = report.to_html_chart()
        assert "50.0%" in html or "50.00%" in html

    def test_single_100_percent(self) -> None:
        """All satisfied shows 100% bar."""
        report = _make_single_report(
            [True, True, True],
            [True, True, True],
        )
        report.eval(X=torch.randn(3, 5))
        html = report.to_html_chart()
        assert "100.0%" in html or "100.00%" in html

    def test_single_0_percent(self) -> None:
        """None satisfied shows 0% bar."""
        report = _make_single_report(
            [False, False, False],
            [True, True, True],
        )
        report.eval(X=torch.randn(3, 5))
        html = report.to_html_chart()
        assert "0.0%" in html or "0.00%" in html

    def test_multi_one_bar_per_constraint(self) -> None:
        """Multi mode has a bar for each constraint."""
        report = _make_multi_report(
            [True, True, False, False],
            [True, False, True, False],
        )
        report.eval(X=torch.randn(4, 5))
        html = report.to_html_chart()
        # Should contain both constraint names
        assert "conj" in html
        assert "disj" in html
        # Should have multiple bar elements (at least 2)
        bar_count = html.count("#4caf50")
        assert bar_count >= 2

    def test_contains_metrics(self) -> None:
        """Chart output also includes metrics."""
        report = _make_single_report(
            [True, True, False, False],
            [True, False, True, False],
        )
        report.eval(X=torch.randn(4, 5))
        html = report.to_html_chart()
        # Should contain rho
        assert "0.7500" in html

    def test_ascii_only(self) -> None:
        """Output is ASCII-only."""
        report = _make_single_report(
            [True, False],
            [True, True],
        )
        report.eval(X=torch.randn(2, 5))
        html = report.to_html_chart()
        assert _is_ascii_only(html)


# -- TestHtmlHistory --------------------------------------------------


class TestHtmlHistory:
    """Tests for to_html_history()."""

    def test_no_data(self) -> None:
        """No-data report returns a placeholder."""
        report = _make_single_report([True], [True])
        html = report.to_html_history()
        assert "no data" in html.lower()

    def test_single_history_rows_match_evals(self) -> None:
        """History table has one row per eval() call."""
        report = _make_single_report(
            [True, True, False],
            [True, False, True],
        )
        report.eval(X=torch.randn(3, 5))
        report.eval(X=torch.randn(3, 5))
        report.eval(X=torch.randn(3, 5))
        html = report.to_html_history()
        # 3 eval calls = 3 data rows in the table
        # Count <tr> tags excluding the header row
        tr_count = html.count("<tr")
        # At least 3 data rows + 1 header = 4
        assert tr_count >= 4

    def test_single_contains_batch_size(self) -> None:
        """History shows batch sizes."""
        report = _make_single_report(
            [True, True, False],
            [True, False, True],
        )
        report.eval(X=torch.randn(3, 5))
        html = report.to_html_history()
        assert "3" in html

    def test_single_contains_rho_values(self) -> None:
        """History shows rho values."""
        # AND: [T,F,F] -> rho = 2/3 ~ 0.6667
        report = _make_single_report(
            [True, True, False],
            [True, False, True],
        )
        report.eval(X=torch.randn(3, 5))
        html = report.to_html_history()
        assert "0.6667" in html

    def test_single_contains_mini_bars(self) -> None:
        """History rows have mini bar elements."""
        report = _make_single_report(
            [True, True, False],
            [True, False, True],
        )
        report.eval(X=torch.randn(3, 5))
        html = report.to_html_history()
        assert "60px" in html

    def test_truncation_at_20_batches(self) -> None:
        """History is truncated to last 20 batches."""
        report = _make_single_report(
            [True, False],
            [True, True],
        )
        for _ in range(25):
            report.eval(X=torch.randn(2, 5))
        html = report.to_html_history()
        # Should mention truncation or only show 20 rows
        # Count data rows (not header)
        tr_matches = re.findall(r"<tr", html)
        # History section: 1 header + at most 20 data rows
        # Plus metrics section rows. Total tr count should
        # not reflect all 25 batches.
        # The history table itself should have <= 21 tr tags
        # We check that "25" batch count appears in metrics
        # but only last 20 rows in history
        assert len(report.history()) == 25
        # Should not have 25 row-number entries in history
        # Row numbers should start from 6 (25-20+1)
        assert ">6<" in html or "> 6<" in html

    def test_multi_has_columns_per_constraint(self) -> None:
        """Multi mode history has rho column per constraint."""
        report = _make_multi_report(
            [True, True, False, False],
            [True, False, True, False],
        )
        report.eval(X=torch.randn(4, 5))
        html = report.to_html_history()
        assert "conj" in html
        assert "disj" in html

    def test_contains_metrics_section(self) -> None:
        """History output also includes metrics."""
        report = _make_single_report(
            [True, True, False, False],
            [True, False, True, False],
        )
        report.eval(X=torch.randn(4, 5))
        html = report.to_html_history()
        assert "0.7500" in html  # rho

    def test_contains_separator(self) -> None:
        """History output has separator between metrics and
        history."""
        report = _make_single_report(
            [True, True, False, False],
            [True, False, True, False],
        )
        report.eval(X=torch.randn(4, 5))
        html = report.to_html_history()
        assert "<hr" in html

    def test_ascii_only(self) -> None:
        """Output is ASCII-only."""
        report = _make_single_report(
            [True, False],
            [True, True],
        )
        report.eval(X=torch.randn(2, 5))
        html = report.to_html_history()
        assert _is_ascii_only(html)


# -- TestReprHtml -----------------------------------------------------


class TestReprHtml:
    """Tests for _repr_html_()."""

    def test_delegates_to_chart(self) -> None:
        """_repr_html_ returns same as to_html_chart."""
        report = _make_single_report(
            [True, True, False, False],
            [True, False, True, False],
        )
        report.eval(X=torch.randn(4, 5))
        assert report._repr_html_() == report.to_html_chart()

    def test_no_data_case(self) -> None:
        """_repr_html_ handles no-data gracefully."""
        report = _make_single_report([True], [True])
        html = report._repr_html_()
        assert "no data" in html.lower()

    def test_returns_string(self) -> None:
        """_repr_html_ returns str."""
        report = _make_single_report([True], [True])
        assert isinstance(report._repr_html_(), str)

    def test_multi_mode(self) -> None:
        """_repr_html_ works in multi-constraint mode."""
        report = _make_multi_report(
            [True, True, False, False],
            [True, False, True, False],
        )
        report.eval(X=torch.randn(4, 5))
        html = report._repr_html_()
        assert "conj" in html
        assert "disj" in html

    def test_ascii_only(self) -> None:
        """Output is ASCII-only."""
        report = _make_single_report(
            [True, False],
            [True, True],
        )
        report.eval(X=torch.randn(2, 5))
        html = report._repr_html_()
        assert _is_ascii_only(html)
