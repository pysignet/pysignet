"""HTML rendering helpers for ConsistencyReport.

Standalone functions that accept pre-computed metrics dicts
and return HTML strings. All rendering uses pure inline CSS,
no JavaScript, and no external dependencies.

All output is ASCII-only (no emojis, no unicode symbols).
"""

from typing import Any, Dict, List, Optional


_MAX_HISTORY_ROWS = 20


def _html_style() -> str:
    """Return shared CSS style block."""
    return (
        "<style>"
        ".pysignet-report {"
        "  font-family: Arial, Helvetica, sans-serif;"
        "  font-size: 13px;"
        "  line-height: 1.5;"
        "}"
        ".pysignet-report table {"
        "  border-collapse: collapse;"
        "  margin: 4px 0;"
        "}"
        ".pysignet-report th,"
        ".pysignet-report td {"
        "  padding: 4px 10px;"
        "  text-align: left;"
        "  border-bottom: 1px solid #ddd;"
        "}"
        ".pysignet-report th {"
        "  font-weight: 600;"
        "}"
        ".pysignet-report .num {"
        "  font-family: monospace;"
        "  text-align: right;"
        "}"
        ".pysignet-report hr {"
        "  border: none;"
        "  border-top: 1px solid #ddd;"
        "  margin: 8px 0;"
        "}"
        "</style>"
    )


def _html_header(title: str) -> str:
    """Return title div."""
    return (
        '<div style="font-weight:bold;'
        'margin-bottom:4px;">'
        f"{title}</div>"
    )


def _html_no_data() -> str:
    """Return no-data placeholder."""
    return (
        '<div class="pysignet-report">'
        + _html_style()
        + _html_header("ConsistencyReport")
        + '<div style="color:#888;">No data recorded yet.'
        " Call eval() to accumulate results.</div>"
        "</div>"
    )


def _html_bar(fraction: float, width_px: int = 120) -> str:
    """Return a horizontal CSS bar.

    Args:
        fraction: Fill fraction in [0, 1].
        width_px: Total bar width in pixels.

    Returns:
        HTML string for the bar.
    """
    pct = max(0.0, min(1.0, fraction)) * 100
    return (
        '<div style="display:inline-block;'
        f"width:{width_px}px;"
        "height:14px;"
        "background:#e0e0e0;"
        "border-radius:3px;"
        'overflow:hidden;">'
        '<div style="'
        f"width:{pct:.1f}%;"
        "height:100%;"
        "background:#4caf50;"
        '"></div>'
        "</div>"
    )


def _html_mini_bar(fraction: float) -> str:
    """Return a 60px mini bar for history tables."""
    return _html_bar(fraction, width_px=60)


def html_metrics_single(
    satisfied: int,
    total: int,
    rho: float,
    consistency: float,
    tau: Optional[float],
) -> str:
    """Render single-expression metrics card.

    Args:
        satisfied: Number of satisfied examples.
        total: Total number of examples.
        rho: Global violation rate.
        consistency: Global consistency rate.
        tau: Conditional violation rate (None if not
            an Implies expression).

    Returns:
        HTML string.
    """
    rows = (
        "<tr>"
        "<td>Satisfied</td>"
        f'<td class="num">{satisfied} / {total}</td>'
        "</tr>"
        "<tr>"
        "<td>Global Violation (rho)</td>"
        f'<td class="num">{rho:.4f}</td>'
        "</tr>"
        "<tr>"
        "<td>Global Consistency</td>"
        f'<td class="num">{consistency:.4f}</td>'
        "</tr>"
    )
    if tau is not None:
        rows += (
            "<tr>"
            "<td>Conditional Violation (tau)</td>"
            f'<td class="num">{tau:.4f}</td>'
            "</tr>"
        )
    return (
        '<div class="pysignet-report">'
        + _html_style()
        + _html_header("ConsistencyReport")
        + "<table>"
        + rows
        + "</table></div>"
    )


def html_metrics_multi(
    constraints: Dict[str, Dict[str, Any]],
    total: int,
    show_tau: bool,
) -> str:
    """Render multi-constraint metrics table.

    Args:
        constraints: Dict mapping constraint name to dict
            with keys: satisfied, rho, consistency, tau.
        total: Total number of examples.
        show_tau: Whether to show the tau column.

    Returns:
        HTML string.
    """
    header = (
        "<tr>"
        "<th>Constraint</th>"
        "<th>Satisfied</th>"
        "<th>rho</th>"
        "<th>Consistency</th>"
    )
    if show_tau:
        header += "<th>tau</th>"
    header += "</tr>"

    rows = ""
    for name, metrics in constraints.items():
        rows += (
            "<tr>"
            f"<td>{name}</td>"
            f'<td class="num">'
            f'{metrics["satisfied"]} / {total}</td>'
            f'<td class="num">{metrics["rho"]:.4f}</td>'
            f'<td class="num">'
            f'{metrics["consistency"]:.4f}</td>'
        )
        if show_tau:
            rows += (
                f'<td class="num">'
                f'{metrics["tau"]:.4f}</td>'
            )
        rows += "</tr>"

    return (
        '<div class="pysignet-report">'
        + _html_style()
        + _html_header(
            f"ConsistencyReport ({total} examples, "
            f"{len(constraints)} constraints)"
        )
        + "<table>"
        + header
        + rows
        + "</table></div>"
    )


def html_chart_single(
    satisfied: int,
    total: int,
    rho: float,
    consistency: float,
    tau: Optional[float],
) -> str:
    """Render single-expression metrics + bar chart.

    Args:
        satisfied: Number of satisfied examples.
        total: Total number of examples.
        rho: Global violation rate.
        consistency: Global consistency rate.
        tau: Conditional violation rate (None if not
            an Implies expression).

    Returns:
        HTML string.
    """
    rows = (
        "<tr>"
        "<td>Satisfied</td>"
        f'<td class="num">{satisfied} / {total}</td>'
        "</tr>"
        "<tr>"
        "<td>Global Violation (rho)</td>"
        f'<td class="num">{rho:.4f}</td>'
        "</tr>"
        "<tr>"
        "<td>Global Consistency</td>"
        f'<td class="num">{consistency:.4f}</td>'
        "<td>" + _html_bar(consistency) + "</td>"
        "</tr>"
    )
    if tau is not None:
        rows += (
            "<tr>"
            "<td>Conditional Violation (tau)</td>"
            f'<td class="num">{tau:.4f}</td>'
            "</tr>"
        )
    return (
        '<div class="pysignet-report">'
        + _html_style()
        + _html_header("ConsistencyReport")
        + "<table>"
        + rows
        + "</table></div>"
    )


def html_chart_multi(
    constraints: Dict[str, Dict[str, Any]],
    total: int,
    show_tau: bool,
) -> str:
    """Render multi-constraint metrics table + bars.

    Args:
        constraints: Dict mapping constraint name to dict
            with keys: satisfied, rho, consistency, tau.
        total: Total number of examples.
        show_tau: Whether to show the tau column.

    Returns:
        HTML string.
    """
    header = (
        "<tr>"
        "<th>Constraint</th>"
        "<th>Satisfied</th>"
        "<th>rho</th>"
        "<th>Consistency</th>"
        "<th></th>"
    )
    if show_tau:
        header += "<th>tau</th>"
    header += "</tr>"

    rows = ""
    for name, metrics in constraints.items():
        rows += (
            "<tr>"
            f"<td>{name}</td>"
            f'<td class="num">'
            f'{metrics["satisfied"]} / {total}</td>'
            f'<td class="num">{metrics["rho"]:.4f}</td>'
            f'<td class="num">'
            f'{metrics["consistency"]:.4f}</td>'
            "<td>"
            + _html_bar(metrics["consistency"])
            + "</td>"
        )
        if show_tau:
            rows += (
                f'<td class="num">'
                f'{metrics["tau"]:.4f}</td>'
            )
        rows += "</tr>"

    return (
        '<div class="pysignet-report">'
        + _html_style()
        + _html_header(
            f"ConsistencyReport ({total} examples, "
            f"{len(constraints)} constraints)"
        )
        + "<table>"
        + header
        + rows
        + "</table></div>"
    )


def html_history_single(
    satisfied: int,
    total: int,
    rho: float,
    consistency: float,
    tau: Optional[float],
    history: List[Dict[str, Any]],
) -> str:
    """Render single-expression metrics + history table.

    Args:
        satisfied: Number of satisfied examples.
        total: Total number of examples.
        rho: Global violation rate.
        consistency: Global consistency rate.
        tau: Conditional violation rate.
        history: List of per-batch history entries.

    Returns:
        HTML string.
    """
    metrics = html_metrics_single(
        satisfied, total, rho, consistency, tau
    )
    # Strip closing </div> to append history
    metrics = metrics.rsplit("</div>", 1)[0]

    header = (
        "<tr>"
        "<th>#</th>"
        "<th>Batch Size</th>"
        "<th>rho</th>"
        "<th></th>"
        "</tr>"
    )

    # Truncate to last N rows
    display_history = history
    offset = 0
    if len(history) > _MAX_HISTORY_ROWS:
        offset = len(history) - _MAX_HISTORY_ROWS
        display_history = history[offset:]

    rows = ""
    for i, entry in enumerate(display_history):
        row_num = offset + i + 1
        batch_rho = entry["rho"]
        rows += (
            "<tr>"
            f'<td class="num">{row_num}</td>'
            f'<td class="num">{entry["batch_size"]}</td>'
            f'<td class="num">{batch_rho:.4f}</td>'
            f"<td>{_html_mini_bar(1.0 - batch_rho)}</td>"
            "</tr>"
        )

    return (
        metrics
        + "<hr>"
        + "<table>"
        + header
        + rows
        + "</table></div>"
    )


def html_history_multi(
    constraints: Dict[str, Dict[str, Any]],
    total: int,
    show_tau: bool,
    constraint_names: List[str],
    history: List[Dict[str, Any]],
) -> str:
    """Render multi-constraint metrics + history table.

    Args:
        constraints: Dict mapping constraint name to dict
            with keys: satisfied, rho, consistency, tau.
        total: Total number of examples.
        show_tau: Whether to show the tau column.
        constraint_names: Ordered list of constraint names.
        history: List of per-batch history entries.

    Returns:
        HTML string.
    """
    metrics = html_metrics_multi(
        constraints, total, show_tau
    )
    # Strip closing </div> to append history
    metrics = metrics.rsplit("</div>", 1)[0]

    header = "<tr><th>#</th><th>Batch Size</th>"
    for name in constraint_names:
        header += f"<th>rho ({name})</th>"
    header += "</tr>"

    # Truncate to last N rows
    display_history = history
    offset = 0
    if len(history) > _MAX_HISTORY_ROWS:
        offset = len(history) - _MAX_HISTORY_ROWS
        display_history = history[offset:]

    rows = ""
    for i, entry in enumerate(display_history):
        row_num = offset + i + 1
        rows += (
            "<tr>"
            f'<td class="num">{row_num}</td>'
            f'<td class="num">{entry["batch_size"]}</td>'
        )
        for name in constraint_names:
            c_rho = entry["constraints"][name]["rho"]
            rows += f'<td class="num">{c_rho:.4f}</td>'
        rows += "</tr>"

    return (
        metrics
        + "<hr>"
        + "<table>"
        + header
        + rows
        + "</table></div>"
    )
