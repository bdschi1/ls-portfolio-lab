"""Markdown client-snapshot report for an L/S portfolio.

The wealth-management plugin's `/client-report` skill assumes asset-class
allocation (equity/bonds/alternatives). That schema doesn't fit an L/S
equity book. This module produces the L/S-appropriate analogue:

- exposure summary (gross, net, long/short, leverage)
- sector exposure (long vs short)
- top holdings (longs + shorts)
- unrealized P&L summary

Optional snapshot history (TWR) can be passed via `history_summary`.
Risk metrics from `core/metrics/` can be passed via `risk_summary`.

Pure function, no Streamlit imports — consumable from the skill, the
Streamlit page, or any CLI export path.
"""

from __future__ import annotations

from datetime import date

from core.portfolio import Portfolio


def render_client_report(
    portfolio: Portfolio,
    *,
    report_date: date | None = None,
    top_n: int = 10,
    history_summary: dict | None = None,
    risk_summary: dict | None = None,
) -> str:
    """Return a markdown snapshot of the portfolio for client distribution.

    `history_summary` (optional) is rendered as a performance table. Expected
    keys: `qtd`, `ytd`, `one_year`, `itd` (each a float return). Missing keys
    render as "—".

    `risk_summary` (optional) is rendered as a risk metrics block. Expected
    keys: `vol_ann`, `beta`, `sharpe`, `max_drawdown`. Missing keys render as
    "—".
    """
    report_date = report_date or date.today()
    sections: list[str] = []
    sections.append(_header(portfolio, report_date))
    sections.append(_exposure_section(portfolio))
    if history_summary is not None:
        sections.append(_performance_section(history_summary))
    if risk_summary is not None:
        sections.append(_risk_section(risk_summary))
    sections.append(_sector_section(portfolio))
    sections.append(_holdings_section(portfolio, top_n))
    sections.append(_disclosure_section())
    return "\n\n".join(sections)


def _header(p: Portfolio, d: date) -> str:
    return (
        f"# Portfolio Snapshot — {p.name}\n\n"
        f"**As of:** {d.isoformat()}  \n"
        f"**NAV:** ${p.nav:,.0f}  \n"
        f"**Inception:** {p.inception_date.isoformat()}  \n"
        f"**Benchmark:** {p.benchmark}"
    )


def _exposure_section(p: Portfolio) -> str:
    return (
        "## Exposure Summary\n\n"
        "| Measure | Value |\n"
        "|---|---|\n"
        f"| Gross exposure | {p.gross_exposure:.1%} |\n"
        f"| Net exposure | {p.net_exposure:+.1%} |\n"
        f"| Long notional | ${p.long_notional:,.0f} ({p.long_count} positions) |\n"
        f"| Short notional | ${p.short_notional:,.0f} ({p.short_count} positions) |\n"
        f"| Long/Short ratio | "
        f"{'∞' if p.long_short_ratio == float('inf') else f'{p.long_short_ratio:.2f}x'} |\n"
        f"| Cash | ${p.cash or 0.0:,.0f} ({p.cash_pct:.1%}) |\n"
        f"| Unrealized P&L | ${p.total_pnl_dollars:+,.0f} ({p.total_pnl_pct:+.2%}) |"
    )


def _performance_section(h: dict) -> str:
    def _fmt(key: str) -> str:
        v = h.get(key)
        return "—" if v is None else f"{v:+.2%}"

    return (
        "## Performance\n\n"
        "| Period | Return |\n"
        "|---|---|\n"
        f"| Quarter-to-date | {_fmt('qtd')} |\n"
        f"| Year-to-date | {_fmt('ytd')} |\n"
        f"| One year | {_fmt('one_year')} |\n"
        f"| Inception-to-date | {_fmt('itd')} |"
    )


def _risk_section(r: dict) -> str:
    def _fmt_pct(key: str) -> str:
        v = r.get(key)
        return "—" if v is None else f"{v:.2%}"

    def _fmt_num(key: str) -> str:
        v = r.get(key)
        return "—" if v is None else f"{v:.2f}"

    return (
        "## Risk Metrics\n\n"
        "| Metric | Value |\n"
        "|---|---|\n"
        f"| Annualized volatility | {_fmt_pct('vol_ann')} |\n"
        f"| Beta to benchmark | {_fmt_num('beta')} |\n"
        f"| Sharpe ratio | {_fmt_num('sharpe')} |\n"
        f"| Max drawdown | {_fmt_pct('max_drawdown')} |"
    )


def _sector_section(p: Portfolio) -> str:
    sectors = p.sector_exposure()
    if not sectors:
        return "## Sector Exposure\n\n_No positions._"
    rows = sorted(sectors.items(), key=lambda kv: abs(kv[1]["net"]), reverse=True)
    lines = [
        "## Sector Exposure",
        "",
        "| Sector | Long | Short | Net |",
        "|---|---|---|---|",
    ]
    for sector, exp in rows:
        lines.append(
            f"| {sector} | {exp['long']:.1%} | {exp['short']:.1%} | {exp['net']:+.1%} |"
        )
    return "\n".join(lines)


def _holdings_section(p: Portfolio, top_n: int) -> str:
    longs = sorted(p.long_positions, key=lambda x: x.notional, reverse=True)[:top_n]
    shorts = sorted(p.short_positions, key=lambda x: x.notional, reverse=True)[:top_n]

    def _row(pos) -> str:
        wt = pos.abs_weight_in(p.nav)
        return (
            f"| {pos.ticker} | {pos.sector or '—'} | "
            f"{pos.shares:,.0f} | ${pos.current_price:,.2f} | "
            f"${pos.notional:,.0f} | {wt:.2%} | {pos.pnl_pct:+.2%} |"
        )

    parts = [f"## Top Holdings (Top {top_n})\n"]
    parts.append("### Longs\n")
    if longs:
        parts.append("| Ticker | Sector | Shares | Price | Notional | Weight | Unrealized % |")
        parts.append("|---|---|---|---|---|---|---|")
        parts.extend(_row(x) for x in longs)
    else:
        parts.append("_No long positions._")
    parts.append("\n### Shorts\n")
    if shorts:
        parts.append("| Ticker | Sector | Shares | Price | Notional | Weight | Unrealized % |")
        parts.append("|---|---|---|---|---|---|---|")
        parts.extend(_row(x) for x in shorts)
    else:
        parts.append("_No short positions._")
    return "\n".join(parts)


def _disclosure_section() -> str:
    return (
        "## Disclosures\n\n"
        "Past performance is not indicative of future results. Portfolio holdings "
        "and exposures shown reflect the date stated above and may change without "
        "notice. Short positions carry unlimited loss potential. This report is "
        "informational and does not constitute investment advice."
    )
