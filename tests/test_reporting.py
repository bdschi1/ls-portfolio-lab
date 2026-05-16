"""Tests for core.reporting — L/S client snapshot markdown."""

from __future__ import annotations

from datetime import date

from core.portfolio import Portfolio, Position
from core.reporting import render_client_report


def _pf() -> Portfolio:
    return Portfolio(
        name="Test Book",
        nav=1_000_000.0,
        inception_date=date(2024, 1, 1),
        positions=[
            Position(
                ticker="AAPL", side="LONG", shares=100, entry_price=180,
                current_price=200, sector="Tech",
            ),
            Position(
                ticker="MSFT", side="LONG", shares=50, entry_price=400,
                current_price=420, sector="Tech",
            ),
            Position(
                ticker="GOOG", side="SHORT", shares=30, entry_price=140,
                current_price=130, sector="Comm",
            ),
        ],
    )


def test_report_contains_required_sections():
    md = render_client_report(_pf(), report_date=date(2026, 5, 1))
    for header in [
        "# Portfolio Snapshot",
        "## Exposure Summary",
        "## Sector Exposure",
        "## Top Holdings",
        "## Disclosures",
    ]:
        assert header in md, f"Missing section: {header}"


def test_report_shows_nav_and_date():
    md = render_client_report(_pf(), report_date=date(2026, 5, 1))
    assert "2026-05-01" in md
    assert "$1,000,000" in md
    assert "Test Book" in md


def test_report_includes_long_and_short_holdings():
    md = render_client_report(_pf(), report_date=date(2026, 5, 1))
    assert "AAPL" in md
    assert "MSFT" in md
    assert "GOOG" in md
    assert "### Longs" in md
    assert "### Shorts" in md


def test_performance_section_only_when_history_passed():
    md_no_hist = render_client_report(_pf(), report_date=date(2026, 5, 1))
    assert "## Performance" not in md_no_hist
    md_with_hist = render_client_report(
        _pf(),
        report_date=date(2026, 5, 1),
        history_summary={"qtd": 0.012, "ytd": 0.085, "one_year": 0.15, "itd": 0.30},
    )
    assert "## Performance" in md_with_hist
    assert "+8.50%" in md_with_hist  # YTD


def test_risk_section_only_when_risk_passed():
    md = render_client_report(
        _pf(),
        report_date=date(2026, 5, 1),
        risk_summary={"vol_ann": 0.12, "beta": 0.45, "sharpe": 1.2, "max_drawdown": -0.08},
    )
    assert "## Risk Metrics" in md
    assert "12.00%" in md
    assert "1.20" in md


def test_missing_history_keys_render_as_dash():
    md = render_client_report(
        _pf(),
        report_date=date(2026, 5, 1),
        history_summary={"ytd": 0.05},
    )
    # QTD, one-year, ITD all missing → "—"
    assert md.count("—") >= 3


def test_top_n_limits_holdings_table():
    # Build a portfolio with many longs
    positions = [
        Position(
            ticker=f"L{i:02d}", side="LONG", shares=10, entry_price=100,
            current_price=100, sector="Tech",
        )
        for i in range(15)
    ]
    pf = Portfolio(name="Many", nav=10_000_000.0, positions=positions)
    md = render_client_report(pf, report_date=date(2026, 5, 1), top_n=5)
    assert "L00" in md and "L04" in md
    assert "L14" not in md  # top 5 only


def test_empty_portfolio_does_not_crash():
    pf = Portfolio(name="Empty", nav=1_000_000.0, positions=[])
    md = render_client_report(pf, report_date=date(2026, 5, 1))
    assert "No long positions" in md
    assert "No short positions" in md
