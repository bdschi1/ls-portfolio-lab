"""Smoke test for the Risk Analytics dashboard page — import + helper-only checks.

Doesn't render Streamlit (no AppContext available in pytest). Verifies the
page imports cleanly, factor-key rename works, sector-returns extraction works,
and the underlying metric calls execute on a synthetic portfolio + returns.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from app.pages.risk_analytics import (
    _SECTOR_ETFS,
    _SPY_SECTOR_BENCHMARK,
    _build_factor_returns,
    _build_sector_returns,
)
from core.metrics.factor_attribution import factor_pnl_attribution
from core.metrics.risk_contributions import per_name_risk_contributions
from core.metrics.style_tilts import style_tilts
from core.metrics.variance_decomp import variance_decomposition
from core.portfolio import Portfolio, Position


def _synthetic_returns_df(n: int = 120, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 1)] * n  # placeholder dates — value isn't used in regressions
    tickers = ["AAPL", "MSFT", "GOOG", "SPY", "IWM", "IWD", "IWF", "MTUM", "XLK"]
    cols = {"date": dates}
    for t in tickers:
        cols[t] = rng.normal(0.0005, 0.012, n)
    return pl.DataFrame(cols)


def _synthetic_portfolio() -> Portfolio:
    positions = [
        Position(
            ticker="AAPL",
            side="LONG",
            shares=100,
            entry_price=180,
            entry_date=date(2024, 1, 1),
            sector="Technology",
        ),
        Position(
            ticker="MSFT",
            side="LONG",
            shares=50,
            entry_price=400,
            entry_date=date(2024, 1, 1),
            sector="Technology",
        ),
        Position(
            ticker="GOOG",
            side="SHORT",
            shares=30,
            entry_price=140,
            entry_date=date(2024, 1, 1),
            sector="Communication Services",
        ),
    ]
    return Portfolio(name="smoke", positions=positions, nav=10_000_000.0)


def test_constants_well_formed():
    assert "Technology" in _SECTOR_ETFS
    assert _SECTOR_ETFS["Technology"] == "XLK"
    assert pytest.approx(sum(_SPY_SECTOR_BENCHMARK.values()), abs=0.02) == 1.00


def test_build_factor_returns_keys():
    df = _synthetic_returns_df()
    f = _build_factor_returns(df)
    # build_proxy_factors emits lowercase: market, smb, hml, momentum
    assert set(f.keys()) <= {"market", "smb", "hml", "momentum"}
    assert "market" in f  # SPY is present in fixture
    assert "smb" in f


def test_build_sector_returns_picks_sector_etfs():
    df = _synthetic_returns_df()
    s = _build_sector_returns(df)
    # XLK is in fixture, others aren't → only Technology should land
    assert "Technology" in s
    assert "Financials" not in s


def test_metric_modules_callable_end_to_end():
    df = _synthetic_returns_df()
    pf = _synthetic_portfolio()
    f = _build_factor_returns(df)
    s = _build_sector_returns(df) or None

    decomp = variance_decomposition(pf, df, f, sector_returns=s)
    assert decomp.total_var >= 0
    assert 0.0 <= decomp.idio_pct <= 1.0

    tilts = style_tilts(pf, df, f, benchmark_weights=_SPY_SECTOR_BENCHMARK)
    assert tilts.style  # one entry per factor
    assert tilts.sectors  # at least one sector

    contribs = per_name_risk_contributions(pf, df, f)
    assert len(contribs) == 3
    assert all(c.total_vol_ann >= 0 for c in contribs)

    attr = factor_pnl_attribution(pf, df, f, sector_returns=s, horizon="1D")
    assert isinstance(attr.total_pnl, float)
    assert len(attr.name_contribs) == 3
