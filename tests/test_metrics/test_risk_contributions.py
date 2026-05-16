"""Tests for core/metrics/risk_contributions.py.

Synthetic 5-name fixture, 252 trading days, 4-factor model (MKT/SMB/HML/MOM).
Each test pins one property of the per-name risk decomposition.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.metrics.risk_contributions import (
    NameRiskContrib,
    per_name_risk_contributions,
    portfolio_idio_vol,
    portfolio_total_vol,
)
from core.portfolio import Portfolio, Position

N_DAYS = 252
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_factor_series(seed: int = 7) -> dict[str, pl.Series]:
    """Build a synthetic factor block: MKT/SMB/HML/MOM, all 252 days."""
    rng = np.random.default_rng(seed)
    return {
        "market": pl.Series("market", rng.normal(0.0004, 0.010, N_DAYS)),
        "smb": pl.Series("smb", rng.normal(0.0, 0.005, N_DAYS)),
        "hml": pl.Series("hml", rng.normal(0.0, 0.005, N_DAYS)),
        "momentum": pl.Series("momentum", rng.normal(0.0, 0.005, N_DAYS)),
    }


def _make_returns_df(
    factors: dict[str, pl.Series],
    spec: dict[str, dict[str, float]],
    idio_std: float = 0.012,
    seed: int = 13,
) -> pl.DataFrame:
    """Build a returns_df with one column per ticker, driven by factor loadings.

    spec: {ticker: {"mkt": float, "smb": float, "hml": float, "mom": float,
                    "alpha": float}}
    """
    rng = np.random.default_rng(seed)
    mkt = factors["market"].to_numpy()
    smb = factors["smb"].to_numpy()
    hml = factors["hml"].to_numpy()
    mom = factors["momentum"].to_numpy()
    cols: dict[str, np.ndarray] = {}
    for ticker, loadings in spec.items():
        idio = rng.normal(0.0, idio_std, N_DAYS)
        rets = (
            loadings.get("alpha", 0.0)
            + loadings.get("mkt", 1.0) * mkt
            + loadings.get("smb", 0.0) * smb
            + loadings.get("hml", 0.0) * hml
            + loadings.get("mom", 0.0) * mom
            + idio
        )
        cols[ticker] = rets
    return pl.DataFrame(cols)


def _build_portfolio(
    sides_shares_prices: list[tuple[str, str, float, float]],
    nav: float = 100_000_000.0,
) -> Portfolio:
    """Helper: build a Portfolio from [(ticker, side, shares, price), ...]."""
    positions = [
        Position(
            ticker=t,
            side=side,
            shares=shares,
            entry_price=price,
            current_price=price,
        )
        for (t, side, shares, price) in sides_shares_prices
    ]
    return Portfolio(name="test", positions=positions, nav=nav)


@pytest.fixture
def factors() -> dict[str, pl.Series]:
    return _make_factor_series()


@pytest.fixture
def returns_df(factors: dict[str, pl.Series]) -> pl.DataFrame:
    spec = {
        "AAA": {"mkt": 1.2, "smb": 0.2, "hml": -0.1, "mom": 0.3, "alpha": 0.0006},
        "BBB": {"mkt": 0.9, "smb": -0.1, "hml": 0.4, "mom": 0.0, "alpha": 0.0002},
        "CCC": {"mkt": 1.1, "smb": 0.5, "hml": 0.1, "mom": -0.2, "alpha": -0.0001},
        "DDD": {"mkt": 0.8, "smb": -0.3, "hml": 0.0, "mom": 0.5, "alpha": 0.0004},
        "EEE": {"mkt": 1.0, "smb": 0.0, "hml": -0.2, "mom": 0.1, "alpha": 0.0000},
    }
    return _make_returns_df(factors, spec)


@pytest.fixture
def portfolio() -> Portfolio:
    # Two longs, two shorts, one extra long. Mixed sizes so weights are distinct.
    return _build_portfolio(
        [
            ("AAA", "LONG", 100_000, 100.0),  # 10MM long
            ("BBB", "LONG", 80_000, 150.0),  # 12MM long
            ("CCC", "SHORT", 60_000, 200.0),  # 12MM short
            ("DDD", "SHORT", 40_000, 250.0),  # 10MM short
            ("EEE", "LONG", 50_000, 100.0),  # 5MM long
        ],
        nav=100_000_000.0,
    )


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


def test_cctr_total_sums_to_portfolio_total_vol(
    portfolio: Portfolio,
    returns_df: pl.DataFrame,
    factors: dict[str, pl.Series],
) -> None:
    """Euler decomposition: sum_i CCTR_total_i = portfolio total vol."""
    contribs = per_name_risk_contributions(portfolio, returns_df, factors)
    sum_cctr = sum(c.cctr_total for c in contribs)
    port_vol = portfolio_total_vol(portfolio, returns_df)
    assert sum_cctr == pytest.approx(port_vol, abs=1e-6)


def test_cctr_idio_sums_to_portfolio_idio_vol_at_name_level(
    portfolio: Portfolio,
    returns_df: pl.DataFrame,
    factors: dict[str, pl.Series],
) -> None:
    """Sum of per-name idio CCTRs equals the name-level idio portfolio vol.

    NOTE: this is the vol from the per-name-residualised covariance, which is
    the right object for the per-name attribution (sum_i CCTR_idio_i = sigma
    from that cov). It is *not* the same as portfolio_idio_vol(...), which
    regresses the aggregate portfolio return on factors — those two views
    coincide only when name residuals are uncorrelated, which is the standard
    factor-model assumption but not guaranteed in finite samples.
    """
    contribs = per_name_risk_contributions(portfolio, returns_df, factors)
    sum_cctr_idio = sum(c.cctr_idio for c in contribs)

    # Reconstruct the name-level idio portfolio vol the same way the module
    # does internally, to verify the Euler identity on its own terms.
    tickers = [p.ticker for p in portfolio.positions]
    mat = returns_df.select(tickers).to_numpy()
    n_obs = mat.shape[0]
    fac_mat = np.column_stack([factors[k].tail(n_obs).to_numpy() for k in factors])
    x = np.column_stack([np.ones(n_obs), fac_mat])
    betas, *_ = np.linalg.lstsq(x, mat, rcond=None)
    residuals = mat - x @ betas
    cov_idio = np.cov(residuals, rowvar=False, ddof=1)
    w = np.array([p.weight_in(portfolio.nav) for p in portfolio.positions])
    port_idio_daily = float(np.sqrt(max(w @ cov_idio @ w, 0.0)))
    expected_ann = port_idio_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    assert sum_cctr_idio == pytest.approx(expected_ann, abs=1e-6)


def test_single_position_cctr_equals_portfolio_vol(
    factors: dict[str, pl.Series],
) -> None:
    """Single-position portfolio: that name's CCTR = portfolio vol."""
    spec = {"AAA": {"mkt": 1.1, "smb": 0.0, "hml": 0.0, "mom": 0.0, "alpha": 0.0003}}
    rdf = _make_returns_df(factors, spec)
    port = _build_portfolio([("AAA", "LONG", 100_000, 100.0)], nav=20_000_000.0)
    contribs = per_name_risk_contributions(port, rdf, factors)
    assert len(contribs) == 1
    port_vol = portfolio_total_vol(port, rdf)
    assert contribs[0].cctr_total == pytest.approx(port_vol, abs=1e-9)


def test_zero_weight_position_has_zero_cctr(factors: dict[str, pl.Series]) -> None:
    """A position whose weight is ~0 (tiny shares) has CCTR ~ 0."""
    spec = {
        "AAA": {"mkt": 1.0, "smb": 0.0, "hml": 0.0, "mom": 0.0, "alpha": 0.0},
        "BBB": {"mkt": 1.0, "smb": 0.0, "hml": 0.0, "mom": 0.0, "alpha": 0.0},
    }
    rdf = _make_returns_df(factors, spec)
    # BBB sized to ~0% of NAV (1 share at $0.01 → $0.01 vs $100MM NAV).
    port = _build_portfolio(
        [
            ("AAA", "LONG", 100_000, 100.0),
            ("BBB", "LONG", 1.0, 0.01),
        ],
        nav=100_000_000.0,
    )
    contribs = per_name_risk_contributions(port, rdf, factors)
    bbb = next(c for c in contribs if c.ticker == "BBB")
    assert bbb.weight == pytest.approx(0.0, abs=1e-9)
    assert bbb.cctr_total == pytest.approx(0.0, abs=1e-9)
    assert bbb.cctr_idio == pytest.approx(0.0, abs=1e-9)


def test_alpha_over_idio_recovers_known_ratio(factors: dict[str, pl.Series]) -> None:
    """Synthetic returns: alpha_over_idio = (realised_mean*252) / (realised_idio_vol_ann).

    We compare against *realised* sample stats (not the population mean / std
    used to generate the series) — at N=252, the sample mean has standard
    error ~ std/sqrt(N) which is comparable to the mean itself for typical
    daily returns. So checking population mean recovery would need either a
    much larger sample or much wider tolerance; checking realised-mean
    recovery pins the formula directly.
    """
    rng = np.random.default_rng(99)
    # Pure idio noise — zero factor loadings — so idio_vol_ann ≈ realised total vol.
    rets = rng.normal(0.001, 0.02, N_DAYS)
    rdf = pl.DataFrame({"ZZZ": rets})
    port = _build_portfolio([("ZZZ", "LONG", 1_000, 100.0)], nav=10_000_000.0)

    contribs = per_name_risk_contributions(port, rdf, factors)
    c = contribs[0]

    # Realised stats — what the module's estimators actually see.
    realised_mean = float(np.mean(rets))
    expected_alpha = realised_mean * TRADING_DAYS_PER_YEAR
    # Idio vol comes from residuals after factor regression, not raw stdev.
    # With factors uncorrelated with ZZZ, residual std ≈ raw std but not exact.
    # The CCTR-style identity we lean on: alpha_over_idio * idio_vol == alpha.
    assert c.alpha_estimate == pytest.approx(expected_alpha, rel=1e-9)
    assert c.alpha_over_idio * c.idio_vol_ann == pytest.approx(c.alpha_estimate, rel=1e-9)
    # Ratio should be in the right ballpark vs raw-stdev-based estimate (loose).
    raw_idio_vol = float(np.std(rets, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    rough_expected_ratio = expected_alpha / raw_idio_vol
    assert c.alpha_over_idio == pytest.approx(rough_expected_ratio, rel=0.10)


def test_alpha_over_idio_zero_when_idio_vol_zero(factors: dict[str, pl.Series]) -> None:
    """Divide-by-zero policy: idio_vol == 0 → alpha_over_idio == 0.0 (not NaN)."""
    # Construct a return series exactly equal to the market factor — after
    # regression on the factor block, residuals are 0, so idio_vol_ann == 0.
    mkt = factors["market"].to_numpy()
    rdf = pl.DataFrame({"MKT_CLONE": mkt})
    port = _build_portfolio([("MKT_CLONE", "LONG", 1_000, 100.0)], nav=10_000_000.0)
    contribs = per_name_risk_contributions(port, rdf, factors)
    c = contribs[0]
    assert c.idio_vol_ann == pytest.approx(0.0, abs=1e-9)
    assert c.alpha_over_idio == 0.0  # exact, not NaN


def test_portfolio_total_vol_matches_weighted_return_stdev(
    portfolio: Portfolio,
    returns_df: pl.DataFrame,
) -> None:
    """portfolio_total_vol == annualised stdev of w·r daily series."""
    weights = portfolio.weight_vector()
    tickers = list(weights.keys())
    mat = returns_df.select(tickers).to_numpy()
    w = np.array([weights[t] for t in tickers])
    port_rets = mat @ w
    expected = float(np.std(port_rets, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    actual = portfolio_total_vol(portfolio, returns_df)
    # Allow a tiny gap: portfolio_total_vol goes through cov-based sqrt(w'Σw),
    # which equals stdev(w·r) up to numerical precision (cov uses mean-centring
    # per column vs stdev mean-centring the linear combo — algebraically equal).
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_idio_vol_below_total_vol_when_factor_loadings_nonzero(
    portfolio: Portfolio,
    returns_df: pl.DataFrame,
    factors: dict[str, pl.Series],
) -> None:
    """Removing factor exposure can only shrink vol (or hold it constant)."""
    total = portfolio_total_vol(portfolio, returns_df)
    idio = portfolio_idio_vol(portfolio, returns_df, factors)
    assert idio < total
    # And both should be positive for this fixture.
    assert total > 0
    assert idio > 0


# ---------------------------------------------------------------------------
# Smoke / structure tests (above the 7-test bar but cheap to keep)
# ---------------------------------------------------------------------------


def test_per_name_contribs_have_expected_structure(
    portfolio: Portfolio,
    returns_df: pl.DataFrame,
    factors: dict[str, pl.Series],
) -> None:
    """Sanity: 5 positions in, 5 NameRiskContrib out, dollar_vol_contrib non-negative."""
    contribs = per_name_risk_contributions(portfolio, returns_df, factors)
    assert len(contribs) == 5
    assert all(isinstance(c, NameRiskContrib) for c in contribs)
    assert all(c.dollar_vol_contrib >= 0.0 for c in contribs)
    # Each name has signed notional matching side direction.
    for c in contribs:
        if c.side == "LONG":
            assert c.notional > 0
            assert c.weight > 0
        else:
            assert c.notional < 0
            assert c.weight < 0
