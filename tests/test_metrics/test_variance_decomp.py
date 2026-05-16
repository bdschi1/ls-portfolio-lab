"""Tests for core/metrics/variance_decomp.py"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.metrics.variance_decomp import (
    VarianceDecomposition,
    variance_decomposition,
)
from core.portfolio import Portfolio, Position

N_DAYS = 252
SEED = 42


def _make_factor_returns(n: int = N_DAYS, seed: int = SEED) -> dict[str, pl.Series]:
    """Three independent style factors at realistic vols."""
    rng = np.random.default_rng(seed)
    return {
        "MKT": pl.Series("MKT", rng.normal(0.0005, 0.01, n)),
        "SMB": pl.Series("SMB", rng.normal(0.0, 0.005, n)),
        "HML": pl.Series("HML", rng.normal(0.0, 0.005, n)),
    }


def _build_returns_df(
    factor_returns: dict[str, pl.Series],
    name_betas: dict[str, dict[str, float]],
    idio_vol: float = 0.01,
    seed: int = SEED + 1,
) -> pl.DataFrame:
    """Synthesize per-name returns as a linear combo of factors + idio noise.

    name_betas: {ticker: {factor_name: beta}}.
    """
    rng = np.random.default_rng(seed)
    n = next(iter(factor_returns.values())).len()
    data: dict[str, np.ndarray] = {}
    for ticker, betas in name_betas.items():
        series = np.zeros(n)
        for fname, beta in betas.items():
            series = series + beta * factor_returns[fname].to_numpy()
        series = series + rng.normal(0.0, idio_vol, n)
        data[ticker] = series
    return pl.DataFrame(data)


def _three_position_portfolio() -> Portfolio:
    positions = [
        Position(
            ticker="LONGA",
            side="LONG",
            shares=1000,
            entry_price=100.0,
            current_price=100.0,
        ),
        Position(
            ticker="LONGB",
            side="LONG",
            shares=500,
            entry_price=200.0,
            current_price=200.0,
        ),
        Position(
            ticker="SHORTC",
            side="SHORT",
            shares=400,
            entry_price=150.0,
            current_price=150.0,
        ),
    ]
    # Long gross = 100k + 100k = 200k; short gross = 60k
    return Portfolio(name="test", positions=positions, nav=1_000_000.0)


class TestSharesSumToOne:
    def test_pct_components_sum_to_one(self) -> None:
        factor_returns = _make_factor_returns()
        portfolio = _three_position_portfolio()
        name_betas = {
            "LONGA": {"MKT": 1.1, "SMB": 0.3, "HML": -0.2},
            "LONGB": {"MKT": 0.9, "SMB": -0.1, "HML": 0.4},
            "SHORTC": {"MKT": 1.0, "SMB": 0.5, "HML": 0.0},
        }
        returns_df = _build_returns_df(factor_returns, name_betas)

        result = variance_decomposition(portfolio, returns_df, factor_returns)
        total_pct = (
            result.market_pct + sum(result.style_pct.values()) + result.sector_pct + result.idio_pct
        )
        assert total_pct == pytest.approx(1.0, abs=1e-6)


class TestZeroBetaIdioDominant:
    def test_zero_factor_betas_implies_idio_one(self) -> None:
        """If names have zero beta to every factor, ~100% of variance is idio."""
        factor_returns = _make_factor_returns()
        portfolio = _three_position_portfolio()
        # All betas zero — returns are pure idio noise
        name_betas = {
            "LONGA": {"MKT": 0.0, "SMB": 0.0, "HML": 0.0},
            "LONGB": {"MKT": 0.0, "SMB": 0.0, "HML": 0.0},
            "SHORTC": {"MKT": 0.0, "SMB": 0.0, "HML": 0.0},
        }
        returns_df = _build_returns_df(factor_returns, name_betas, idio_vol=0.02)

        result = variance_decomposition(portfolio, returns_df, factor_returns)
        # Regression will pick up trivial spurious betas, so allow some slack
        assert result.idio_pct > 0.9


class TestPureMarketPortfolio:
    def test_single_market_position_market_pct_dominant(self) -> None:
        """One long position synthesized as pure market exposure -> market_pct dominates."""
        factor_returns = _make_factor_returns()
        positions = [
            Position(
                ticker="SPY",
                side="LONG",
                shares=1000,
                entry_price=100.0,
                current_price=100.0,
            )
        ]
        portfolio = Portfolio(name="solo", positions=positions, nav=1_000_000.0)
        # Pure market beta of 1.0, tiny idio
        name_betas = {"SPY": {"MKT": 1.0, "SMB": 0.0, "HML": 0.0}}
        returns_df = _build_returns_df(factor_returns, name_betas, idio_vol=0.001)

        result = variance_decomposition(portfolio, returns_df, factor_returns)
        assert result.market_pct > 0.85
        # Style contribution should be near zero
        for s_pct in result.style_pct.values():
            assert s_pct < 0.05


class TestStyleFactorPassthrough:
    def test_synthetic_style_proportions(self) -> None:
        """Build returns as 0.5*MKT + 0.3*SMB + tiny noise; check shares roughly match."""
        factor_returns = _make_factor_returns()
        positions = [
            Position(
                ticker="STYLY",
                side="LONG",
                shares=1000,
                entry_price=100.0,
                current_price=100.0,
            )
        ]
        portfolio = Portfolio(name="style", positions=positions, nav=1_000_000.0)
        # Single position, beta_mkt=0.5, beta_smb=0.3, beta_hml=0, near-zero idio
        name_betas = {"STYLY": {"MKT": 0.5, "SMB": 0.3, "HML": 0.0}}
        returns_df = _build_returns_df(factor_returns, name_betas, idio_vol=1e-5)

        result = variance_decomposition(portfolio, returns_df, factor_returns)

        # Expected variance contributions from analytical betas:
        # MKT: 0.5^2 * var(MKT) * 252; SMB: 0.3^2 * var(SMB) * 252
        var_mkt = float(np.var(factor_returns["MKT"].to_numpy(), ddof=1))
        var_smb = float(np.var(factor_returns["SMB"].to_numpy(), ddof=1))
        expected_market = 0.5**2 * var_mkt
        expected_smb = 0.3**2 * var_smb
        expected_total = expected_market + expected_smb
        expected_market_share = expected_market / expected_total
        expected_smb_share = expected_smb / expected_total

        # Allow generous tolerance: factor are noisy, OLS will not recover the
        # exact betas; the *ratios* should be close though.
        assert result.market_pct == pytest.approx(expected_market_share, abs=0.10)
        assert result.style_pct["SMB"] == pytest.approx(expected_smb_share, abs=0.10)
        # HML beta is zero -> HML share should be tiny
        assert result.style_pct["HML"] < 0.05


class TestNoSectorReturns:
    def test_sector_var_zero_when_no_sector_returns(self) -> None:
        factor_returns = _make_factor_returns()
        portfolio = _three_position_portfolio()
        name_betas = {
            "LONGA": {"MKT": 1.0, "SMB": 0.2, "HML": 0.1},
            "LONGB": {"MKT": 0.8, "SMB": -0.1, "HML": 0.3},
            "SHORTC": {"MKT": 1.2, "SMB": 0.4, "HML": -0.2},
        }
        returns_df = _build_returns_df(factor_returns, name_betas)

        result = variance_decomposition(portfolio, returns_df, factor_returns, sector_returns=None)
        assert result.sector_var == 0.0
        assert result.sector_pct == 0.0


class TestAllVariancesNonNegative:
    def test_components_non_negative(self) -> None:
        factor_returns = _make_factor_returns()
        portfolio = _three_position_portfolio()
        name_betas = {
            "LONGA": {"MKT": 1.3, "SMB": 0.5, "HML": -0.3},
            "LONGB": {"MKT": 0.7, "SMB": -0.2, "HML": 0.6},
            "SHORTC": {"MKT": 1.1, "SMB": 0.4, "HML": 0.1},
        }
        returns_df = _build_returns_df(factor_returns, name_betas)

        # Add a sector pass to exercise that codepath too
        rng = np.random.default_rng(SEED + 99)
        sector_returns = {
            "Tech": pl.Series("Tech", rng.normal(0.0, 0.008, N_DAYS)),
            "Energy": pl.Series("Energy", rng.normal(0.0, 0.012, N_DAYS)),
        }
        result = variance_decomposition(
            portfolio, returns_df, factor_returns, sector_returns=sector_returns
        )

        assert result.total_var >= 0.0
        assert result.market_var >= 0.0
        assert result.sector_var >= 0.0
        assert result.idio_var >= 0.0
        for v in result.style_var.values():
            assert v >= 0.0
        # And the percent shares
        assert result.market_pct >= 0.0
        assert result.sector_pct >= 0.0
        assert result.idio_pct >= 0.0
        for p in result.style_pct.values():
            assert p >= 0.0


class TestSectorPassChangesDecomposition:
    def test_sector_var_positive_when_supplied(self) -> None:
        """With sector ETF returns supplied, sector_var should be > 0 in general."""
        factor_returns = _make_factor_returns()
        portfolio = _three_position_portfolio()
        # Mix in a sector "exposure" by giving LONGA a residual sector tilt
        rng = np.random.default_rng(SEED + 7)
        sector_tech = rng.normal(0.0, 0.01, N_DAYS)
        sector_energy = rng.normal(0.0, 0.012, N_DAYS)

        # Build per-name returns: pure factors + idio
        name_betas = {
            "LONGA": {"MKT": 1.0, "SMB": 0.0, "HML": 0.0},
            "LONGB": {"MKT": 0.9, "SMB": 0.1, "HML": 0.0},
            "SHORTC": {"MKT": 1.1, "SMB": 0.0, "HML": 0.2},
        }
        returns_df = _build_returns_df(factor_returns, name_betas, idio_vol=0.005)
        # Inject sector tech exposure into LONGA residual
        new_longa = returns_df["LONGA"].to_numpy() + 0.7 * sector_tech
        returns_df = returns_df.with_columns(pl.Series("LONGA", new_longa))

        sector_returns = {
            "Tech": pl.Series("Tech", sector_tech),
            "Energy": pl.Series("Energy", sector_energy),
        }

        result = variance_decomposition(
            portfolio, returns_df, factor_returns, sector_returns=sector_returns
        )
        assert isinstance(result, VarianceDecomposition)
        assert result.sector_var > 0.0
        # Shares still sum to 1
        total_pct = (
            result.market_pct + sum(result.style_pct.values()) + result.sector_pct + result.idio_pct
        )
        assert total_pct == pytest.approx(1.0, abs=1e-6)
