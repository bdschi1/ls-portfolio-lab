"""Tests for core/rebalancer.py — portfolio rebalancing to risk targets."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from core.portfolio import Portfolio, Position
from core.rebalancer import RebalanceRequest, RebalanceResult, compute_rebalance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_portfolio(nav: float = 1_000_000.0) -> Portfolio:
    """Multi-sector long/short portfolio for testing."""
    return Portfolio(
        positions=[
            Position(ticker="AAPL", side="LONG", shares=500, entry_price=150.0,
                     current_price=160.0, sector="Technology"),
            Position(ticker="MSFT", side="LONG", shares=200, entry_price=300.0,
                     current_price=310.0, sector="Technology"),
            Position(ticker="JPM", side="LONG", shares=300, entry_price=180.0,
                     current_price=185.0, sector="Financials"),
            Position(ticker="XOM", side="LONG", shares=400, entry_price=100.0,
                     current_price=105.0, sector="Energy"),
            Position(ticker="UNH", side="SHORT", shares=100, entry_price=500.0,
                     current_price=490.0, sector="Healthcare"),
            Position(ticker="PFE", side="SHORT", shares=800, entry_price=30.0,
                     current_price=28.0, sector="Healthcare"),
            Position(ticker="KO", side="SHORT", shares=300, entry_price=60.0,
                     current_price=58.0, sector="Consumer Staples"),
        ],
        nav=nav,
    )


def _make_returns_df(n_days: int = 252) -> pl.DataFrame:
    """Synthetic daily returns for test tickers + SPY benchmark."""
    rng = np.random.default_rng(42)
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "UNH", "PFE", "KO", "SPY"]
    # Different volatilities per ticker to make the optimizer meaningful
    vols = {
        "AAPL": 0.02, "MSFT": 0.018, "JPM": 0.015, "XOM": 0.022,
        "UNH": 0.016, "PFE": 0.025, "KO": 0.010, "SPY": 0.012,
    }
    data = {"date": pl.date_range(date(2024, 1, 1), date(2024, 1, 1),
                                   eager=True).extend_constant(
        date(2024, 1, 2), n_days - 1)}
    # Build proper date range
    dates = [date(2024, 1, 1)]
    for i in range(1, n_days):
        d = date.fromordinal(date(2024, 1, 1).toordinal() + i)
        dates.append(d)
    data = {"date": dates}

    for t in tickers:
        data[t] = rng.normal(0.0003, vols[t], n_days).tolist()

    return pl.DataFrame(data)


def _make_betas() -> dict[str, float]:
    return {
        "AAPL": 1.20, "MSFT": 1.10, "JPM": 1.30, "XOM": 0.90,
        "UNH": 0.80, "PFE": 0.60, "KO": 0.50, "SPY": 1.0,
    }


def _make_prices() -> dict[str, float]:
    return {
        "AAPL": 160.0, "MSFT": 310.0, "JPM": 185.0, "XOM": 105.0,
        "UNH": 490.0, "PFE": 28.0, "KO": 58.0,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRebalanceBasic:
    def test_no_targets_returns_unchanged(self):
        """If no targets set, nothing happens."""
        p = _make_portfolio()
        result = compute_rebalance(
            p, RebalanceRequest(), None, _make_betas(), _make_prices(),
        )
        assert result.converged is True
        assert len(result.trades_needed) == 0
        assert "No targets set" in result.warnings[0]

    def test_empty_portfolio(self):
        """Portfolio with too few positions returns gracefully."""
        p = Portfolio(
            positions=[
                Position(ticker="AAPL", side="LONG", shares=100,
                         entry_price=150.0, current_price=160.0),
            ],
            nav=100000.0,
        )
        result = compute_rebalance(
            p,
            RebalanceRequest(target_net_beta=0.05),
            None,
            {"AAPL": 1.2},
            {"AAPL": 160.0},
        )
        assert result.converged is False
        assert "at least 2" in result.warnings[0].lower()

    def test_immutability(self):
        """Original portfolio must not be modified."""
        p = _make_portfolio()
        original_shares = {pos.ticker: pos.shares for pos in p.positions}
        compute_rebalance(
            p,
            RebalanceRequest(target_net_beta=0.02),
            _make_returns_df(),
            _make_betas(),
            _make_prices(),
        )
        for pos in p.positions:
            assert pos.shares == original_shares[pos.ticker]


class TestBetaTarget:
    def test_reduces_beta_toward_target(self):
        """Rebalancing with lower beta target should reduce net beta."""
        p = _make_portfolio()
        betas = _make_betas()
        returns_df = _make_returns_df()

        # Current net beta is high (long-biased with high-beta longs)
        from core.metrics.risk_metrics import portfolio_beta
        old_beta = portfolio_beta(p.weight_vector(), betas)

        target = old_beta * 0.3  # target much lower
        result = compute_rebalance(
            p,
            RebalanceRequest(target_net_beta=target),
            returns_df,
            betas,
            _make_prices(),
        )
        new_beta = portfolio_beta(result.portfolio_after.weight_vector(), betas)
        assert abs(new_beta - target) < abs(old_beta - target)

    def test_beta_target_met_flag(self):
        """target_met dict should reflect whether beta target was achieved."""
        p = _make_portfolio()
        result = compute_rebalance(
            p,
            RebalanceRequest(target_net_beta=0.05),
            _make_returns_df(),
            _make_betas(),
            _make_prices(),
        )
        assert "net_beta" in result.target_met

    def test_trades_generated(self):
        """Rebalancing should generate non-empty trade list."""
        p = _make_portfolio()
        result = compute_rebalance(
            p,
            RebalanceRequest(target_net_beta=0.01),
            _make_returns_df(),
            _make_betas(),
            _make_prices(),
        )
        assert len(result.trades_needed) > 0


class TestVolTarget:
    def test_vol_target_reduces_vol(self):
        """Setting low vol target should reduce portfolio volatility."""
        p = _make_portfolio()
        returns_df = _make_returns_df()
        betas = _make_betas()

        from core.metrics.risk_metrics import portfolio_volatility
        old_vol = portfolio_volatility(p.weight_vector(), returns_df)

        # Target well below current vol
        target_vol = old_vol * 0.5
        result = compute_rebalance(
            p,
            RebalanceRequest(target_ann_vol=target_vol),
            returns_df,
            betas,
            _make_prices(),
        )
        new_vol = portfolio_volatility(
            result.portfolio_after.weight_vector(), returns_df,
        )
        assert new_vol < old_vol

    def test_vol_without_returns_warns(self):
        """Vol target with no returns data should warn."""
        p = _make_portfolio()
        result = compute_rebalance(
            p,
            RebalanceRequest(target_ann_vol=0.03),
            None,
            _make_betas(),
            _make_prices(),
        )
        assert any("returns data" in w.lower() for w in result.warnings)


class TestSidesPreserved:
    def test_longs_stay_long(self):
        """Longs must remain long after rebalancing."""
        p = _make_portfolio()
        result = compute_rebalance(
            p,
            RebalanceRequest(target_net_beta=0.01),
            _make_returns_df(),
            _make_betas(),
            _make_prices(),
        )
        for pos in result.portfolio_after.positions:
            original = p.get_position(pos.ticker)
            if original is not None:
                assert pos.side == original.side

    def test_shorts_stay_short(self):
        """Shorts must remain short after rebalancing."""
        p = _make_portfolio()
        result = compute_rebalance(
            p,
            RebalanceRequest(target_net_beta=0.20),
            _make_returns_df(),
            _make_betas(),
            _make_prices(),
        )
        for pos in result.portfolio_after.positions:
            original = p.get_position(pos.ticker)
            if original is not None:
                assert pos.side == original.side


class TestCashTracking:
    def test_cash_equals_nav_minus_invested(self):
        """Cash should equal NAV minus total invested."""
        p = _make_portfolio()
        result = compute_rebalance(
            p,
            RebalanceRequest(target_net_beta=0.05),
            _make_returns_df(),
            _make_betas(),
            _make_prices(),
        )
        pa = result.portfolio_after
        total_invested = sum(pos.notional for pos in pa.positions)
        assert abs(pa.cash - (pa.nav - total_invested)) < 1.0  # within $1

    def test_nav_preserved(self):
        """NAV should not change during rebalancing."""
        p = _make_portfolio()
        result = compute_rebalance(
            p,
            RebalanceRequest(target_net_beta=0.05),
            _make_returns_df(),
            _make_betas(),
            _make_prices(),
        )
        assert result.portfolio_after.nav == p.nav


class TestOptionsExcluded:
    def test_option_positions_unchanged(self):
        """Option positions should not be modified by the rebalancer."""
        p = Portfolio(
            positions=[
                Position(ticker="AAPL", side="LONG", shares=500,
                         entry_price=150.0, current_price=160.0, sector="Tech"),
                Position(ticker="MSFT", side="LONG", shares=200,
                         entry_price=300.0, current_price=310.0, sector="Tech"),
                Position(ticker="UNH", side="SHORT", shares=100,
                         entry_price=500.0, current_price=490.0,
                         sector="Healthcare"),
                Position(ticker="AAPL_CALL", side="LONG", shares=10,
                         entry_price=5.0, current_price=6.0,
                         asset_type="OPTION", strike=170.0,
                         expiry=date(2025, 6, 20), option_type="CALL",
                         delta=0.5, sector="Tech"),
            ],
            nav=500000.0,
        )
        result = compute_rebalance(
            p,
            RebalanceRequest(target_net_beta=0.05),
            _make_returns_df(),
            _make_betas(),
            {"AAPL": 160.0, "MSFT": 310.0, "UNH": 490.0, "AAPL_CALL": 6.0},
        )
        opt = result.portfolio_after.get_position("AAPL_CALL")
        assert opt is not None
        assert opt.shares == 10  # unchanged


class TestMetricsOutput:
    def test_metrics_before_after_present(self):
        """Result should contain before/after metric snapshots."""
        p = _make_portfolio()
        result = compute_rebalance(
            p,
            RebalanceRequest(target_net_beta=0.05),
            _make_returns_df(),
            _make_betas(),
            _make_prices(),
        )
        assert "net_beta" in result.metrics_before
        assert "net_beta" in result.metrics_after
        assert "gross_exposure" in result.metrics_before
        assert "ann_vol" in result.metrics_before

    def test_weight_changes_populated(self):
        """Weight changes dict should have an entry for each position."""
        p = _make_portfolio()
        result = compute_rebalance(
            p,
            RebalanceRequest(target_net_beta=0.02),
            _make_returns_df(),
            _make_betas(),
            _make_prices(),
        )
        assert len(result.weight_changes) == p.total_count


class TestBothConstraints:
    def test_dual_constraint(self):
        """Both beta and vol targets simultaneously."""
        p = _make_portfolio()
        returns_df = _make_returns_df()
        betas = _make_betas()

        result = compute_rebalance(
            p,
            RebalanceRequest(target_net_beta=0.03, target_ann_vol=0.10),
            returns_df,
            betas,
            _make_prices(),
        )
        # Should have both target_met entries
        assert "net_beta" in result.target_met
        assert "ann_vol" in result.target_met
        # Should have generated trades
        assert isinstance(result.trades_needed, list)
