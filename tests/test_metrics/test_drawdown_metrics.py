"""Tests for core/metrics/drawdown_metrics.py"""

import polars as pl
import pytest

from core.metrics.drawdown_metrics import (
    current_drawdown,
    drawdown_duration,
    drawdown_series,
    drawdown_summary,
    identify_drawdown_episodes,
    max_drawdown,
)


class TestDrawdownSeries:
    def test_always_at_peak(self):
        """Monotonically increasing → drawdown is always 0."""
        prices = pl.Series([100, 101, 102, 103, 104])
        dd = drawdown_series(prices)
        assert all(v == 0 for v in dd.to_list())

    def test_simple_drawdown(self):
        prices = pl.Series([100.0, 110.0, 100.0, 115.0])
        dd = drawdown_series(prices)
        # After 110 → 100: dd = (100-110)/110 ≈ -9.09%
        assert dd[2] == pytest.approx(-10.0 / 110.0)
        # After 115: new peak → dd = 0
        assert dd[3] == pytest.approx((115 - 115) / 115)


class TestMaxDrawdown:
    def test_no_drawdown(self):
        prices = pl.Series([100, 101, 102, 103])
        assert max_drawdown(prices) == 0.0

    def test_known_drawdown(self):
        prices = pl.Series([100.0, 120.0, 90.0, 130.0])
        # Peak 120 → trough 90: dd = (90-120)/120 = -25%
        md = max_drawdown(prices)
        assert md == pytest.approx(-0.25)

    def test_current_drawdown_at_peak(self):
        prices = pl.Series([100, 110, 120])
        assert current_drawdown(prices) == 0.0

    def test_current_drawdown_below_peak(self):
        prices = pl.Series([100.0, 120.0, 110.0])
        cd = current_drawdown(prices)
        assert cd == pytest.approx(-10.0 / 120.0)


class TestDrawdownDuration:
    def test_at_peak(self):
        prices = pl.Series([100, 110, 120])
        assert drawdown_duration(prices) == 0

    def test_in_drawdown(self):
        prices = pl.Series([100.0, 120.0, 115.0, 110.0])
        dur = drawdown_duration(prices)
        assert dur == 2  # 2 periods since last peak


class TestDrawdownEpisodes:
    def test_single_episode(self):
        prices = pl.Series([100.0, 120.0, 90.0, 130.0])
        episodes = identify_drawdown_episodes(prices, min_depth=-0.01)
        assert len(episodes) >= 1
        assert episodes[0].depth == pytest.approx(-0.25)
        assert not episodes[0].is_active  # recovered to new peak

    def test_active_drawdown(self):
        prices = pl.Series([100.0, 120.0, 90.0])
        episodes = identify_drawdown_episodes(prices, min_depth=-0.01)
        assert len(episodes) == 1
        assert episodes[0].is_active

    def test_no_episodes_if_min_depth(self):
        prices = pl.Series([100.0, 101.0, 100.5, 102.0])
        episodes = identify_drawdown_episodes(prices, min_depth=-0.02)
        assert len(episodes) == 0  # tiny drawdown filtered out


class TestDrawdownSummary:
    def test_summary_keys(self):
        prices = pl.Series([100.0, 120.0, 90.0, 130.0, 100.0])
        summary = drawdown_summary(prices)
        assert "max_drawdown" in summary
        assert "current_drawdown" in summary
        assert "num_episodes" in summary
        assert summary["num_episodes"] >= 1
