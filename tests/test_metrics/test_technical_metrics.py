"""Tests for core/metrics/technical_metrics.py"""

import numpy as np
import polars as pl
import pytest

from core.metrics.technical_metrics import (
    high_low_52w,
    momentum,
    position_technical_summary,
    price_vs_sma,
    rsi,
    rsi_classification,
    rsi_current,
    sma,
)


def _trending_up_prices(n: int = 100) -> pl.Series:
    """Generate a trending up price series."""
    return pl.Series("price", [100.0 + i * 0.5 + np.sin(i / 5) for i in range(n)])


def _trending_down_prices(n: int = 100) -> pl.Series:
    """Generate a trending down price series."""
    return pl.Series("price", [200.0 - i * 0.5 + np.sin(i / 5) for i in range(n)])


class TestRSI:
    def test_rsi_length(self):
        prices = _trending_up_prices(50)
        r = rsi(prices, period=14)
        assert r.len() == 50

    def test_rsi_trending_up_is_high(self):
        prices = _trending_up_prices(100)
        r = rsi_current(prices)
        assert r > 50  # should be bullish/overbought

    def test_rsi_trending_down_is_low(self):
        prices = _trending_down_prices(100)
        r = rsi_current(prices)
        assert r < 50  # should be bearish/oversold

    def test_rsi_bounded(self):
        """RSI should always be between 0 and 100."""
        rng = np.random.default_rng(42)
        prices = pl.Series(100 + np.cumsum(rng.normal(0, 2, 200)))
        r = rsi(prices, period=14)
        vals = r.drop_nulls()
        assert all(0 <= v <= 100 for v in vals.to_list())


class TestRSIClassification:
    def test_overbought(self):
        assert rsi_classification(75) == "OVERBOUGHT"

    def test_oversold(self):
        assert rsi_classification(25) == "OVERSOLD"

    def test_neutral(self):
        assert rsi_classification(50) == "NEUTRAL"

    def test_bullish(self):
        assert rsi_classification(65) == "BULLISH"

    def test_bearish(self):
        assert rsi_classification(35) == "BEARISH"


class TestSMA:
    def test_sma_value(self):
        prices = pl.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        s = sma(prices, period=3)
        assert s[-1] == pytest.approx(40.0)  # (30+40+50)/3


class TestPriceVsSMA:
    def test_above_sma(self):
        prices = _trending_up_prices(100)
        pct = price_vs_sma(prices, 20)
        assert pct > 0

    def test_insufficient_data(self):
        prices = pl.Series([100.0, 101.0])
        assert price_vs_sma(prices, 50) == 0.0


class TestHighLow52W:
    def test_at_high(self):
        prices = _trending_up_prices(253)
        hl = high_low_52w(prices)
        assert hl["pct_from_high"] == pytest.approx(0.0, abs=1.0)
        assert hl["pct_from_low"] > 0

    def test_at_low(self):
        prices = _trending_down_prices(253)
        hl = high_low_52w(prices)
        assert hl["pct_from_low"] == pytest.approx(0.0, abs=1.0)
        assert hl["pct_from_high"] < 0


class TestMomentum:
    def test_positive_momentum(self):
        prices = _trending_up_prices(50)
        m = momentum(prices, 21)
        assert m > 0

    def test_negative_momentum(self):
        prices = _trending_down_prices(50)
        m = momentum(prices, 21)
        assert m < 0

    def test_insufficient_data(self):
        prices = pl.Series([100.0, 101.0])
        assert momentum(prices, 21) == 0.0


class TestTechnicalSummary:
    def test_summary_keys(self):
        prices = _trending_up_prices(300)
        summary = position_technical_summary(prices)
        assert "rsi" in summary
        assert "rsi_zone" in summary
        assert "vs_sma_50" in summary
        assert "high_52w" in summary
        assert "momentum_3m" in summary
