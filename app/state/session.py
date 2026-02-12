"""Streamlit session state management.

Centralizes all session state access. Avoids scattered st.session_state
references throughout the app.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

from core.portfolio import Portfolio
from data.cache import DataCache
from history.snapshot import SnapshotStore
from history.trade_log import TradeLog

CONFIG_PATH = Path("config.yaml")
DATA_DIR = Path("data_cache")
HISTORY_DIR = DATA_DIR / "history"


def _load_config() -> dict:
    """Load configuration from YAML file."""
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open() as f:
            return yaml.safe_load(f) or {}
    return {}


def init_session_state() -> None:
    """Initialize all session state on first run."""
    if "initialized" in st.session_state:
        return

    config = _load_config()

    # Portfolio
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = None  # Will be Portfolio or None

    # Data cache
    if "data_cache" not in st.session_state:
        st.session_state.data_cache = DataCache(
            cache_dir=DATA_DIR,
            price_staleness_hours=config.get("cache", {}).get("price_staleness_hours", 18),
            info_staleness_days=config.get("cache", {}).get("info_staleness_days", 7),
            max_history_years=config.get("cache", {}).get("max_history_years", 3),
        )

    # Settings
    if "settings" not in st.session_state:
        metrics_cfg = config.get("metrics", {})
        st.session_state.settings = {
            "lookback_days": metrics_cfg.get("lookback_days", 252),
            "rsi_period": metrics_cfg.get("rsi_period", 14),
            "var_confidence": metrics_cfg.get("var_confidence", 0.95),
            "risk_free_rate": metrics_cfg.get("risk_free_rate", "auto"),
            "factor_model": metrics_cfg.get("factor_model", "CAPM"),
            "benchmark": config.get("portfolio", {}).get("benchmark", "SPY"),
            "nav": config.get("portfolio", {}).get("default_nav", 3_000_000_000),
            "min_cash_pct": config.get("portfolio", {}).get("min_cash_pct", 0.05),
            "max_etf_positions": config.get("portfolio", {}).get("max_etf_positions", 3),
        }

    # Alert thresholds
    if "alerts" not in st.session_state:
        alerts_cfg = config.get("alerts", {})
        st.session_state.alerts = {
            "max_sector_net_exposure": alerts_cfg.get("max_sector_net_exposure", 0.50),
            "max_subsector_net_exposure": alerts_cfg.get("max_subsector_net_exposure", 0.50),
            "max_single_position_weight": alerts_cfg.get("max_single_position_weight", 0.10),
            "max_net_beta": alerts_cfg.get("max_net_beta", 0.30),
            "min_net_beta": alerts_cfg.get("min_net_beta", -0.10),
            "max_gross_exposure": alerts_cfg.get("max_gross_exposure", 2.50),
            "max_ann_vol": alerts_cfg.get("max_ann_vol", 0.06),
        }

    # Paper portfolio mode
    if "paper_mode" not in st.session_state:
        st.session_state.paper_mode = False

    if "trade_log" not in st.session_state:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        st.session_state.trade_log = TradeLog(HISTORY_DIR / "trade_log.jsonl")

    if "snapshot_store" not in st.session_state:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        st.session_state.snapshot_store = SnapshotStore(HISTORY_DIR / "snapshots.jsonl")

    # Cached computed data
    if "returns_df" not in st.session_state:
        st.session_state.returns_df = None  # Will be pl.DataFrame

    if "prices_df" not in st.session_state:
        st.session_state.prices_df = None  # Raw long-format prices for ADV$ etc.

    if "betas" not in st.session_state:
        st.session_state.betas = {}  # {ticker: float}

    if "risk_free_rate_value" not in st.session_state:
        st.session_state.risk_free_rate_value = 0.05

    # Target prices (user-defined per-ticker)
    if "target_prices" not in st.session_state:
        st.session_state.target_prices = {}  # {ticker: float}

    # Rebalancer state
    if "rebalance_requested" not in st.session_state:
        st.session_state.rebalance_requested = False
    if "rebalance_result" not in st.session_state:
        st.session_state.rebalance_result = None

    # Data source
    if "data_source" not in st.session_state:
        st.session_state.data_source = "Yahoo Finance"

    # UI state
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Portfolio"

    st.session_state.initialized = True


def get_portfolio() -> Portfolio | None:
    """Get the current portfolio from session state."""
    return st.session_state.get("portfolio")


def set_portfolio(portfolio: Portfolio) -> None:
    """Set the portfolio in session state."""
    st.session_state.portfolio = portfolio


def get_cache() -> DataCache:
    """Get the data cache."""
    return st.session_state.data_cache


def get_settings() -> dict:
    """Get current settings."""
    return st.session_state.settings


def is_paper_mode() -> bool:
    """Check if paper portfolio mode is enabled."""
    return st.session_state.get("paper_mode", False)


def get_trade_log() -> TradeLog:
    """Get the trade log."""
    return st.session_state.trade_log


def get_snapshot_store() -> SnapshotStore:
    """Get the snapshot store."""
    return st.session_state.snapshot_store
