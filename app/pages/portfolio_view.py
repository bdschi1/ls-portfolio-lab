"""Portfolio View — Main dashboard page.

Left side: portfolio table with all positions
Right side: portfolio-level metrics panel
Bottom: interactive charts
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import streamlit as st

from app.components.chart_gallery import render_charts
from app.components.metrics_panel import (
    render_metrics_detail,
    render_sector_exposure,
    render_top_metrics_bar,
)
from app.components.portfolio_table import render_portfolio_table
from app.state.persistence import list_saved_portfolios, load_portfolio, save_portfolio
from app.state.session import get_cache, get_portfolio, get_settings, set_portfolio
from core.metrics import risk_metrics
from core.mock_portfolio import generate_mock_portfolio
from core.rebalancer import RebalanceRequest, compute_rebalance
from data.ingest import load_from_excel


def _refresh_market_data() -> None:
    """Fetch/refresh price data and compute derived data for the current portfolio."""
    portfolio = get_portfolio()
    if portfolio is None:
        return

    cache = get_cache()
    settings = get_settings()
    lookback = settings.get("lookback_days", 252)

    tickers = portfolio.tickers
    benchmark = settings.get("benchmark", "SPY")
    all_tickers = list(set(tickers + [benchmark]))

    # Fetch prices
    end = date.today()
    start = end - timedelta(days=int(lookback * 1.5))  # extra buffer for weekends

    with st.spinner(f"Fetching price data for {len(all_tickers)} tickers..."):
        prices_df = cache.get_daily_prices(all_tickers, start, end)

    if prices_df.height == 0:
        st.warning("No price data available. Check your internet connection.")
        return

    # Store raw prices for use by portfolio table (ADV$ calc, etc.)
    st.session_state.prices_df = prices_df

    # Update portfolio with current prices
    current_prices = cache.get_current_prices(tickers)
    portfolio = portfolio.update_prices(current_prices)
    set_portfolio(portfolio)

    # Build returns DataFrame (wide format: one column per ticker)
    returns_frames = []
    for ticker in all_tickers:
        ticker_prices = prices_df.filter(pl.col("ticker") == ticker).sort("date")
        if ticker_prices.height < 2:
            continue
        price_col = "adj_close" if "adj_close" in ticker_prices.columns else "close"
        rets = ticker_prices.select(
            pl.col("date"),
            pl.col(price_col).pct_change().alias(ticker),
        ).drop_nulls()
        returns_frames.append(rets)

    if returns_frames:
        # Join all returns on date
        returns_df = returns_frames[0]
        for frame in returns_frames[1:]:
            returns_df = returns_df.join(frame, on="date", how="inner")

        st.session_state.returns_df = returns_df
    else:
        st.session_state.returns_df = None

    # Compute betas
    betas = {}
    if st.session_state.returns_df is not None and benchmark in st.session_state.returns_df.columns:
        market_rets = st.session_state.returns_df[benchmark]
        for ticker in tickers:
            if ticker in st.session_state.returns_df.columns:
                betas[ticker] = risk_metrics.position_beta(
                    st.session_state.returns_df[ticker],
                    market_rets,
                )
    st.session_state.betas = betas

    # Risk-free rate
    rf_setting = settings.get("risk_free_rate", "auto")
    if rf_setting == "auto":
        st.session_state.risk_free_rate_value = cache.get_risk_free_rate()
    else:
        st.session_state.risk_free_rate_value = float(rf_setting)

    # Update position metadata (sector, market cap) from cache
    new_positions = []
    for p in portfolio.positions:
        info = cache.get_ticker_info(p.ticker)
        from data.sector_map import classify_ticker
        sector, subsector = classify_ticker(
            p.ticker,
            yf_sector=info.get("sector", ""),
            yf_industry=info.get("industry", ""),
        )
        new_positions.append(p.model_copy(update={
            "sector": sector or p.sector,
            "subsector": subsector or p.subsector,
            "market_cap": info.get("market_cap", p.market_cap),
        }))
    portfolio = portfolio.model_copy(update={"positions": new_positions})
    set_portfolio(portfolio)


def _handle_rebalance(portfolio) -> None:
    """Handle the rebalance request/preview/apply lifecycle."""
    # Step 1: If rebalance was requested, compute the result
    if st.session_state.get("rebalance_requested"):
        st.session_state.rebalance_requested = False

        if portfolio is None:
            st.warning("Load a portfolio first.")
            return

        alerts = st.session_state.get("alerts", {})
        target_beta = alerts.get("max_net_beta")
        target_vol = alerts.get("max_ann_vol")

        # Build current prices dict from portfolio
        current_prices = {p.ticker: p.current_price for p in portfolio.positions}

        with st.spinner("Running optimizer..."):
            result = compute_rebalance(
                portfolio=portfolio,
                request=RebalanceRequest(
                    target_net_beta=target_beta,
                    target_ann_vol=target_vol,
                ),
                returns_df=st.session_state.get("returns_df"),
                betas=st.session_state.get("betas", {}),
                current_prices=current_prices,
            )
        st.session_state.rebalance_result = result

    # Step 2: If we have a result, show the preview
    result = st.session_state.get("rebalance_result")
    if result is None:
        return

    st.divider()
    st.subheader("⚖️ Rebalance Preview")

    # Warnings
    for w in result.warnings:
        st.warning(w)

    # Target status
    status_parts = []
    for key, met in result.target_met.items():
        icon = "✅" if met else "⚠️"
        label = key.replace("_", " ").title()
        status_parts.append(f"{icon} {label}")
    if status_parts:
        st.markdown("**Targets:** " + "  |  ".join(status_parts))

    if not result.converged:
        st.warning("Optimizer did not fully converge — results are best-effort.")

    # Before / After metrics
    col1, col2, col3 = st.columns(3)
    mb = result.metrics_before
    ma = result.metrics_after

    with col1:
        st.markdown("**Metric**")
        for key in mb:
            st.text(key.replace("_", " ").title())

    with col2:
        st.markdown("**Before**")
        for key, val in mb.items():
            st.text(f"{val:.3f}" if abs(val) < 10 else f"{val:.1f}")

    with col3:
        st.markdown("**After**")
        for key, val in ma.items():
            delta = val - mb[key]
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "–")
            st.text(f"{val:.3f} {arrow}" if abs(val) < 10 else f"{val:.1f} {arrow}")

    # Top trades table
    trades = result.trades_needed
    if trades:
        st.markdown(f"**{len(trades)} position(s) to adjust:**")
        top_trades = trades[:15]  # show top 15
        header = "| Ticker | Action | Old Wt% | New Wt% | Δ Notional |"
        separator = "|--------|--------|---------|---------|------------|"
        rows = [header, separator]
        for t in top_trades:
            rows.append(
                f"| {t['ticker']} | {t['action']} | "
                f"{t['old_weight']:+.2f}% | {t['new_weight']:+.2f}% | "
                f"${t['delta_notional']:+,.0f} |"
            )
        if len(trades) > 15:
            rows.append(f"| ... | +{len(trades) - 15} more | | | |")
        st.markdown("\n".join(rows))

    # Apply / Discard buttons
    btn_col1, btn_col2, _ = st.columns([1, 1, 4])
    with btn_col1:
        if st.button("✅ Apply Rebalance", use_container_width=True, type="primary"):
            set_portfolio(result.portfolio_after)
            st.session_state.rebalance_result = None
            # Clear cached returns so metrics recalculate with new weights
            st.session_state.returns_df = None
            st.success("Portfolio rebalanced.")
            st.rerun()
    with btn_col2:
        if st.button("❌ Discard", use_container_width=True):
            st.session_state.rebalance_result = None
            st.rerun()


def render() -> None:
    """Render the portfolio view page."""
    st.header("📊 Portfolio Dashboard")

    # --- Top action bar ---
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
        uploaded = st.file_uploader(
            "Upload Portfolio",
            type=["xlsx", "csv", "xls"],
            key="portfolio_upload",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            try:
                import tempfile
                ext = uploaded.name.split('.')[-1]
                with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                nav = get_settings().get("nav", 3_000_000_000)
                portfolio = load_from_excel(tmp_path, nav=nav)
                set_portfolio(portfolio)
                st.success(f"Loaded {portfolio.total_count} positions from {uploaded.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load: {e}")

    with col2:
        if st.button("🎲 Generate Mock Portfolio", use_container_width=True):
            with st.spinner("Generating mock portfolio..."):
                cache = get_cache()
                from data.universe import flat_universe
                all_tickers = flat_universe()
                prices = cache.get_current_prices(all_tickers)
                # Fetch betas from info
                betas = {}
                for t in all_tickers:
                    info = cache.get_ticker_info(t)
                    betas[t] = info.get("beta", 1.0)

                portfolio = generate_mock_portfolio(
                    betas=betas,
                    prices=prices,
                    nav=get_settings().get("nav", 3_000_000_000),
                )
                set_portfolio(portfolio)
            st.success(f"Generated: {portfolio.long_count}L / {portfolio.short_count}S")
            st.rerun()

    with col3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            if get_portfolio() is not None:
                _refresh_market_data()
                st.success("Data refreshed")
                st.rerun()
            else:
                st.warning("Load a portfolio first")

    with col4:
        if st.button("💾 Save Portfolio", use_container_width=True):
            portfolio = get_portfolio()
            if portfolio is not None:
                path = save_portfolio(portfolio)
                st.success(f"Saved to {path.name}")
            else:
                st.warning("No portfolio to save")

    with col5:
        saved = list_saved_portfolios()
        if saved:
            options = ["— Load Saved —"] + [f"{s['name']} ({s['positions']} pos)" for s in saved]
            choice = st.selectbox("Load", options, key="load_saved", label_visibility="collapsed")
            idx = options.index(choice) - 1
            if idx >= 0:
                portfolio = load_portfolio(saved[idx]["filepath"])
                set_portfolio(portfolio)
                st.rerun()

    st.divider()

    # --- Main content ---
    portfolio = get_portfolio()
    if portfolio is None:
        st.info(
            "👋 **Welcome to LS Portfolio Lab**\n\n"
            "Upload a portfolio (Excel/CSV) or generate a mock "
            "portfolio to get started.\n\n"
            "The mock portfolio generates **~30 longs / ~40 shorts** "
            "across all 11 GICS sectors from a ~440-name Russell 1000 "
            "universe with **$3B NAV** and ~5% cash reserve. "
            "Supports equities, ETFs, and options.\n\n"
            "**Features:** Top metrics bar (Vol, Beta, Sharpe, RSI), "
            "risk parameter dials, per-name annualized vol, drawdown "
            "analytics (Bailey & Lopez de Prado), factor tilt "
            "estimation, and full PM performance tracking."
        )
        return

    # Auto-refresh data if needed
    if st.session_state.returns_df is None:
        _refresh_market_data()

    # --- Top metrics bar (full width) ---
    render_top_metrics_bar(
        portfolio,
        returns_df=st.session_state.returns_df,
        betas=st.session_state.betas,
        risk_free_rate=st.session_state.risk_free_rate_value,
    )

    st.divider()

    # --- Detail metrics (full width, horizontal) ---
    render_metrics_detail(
        portfolio,
        returns_df=st.session_state.returns_df,
        betas=st.session_state.betas,
        risk_free_rate=st.session_state.risk_free_rate_value,
    )

    # --- Rebalance flow ---
    _handle_rebalance(portfolio)

    st.divider()

    # --- Portfolio Table (full width) ---
    render_portfolio_table(portfolio)

    # --- Sector Exposure (below portfolio table) ---
    st.divider()
    render_sector_exposure(portfolio)

    # Charts section
    st.divider()
    render_charts(
        portfolio,
        returns_df=st.session_state.returns_df,
        betas=st.session_state.betas,
    )

    # --- Data source disclaimer ---
    st.divider()
    cache = get_cache()
    _provider_name = cache.provider_name
    _source_details = {
        "Yahoo Finance": "End-of-day prices via yfinance (free, no API key).",
        "Bloomberg": "Real-time prices via Bloomberg Professional API (DAPI).",
        "Interactive Brokers": "Real-time prices via IB TWS / Gateway (ib_insync).",
    }
    _src = _source_details.get(_provider_name, f"Data from {_provider_name}.")
    st.caption(
        f"**Data:** {_src} "
        "Prices refresh every 18 hours; fundamentals every 7 days. "
        "Cached locally in SQLite. "
        "Drawdown analytics: Bailey & Lopez de Prado (2014). "
        "Factor models: Fama & French (1993), Carhart (1997). "
        "See [REFERENCES.md](https://github.com) for full citations."
    )
