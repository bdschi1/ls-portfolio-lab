"""Portfolio table component — sortable, filterable position display.

Shows all positions with key per-position metrics including annualized vol,
beta-weighted exposure, target price, and upside %.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import streamlit as st

from app.state.session import get_settings
from core.metrics import return_metrics, risk_metrics, technical_metrics
from core.portfolio import Portfolio

# Default columns shown in the table
DEFAULT_VISIBLE_COLUMNS = [
    "Ticker", "Side", "Pos Size $", "Weight %", "Price",
    "P&L %", "Beta", "Ann Vol", "RSI", "Sector",
]

ALL_COLUMNS = [
    "Ticker", "Type", "Side", "Shares", "Pos Size $", "Beta-Wtd $",
    "Weight %", "Price", "P&L %", "P&L $", "Beta", "Ann Vol",
    "Idio Vol", "α 30d", "α 1yr", "RSI", "ADV$", "Target", "Upside %",
    "Sector", "Subsector",
]


_BASE_TABLE_FONT = 1.19  # rem — baseline after all previous bumps
_FONT_STEP = 0.12       # rem per click


def render_portfolio_table(portfolio: Portfolio) -> None:
    """Render the portfolio positions as an interactive table."""
    # --- Font size control (persisted in session state) ---
    if "table_font_offset" not in st.session_state:
        st.session_state.table_font_offset = 0

    font_size = _BASE_TABLE_FONT + st.session_state.table_font_offset

    # +/- buttons next to the header
    hdr_col, btn_minus, btn_plus = st.columns([8, 0.5, 0.5])
    with hdr_col:
        st.subheader(f"📊 Portfolio ({portfolio.total_count} positions)")
    with btn_minus:
        if st.button("A−", key="tbl_font_down", help="Decrease table font"):
            st.session_state.table_font_offset -= _FONT_STEP
            st.rerun()
    with btn_plus:
        if st.button("A+", key="tbl_font_up", help="Increase table font"):
            st.session_state.table_font_offset += _FONT_STEP
            st.rerun()

    # --- Dynamic CSS based on current font size ---
    st.markdown(f"""<style>
    div[data-testid="stDataFrame"] table {{
        font-size: {font_size:.2f}rem !important;
    }}
    div[data-testid="stDataFrame"] th {{
        font-size: {font_size:.2f}rem !important;
    }}
    div[data-testid="stDataFrame"] td {{
        font-size: {font_size:.2f}rem !important;
    }}
    div[data-testid="stCaptionContainer"] p {{
        font-size: {font_size:.2f}rem !important;
    }}
    </style>""", unsafe_allow_html=True)
    st.caption(
        "All current positions with per-name risk metrics. "
        "Use filters to narrow the view and column selector to customize."
    )

    # --- Column Visibility ---
    visible_cols = st.multiselect(
        "Columns",
        ALL_COLUMNS,
        default=DEFAULT_VISIBLE_COLUMNS,
        key="table_visible_columns",
    )
    if not visible_cols:
        visible_cols = DEFAULT_VISIBLE_COLUMNS

    # --- Filters ---
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        side_filter = st.selectbox(
            "Side", ["All", "Long", "Short"],
            key="table_side_filter",
        )
    with filter_col2:
        sectors = sorted(set(p.sector for p in portfolio.positions if p.sector))
        sector_filter = st.selectbox(
            "Sector", ["All"] + sectors,
            key="table_sector_filter",
        )
    with filter_col3:
        _sort_by = st.selectbox(  # noqa: F841 — UI control, sort handled by st.dataframe
            "Sort by", ["Ticker", "Weight", "P&L %", "Beta", "Ann Vol", "Sector", "Side"],
            key="table_sort",
        )

    # Build table data
    rows = []
    returns_df = st.session_state.get("returns_df")
    prices_df = st.session_state.get("prices_df")
    betas = st.session_state.get("betas", {})
    target_prices = st.session_state.get("target_prices", {})
    settings = get_settings()
    rsi_period = settings.get("rsi_period", 14)

    for p in portfolio.positions:
        # Apply filters
        if side_filter == "Long" and p.side != "LONG":
            continue
        if side_filter == "Short" and p.side != "SHORT":
            continue
        if sector_filter != "All" and p.sector != sector_filter:
            continue

        beta = betas.get(p.ticker, 1.0)
        weight = p.weight_in(portfolio.nav) * 100  # as percentage

        # RSI — compute if we have price data
        rsi_val = None
        if returns_df is not None and p.ticker in returns_df.columns:
            rets = returns_df[p.ticker]
            price_series = (1 + rets).cum_prod() * 100
            rsi_val = technical_metrics.rsi_current(price_series, rsi_period)

        # Annualized vol per name
        ann_vol = None
        idio_vol = None
        if returns_df is not None and p.ticker in returns_df.columns:
            ticker_rets = returns_df[p.ticker]
            if ticker_rets.len() > 20:
                ann_vol = return_metrics.annualized_volatility(ticker_rets)
                # Idiosyncratic vol — residual after removing market component
                benchmark = settings.get("benchmark", "SPY")
                if benchmark in returns_df.columns:
                    idio_vol = risk_metrics.idiosyncratic_volatility(
                        ticker_rets, returns_df[benchmark],
                    )

        # Excess return vs SPX (30d and 1yr)
        alpha_30d = None
        alpha_1yr = None
        if returns_df is not None and p.ticker in returns_df.columns:
            benchmark = settings.get("benchmark", "SPY")
            if benchmark in returns_df.columns:
                t_rets = returns_df[p.ticker]
                b_rets = returns_df[benchmark]
                # 30-day excess return
                if t_rets.len() >= 30 and b_rets.len() >= 30:
                    t30 = float((1 + t_rets.tail(30)).product() - 1)
                    b30 = float((1 + b_rets.tail(30)).product() - 1)
                    alpha_30d = (t30 - b30) * 100  # as pct
                # 1-year (252d) excess return
                if t_rets.len() >= 252 and b_rets.len() >= 252:
                    t252 = float((1 + t_rets.tail(252)).product() - 1)
                    b252 = float((1 + b_rets.tail(252)).product() - 1)
                    alpha_1yr = (t252 - b252) * 100  # as pct

        # Avg daily trading volume $ (last 20 days: mean of close * volume)
        adv_dollar = None
        if prices_df is not None:
            tkr_prices = prices_df.filter(pl.col("ticker") == p.ticker).sort("date")
            if tkr_prices.height >= 20:
                last20 = tkr_prices.tail(20)
                dollar_vol = last20.select(
                    (pl.col("close") * pl.col("volume")).alias("dv")
                )["dv"]
                adv_dollar = float(dollar_vol.mean())

        # Beta-weighted notional
        beta_weighted = p.notional * beta

        # Target price & upside
        target = target_prices.get(p.ticker, 0.0)
        upside = None
        if target > 0 and p.current_price > 0:
            upside = (target - p.current_price) / p.current_price * 100

        rows.append({
            "Ticker": p.ticker,
            "Type": p.asset_type,
            "Side": p.side,
            "Shares": f"{p.shares:,.0f}",
            "Pos Size $": f"${p.notional:,.0f}",
            "Beta-Wtd $": f"${beta_weighted:,.0f}",
            "Weight %": f"{weight:+.1f}%",
            "Price": f"${p.current_price:,.2f}" if p.current_price > 0 else "—",
            "P&L %": f"{p.pnl_pct * 100:+.1f}%" if p.current_price > 0 else "—",
            "P&L $": f"${p.pnl_dollars:+,.0f}" if p.current_price > 0 else "—",
            "Beta": f"{beta:.2f}",
            "Ann Vol": f"{ann_vol:.1%}" if ann_vol is not None else "—",
            "Idio Vol": f"{idio_vol:.1%}" if idio_vol is not None else "—",
            "α 30d": f"{alpha_30d:+.1f}%" if alpha_30d is not None else "—",
            "α 1yr": f"{alpha_1yr:+.1f}%" if alpha_1yr is not None else "—",
            "RSI": f"{rsi_val:.0f}" if rsi_val is not None else "—",
            "ADV$": f"${adv_dollar:,.0f}" if adv_dollar is not None else "—",
            "Target": f"${target:,.2f}" if target > 0 else "—",
            "Upside %": f"{upside:+.1f}%" if upside is not None else "—",
            "Sector": p.sector,
            "Subsector": p.subsector,
        })

    if not rows:
        st.info("No positions match the current filters.")
        return

    df = pl.DataFrame(rows)

    # Filter to visible columns only
    display_cols = [c for c in visible_cols if c in df.columns]
    if display_cols:
        df = df.select(display_cols)

    # Column config for display widths
    col_config = {
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Type": st.column_config.TextColumn("Type", width="small"),
        "Side": st.column_config.TextColumn("Side", width="small"),
        "Shares": st.column_config.TextColumn("Shares", width="small"),
        "Pos Size $": st.column_config.TextColumn("Pos$", width="small"),
        "Beta-Wtd $": st.column_config.TextColumn("β$", width="small"),
        "Weight %": st.column_config.TextColumn("Wt%", width="small"),
        "Price": st.column_config.TextColumn("Price", width="small"),
        "P&L %": st.column_config.TextColumn("P&L%", width="small"),
        "P&L $": st.column_config.TextColumn("P&L$", width="small"),
        "Beta": st.column_config.TextColumn("β", width="small"),
        "Ann Vol": st.column_config.TextColumn("Vol", width="small"),
        "Idio Vol": st.column_config.TextColumn("Idio", width="small"),
        "α 30d": st.column_config.TextColumn("α30d", width="small"),
        "α 1yr": st.column_config.TextColumn("α1yr", width="small"),
        "RSI": st.column_config.TextColumn("RSI", width="small"),
        "ADV$": st.column_config.TextColumn("ADV$", width="small"),
        "Target": st.column_config.TextColumn("Target", width="small"),
        "Upside %": st.column_config.TextColumn("Up%", width="small"),
        "Sector": st.column_config.TextColumn("Sector", width="medium"),
        "Subsector": st.column_config.TextColumn("Sub", width="medium"),
    }

    # Only include configs for visible columns
    active_config = {k: v for k, v in col_config.items() if k in display_cols}

    st.dataframe(
        df,
        use_container_width=True,
        height=min(35 * len(rows) + 50, 600),
        column_config=active_config,
    )

    # Summary row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Long", portfolio.long_count)
    with col2:
        st.metric("Short", portfolio.short_count)
    with col3:
        st.metric("Total P&L", f"${portfolio.total_pnl_dollars:+,.0f}")
    with col4:
        st.metric("Total P&L %", f"{portfolio.total_pnl_pct * 100:+.2f}%")
