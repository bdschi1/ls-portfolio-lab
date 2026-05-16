"""Risk Analytics Dashboard.

Five blocks: factor-vs-idio variance decomposition, style + sector tilts vs
target/benchmark, per-name risk contributions with the alpha/idio-vol
sizing yardstick, factor-decomposed P&L attribution, and a deep-link to
the trade simulator with a one-line impact preview.

Two design rules followed throughout:
- Every number sits next to a counterfactual (vs. target / vs. benchmark /
  vs. yesterday).
- One question per glance: "Am I being paid for the bets I meant to make?"
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import streamlit as st

from app.state.session import get_cache, get_portfolio, get_settings
from core.factor_model import build_proxy_factors
from core.metrics.factor_attribution import (
    factor_pnl_attribution,
    hit_rate,
    slugging,
)
from core.metrics.risk_contributions import per_name_risk_contributions
from core.metrics.style_tilts import style_tilts
from core.metrics.variance_decomp import variance_decomposition

_FACTOR_ETFS = ["SPY", "IWM", "IWD", "IWF", "MTUM"]
_SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}
# SPY GICS sector weights (approximate, S&P 500, late-2024 vintage)
_SPY_SECTOR_BENCHMARK = {
    "Technology": 0.32,
    "Financials": 0.13,
    "Health Care": 0.11,
    "Consumer Discretionary": 0.10,
    "Communication Services": 0.09,
    "Industrials": 0.08,
    "Consumer Staples": 0.06,
    "Energy": 0.04,
    "Utilities": 0.02,
    "Real Estate": 0.02,
    "Materials": 0.02,
}


def _ensure_factor_returns() -> pl.DataFrame | None:
    """Fetch factor + sector ETFs alongside the portfolio universe if missing.

    Returns the augmented returns_df, or None if no portfolio is loaded.
    """
    portfolio = get_portfolio()
    if portfolio is None:
        return None

    settings = get_settings()
    lookback = settings.get("lookback_days", 252)
    cache = get_cache()

    existing = st.session_state.get("returns_df")
    needed = set(_FACTOR_ETFS) | set(_SECTOR_ETFS.values()) | set(portfolio.tickers)
    if existing is not None:
        have = set(existing.columns) - {"date"}
        missing = needed - have
        if not missing:
            return existing
        fetch_list = list(missing)
    else:
        fetch_list = list(needed)

    end = date.today()
    start = end - timedelta(days=int(lookback * 1.5))
    with st.spinner(f"Fetching {len(fetch_list)} factor/sector tickers..."):
        prices_df = cache.get_daily_prices(fetch_list, start, end)

    if prices_df.height == 0:
        return existing

    frames = []
    for ticker in fetch_list:
        rows = prices_df.filter(pl.col("ticker") == ticker).sort("date")
        if rows.height < 2:
            continue
        price_col = "adj_close" if "adj_close" in rows.columns else "close"
        rets = rows.select(
            pl.col("date"),
            pl.col(price_col).pct_change().alias(ticker),
        ).drop_nulls()
        frames.append(rets)

    if not frames:
        return existing

    new_returns = frames[0]
    for frame in frames[1:]:
        new_returns = new_returns.join(frame, on="date", how="inner")

    if existing is not None:
        merged = existing.join(new_returns, on="date", how="inner")
    else:
        merged = new_returns

    st.session_state.returns_df = merged
    return merged


def _build_factor_returns(returns_df: pl.DataFrame) -> dict[str, pl.Series]:
    """Return {market, smb, hml, momentum} from build_proxy_factors.

    Lowercase keys are kept because factor_attribution.py hardcodes 'market';
    variance_decomp/risk_contributions/style_tilts accept arbitrary keys
    (variance_decomp via a canonical-name map).
    """
    return build_proxy_factors(returns_df)


def _build_sector_returns(returns_df: pl.DataFrame) -> dict[str, pl.Series]:
    out: dict[str, pl.Series] = {}
    for sector, etf in _SECTOR_ETFS.items():
        if etf in returns_df.columns:
            out[sector] = returns_df[etf]
    return out


def _delta_caption(current: float | None, baseline: float | None, fmt: str = "{:+.1%}") -> str:
    """Render a small Δ vs. baseline caption."""
    if current is None or baseline is None:
        return "—"
    return f"Δ {fmt.format(current - baseline)} vs prior"


# ---------------------------------------------------------------------------
# Block 1 — Factor risk decomposition
# ---------------------------------------------------------------------------


def _render_block1(decomp) -> None:
    st.subheader("① Risk Decomposition")
    st.caption(
        "Where is portfolio variance coming from? In a stock-picker's book, "
        "**idiosyncratic should dominate** — that's what you're paid for."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Market variance",
        f"{decomp.market_pct * 100:.1f}%",
        help=f"${(decomp.market_var**0.5) * 100:.2f}/day equivalent vol contribution",
    )
    style_share = sum(decomp.style_pct.values())
    c2.metric(
        "Style factors",
        f"{style_share * 100:.1f}%",
        help=f"Per-factor: {', '.join(f'{k}={v * 100:.1f}%' for k, v in decomp.style_pct.items())}",
    )
    c3.metric(
        "Sector",
        f"{decomp.sector_pct * 100:.1f}%",
        help="Net sector-ETF residual variance after market + style strip",
    )
    c4.metric(
        "Idiosyncratic",
        f"{decomp.idio_pct * 100:.1f}%",
        delta=f"target ≥ 70%",  # noqa: F541 — Streamlit metric expects a string
        delta_color="off",
        help="Stock-specific risk after factor strip. Higher = more skill-driven.",
    )
    annualized_vol = decomp.total_var**0.5
    st.caption(
        f"Total annualized portfolio variance: {decomp.total_var:.6f} "
        f"(equiv ann. vol ≈ {annualized_vol * 100:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Block 2 — Style & sector tilts
# ---------------------------------------------------------------------------


def _render_block2(tilts) -> None:
    st.subheader("② Style & Sector Tilts")
    st.caption("Loadings vs. your targets. Sector net vs. SPY benchmark.")

    left, right = st.columns(2)

    with left:
        st.markdown("**Style loadings**")
        style_rows = [
            {
                "Factor": s.factor,
                "Loading": f"{s.portfolio_loading:+.2f}",
                "Target": "—" if s.target_loading is None else f"{s.target_loading:+.2f}",
                "Drift": "—" if s.drift is None else f"{s.drift:+.2f}",
            }
            for s in tilts.style
        ]
        if style_rows:
            st.dataframe(pl.DataFrame(style_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No style loadings — need ≥30 days of factor returns.")

    with right:
        st.markdown("**Sector tilts (net signed weight)**")
        sector_rows = [
            {
                "Sector": s.sector,
                "Long %": f"{s.long_weight * 100:+.1f}%",
                "Short %": f"-{s.short_weight * 100:.1f}%",
                "Net %": f"{s.portfolio_weight * 100:+.1f}%",
                "Bench %": "—"
                if s.benchmark_weight is None
                else f"{s.benchmark_weight * 100:.1f}%",
                "Active %": "—" if s.active_weight is None else f"{s.active_weight * 100:+.1f}%",
            }
            for s in tilts.sectors
        ]
        if sector_rows:
            st.dataframe(pl.DataFrame(sector_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No sector positions classified.")

    crowding_with_data = [c for c in tilts.crowding if c.source != "placeholder"]
    if crowding_with_data:
        st.markdown("**Crowding**")
        st.dataframe(
            pl.DataFrame(
                [
                    {"Ticker": c.ticker, "Score": f"{c.score:.2f}", "Source": c.source}
                    for c in crowding_with_data
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption(
            "Crowding scores: placeholder — wire HF 13F ownership or short-interest "
            "data to make this meaningful."
        )


# ---------------------------------------------------------------------------
# Block 3 — Per-name risk contributions
# ---------------------------------------------------------------------------


def _render_block3(contribs: list, portfolio_total_vol_val: float, nav: float) -> None:
    st.subheader("③ Per-Name Risk Contributions")
    st.caption(
        "**alpha/idio** is the IR-style sizing yardstick. Names where it's low but "
        "position is large are the trim candidates."
    )
    if not contribs:
        st.info("No per-name contributions to display.")
        return

    rows = [
        {
            "Ticker": c.ticker,
            "Side": c.side,
            "$ Notional": f"${c.notional:,.0f}",
            "Wt %": f"{c.weight * 100:+.1f}%",
            "β mkt": f"{c.beta_market:+.2f}",
            "Idio vol": f"{c.idio_vol_ann * 100:.1f}%",
            "Total vol": f"{c.total_vol_ann * 100:.1f}%",
            "MCTR total": f"{c.mctr_total * 100:+.2f}%",
            "CCTR total": f"{c.cctr_total * 100:+.2f}%",
            "MCTR idio": f"{c.mctr_idio * 100:+.2f}%",
            "$ Vol contrib": f"${c.dollar_vol_contrib:,.0f}",
            "α est (ann)": f"{c.alpha_estimate * 100:+.1f}%",
            "α / idio": f"{c.alpha_over_idio:+.2f}",
        }
        for c in contribs
    ]
    df = pl.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, height=420)

    sum_cctr = sum(c.cctr_total for c in contribs)
    delta_vol = sum_cctr - portfolio_total_vol_val
    st.caption(
        f"Σ CCTR = {sum_cctr * 100:.2f}% vs portfolio total vol = "
        f"{portfolio_total_vol_val * 100:.2f}% (reconciliation Δ = "
        f"{delta_vol * 100:+.4f}%) · NAV = ${nav:,.0f}"
    )


# ---------------------------------------------------------------------------
# Block 4 — Factor-decomposed P&L attribution
# ---------------------------------------------------------------------------


def _render_block4(
    portfolio,
    returns_df: pl.DataFrame,
    factor_returns: dict[str, pl.Series],
    sector_returns: dict[str, pl.Series] | None,
) -> None:
    st.subheader("④ P&L Attribution")
    st.caption("Decompose realized P&L by horizon. Idio P&L is the one that matters.")

    horizon = st.radio(
        "Horizon",
        options=["1D", "WTD", "MTD", "YTD"],
        index=2,
        horizontal=True,
        key="risk_analytics_horizon",
    )
    attr = factor_pnl_attribution(
        portfolio,
        returns_df,
        factor_returns,
        sector_returns=sector_returns,
        horizon=horizon,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total P&L", f"${attr.total_pnl:+,.0f}")
    c2.metric("Market", f"${attr.market_pnl:+,.0f}")
    c3.metric(
        "Style",
        f"${sum(attr.style_pnl.values()):+,.0f}",
        help=f"Per-factor: {', '.join(f'{k}={v:+,.0f}' for k, v in attr.style_pnl.items())}",
    )
    c4.metric("Sector", f"${attr.sector_pnl:+,.0f}")
    c5.metric(
        "Idiosyncratic",
        f"${attr.idio_pnl:+,.0f}",
        help="Stock-specific P&L — the part you're paid for.",
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Top 10 idio winners** ($)")
        if attr.top_idio_winners:
            st.dataframe(
                pl.DataFrame(
                    [
                        {"Ticker": n.ticker, "Side": n.side, "Idio $": f"${n.pnl_idio:+,.0f}"}
                        for n in attr.top_idio_winners
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No idio winners over this horizon.")
    with right:
        st.markdown("**Top 10 idio losers** ($)")
        if attr.top_idio_losers:
            st.dataframe(
                pl.DataFrame(
                    [
                        {"Ticker": n.ticker, "Side": n.side, "Idio $": f"${n.pnl_idio:+,.0f}"}
                        for n in attr.top_idio_losers
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No idio losers over this horizon.")

    hr = hit_rate(attr.name_contribs)
    sl = slugging(attr.name_contribs)
    sl_str = "—" if sl != sl else f"{sl:.2f}×"  # NaN check
    st.caption(
        f"Idio hit rate: **{hr * 100:.0f}%** · Slugging (avg win $ / avg loss $): **{sl_str}**"
    )


# ---------------------------------------------------------------------------
# Block 5 — Trade staging deep-link
# ---------------------------------------------------------------------------


def _render_block5() -> None:
    st.subheader("⑤ Trade Staging")
    st.caption(
        "Stage proposed trades and preview their impact on net beta, factor "
        "loadings, and risk budget. Open the simulator for the full before/after view."
    )
    if st.button("→ Open Trade Simulator", use_container_width=False):
        st.session_state.nav_radio = "Trade Simulator"
        st.session_state.active_page = "Trade Simulator"
        st.rerun()


# ---------------------------------------------------------------------------
# Page entry
# ---------------------------------------------------------------------------


def render() -> None:
    st.title("📐 Risk Analytics")
    st.caption(
        "Risk-decomposed cockpit. One question per glance: **Am I being paid "
        "for the bets I meant to make, or for bets I didn't know I had?**"
    )

    portfolio = get_portfolio()
    if portfolio is None:
        st.info("Load a portfolio on the **Portfolio Dashboard** page first, then return here.")
        return

    returns_df = _ensure_factor_returns()
    if returns_df is None or returns_df.height < 30:
        st.warning(
            "Need at least 30 days of overlapping returns for the factor regressions. "
            "Try a longer lookback window."
        )
        return

    factor_returns = _build_factor_returns(returns_df)
    if not factor_returns:
        st.warning("Could not build factor returns — check SPY/IWM/IWD/IWF/MTUM are available.")
        return

    sector_returns = _build_sector_returns(returns_df) or None

    # Targets: if user previously set them, pull from session state; else None
    style_targets = st.session_state.get("risk_analytics_style_targets")

    # ---- compute all 4 modules ----
    with st.spinner("Computing risk decomposition..."):
        decomp = variance_decomposition(
            portfolio, returns_df, factor_returns, sector_returns=sector_returns
        )
    with st.spinner("Computing tilts..."):
        tilts = style_tilts(
            portfolio,
            returns_df,
            factor_returns,
            targets=style_targets,
            benchmark_weights=_SPY_SECTOR_BENCHMARK,
        )
    with st.spinner("Computing per-name contributions..."):
        contribs = per_name_risk_contributions(portfolio, returns_df, factor_returns)
        from core.metrics.risk_contributions import portfolio_total_vol

        port_vol = portfolio_total_vol(portfolio, returns_df)

    # ---- render ----
    _render_block1(decomp)
    st.divider()
    _render_block2(tilts)
    st.divider()
    _render_block3(contribs, port_vol, portfolio.nav)
    st.divider()
    _render_block4(portfolio, returns_df, factor_returns, sector_returns)
    st.divider()
    _render_block5()

    # ---- footer: optional style target inputs ----
    with st.expander("⚙️ Set style-loading targets (for drift)"):
        st.caption("Targets drive the Drift column in block ② above.")
        targets: dict[str, float] = {}
        cols = st.columns(len(factor_returns))
        for col, fname in zip(cols, factor_returns.keys(), strict=False):
            with col:
                val = st.number_input(
                    f"Target {fname}",
                    value=float(style_targets.get(fname, 0.0)) if style_targets else 0.0,
                    step=0.05,
                    format="%.2f",
                    key=f"target_{fname}",
                )
                targets[fname] = val
        if st.button("Save targets"):
            st.session_state.risk_analytics_style_targets = targets
            st.rerun()
