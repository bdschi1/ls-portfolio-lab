"""Metrics panel — horizontal top bar + right-side detail panel.

Top bar: key PM metrics at a glance (Portfolio Vol, Net Beta, Sharpe, RSI, etc.)
Detail panel: deeper risk, drawdown analytics, factor decomposition, correlation.
"""

from __future__ import annotations

import math

import plotly.graph_objects as go
import polars as pl
import streamlit as st
import streamlit.components.v1 as components

from core.factor_model import (
    build_proxy_factors,
    capm_regression,
    multi_factor_regression,
)
from core.metrics import (
    correlation_metrics,
    drawdown_metrics,
    exposure_metrics,
    return_metrics,
    risk_metrics,
    technical_metrics,
)
from core.metrics.drawdown_analytics import drawdown_table, time_in_drawdown_pct
from core.metrics.quality_score import compute_quality_score
from core.portfolio import Portfolio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weighted_rsi(
    portfolio: Portfolio,
    returns_df: pl.DataFrame | None,
    side: str,
    rsi_period: int = 14,
) -> float | None:
    """Notional-weighted average RSI for the long or short book.

    Args:
        side: "LONG" or "SHORT"
    Returns:
        Weighted RSI value, or None if no data.
    """
    if returns_df is None:
        return None

    positions = [p for p in portfolio.positions if p.side == side]
    if not positions:
        return None

    total_notional = sum(p.notional for p in positions)
    if total_notional == 0:
        return None

    weighted_sum = 0.0
    valid_notional = 0.0

    for p in positions:
        if p.ticker in returns_df.columns:
            rets = returns_df[p.ticker]
            price_series = (1 + rets).cum_prod() * 100
            rsi_val = technical_metrics.rsi_current(price_series, rsi_period)
            weighted_sum += rsi_val * p.notional
            valid_notional += p.notional

    if valid_notional == 0:
        return None
    return weighted_sum / valid_notional


def _safe_fmt(value: float, fmt: str, fallback: str = "—") -> str:
    """Format a float, returning fallback for inf/nan."""
    if math.isinf(value) or math.isnan(value):
        return fallback
    return fmt.format(value)


# ---------------------------------------------------------------------------
# Top Metrics Bar (full-width, compact)
# ---------------------------------------------------------------------------

def render_top_metrics_bar(
    portfolio: Portfolio,
    returns_df: pl.DataFrame | None = None,
    betas: dict[str, float] | None = None,
    risk_free_rate: float = 0.05,
) -> None:
    """Render a compact horizontal metrics bar at the top of the dashboard.

    Shows the most important PM metrics at a glance.
    """
    if betas is None:
        betas = {}

    settings = st.session_state.get("settings", {})
    rsi_period = settings.get("rsi_period", 14)

    # Compute portfolio-level values
    port_vol: float | None = None
    sharpe: float | None = None
    time_dd: float | None = None

    if returns_df is not None and returns_df.height > 20:
        weights = portfolio.weight_vector()
        port_rets = return_metrics.portfolio_daily_returns(weights, returns_df)
        if port_rets.len() > 20:
            port_vol = risk_metrics.portfolio_volatility(weights, returns_df)
            sharpe = return_metrics.sharpe_ratio(port_rets, risk_free_rate)
            if sharpe and sharpe > 0:
                time_dd = time_in_drawdown_pct(sharpe)

    net_beta = exposure_metrics.net_beta_exposure(portfolio, betas) if betas else 0.0
    rsi_long = _weighted_rsi(portfolio, returns_df, "LONG", rsi_period)
    rsi_short = _weighted_rsi(portfolio, returns_df, "SHORT", rsi_period)

    # --- Render metrics in a compact row: bold green, tighter spacing ---
    st.markdown(
        """<style>
        div[data-testid="stMetric"] {
            padding: 0.15rem 0 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricLabel"] p {
            font-size: clamp(10px, 0.85vw, 16px) !important;
            font-weight: 700 !important;
            color: #2ecc71 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            font-size: clamp(13px, 1.1vw, 21px) !important;
            font-weight: 700 !important;
            color: #2ecc71 !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    # Compute Quality Score
    qs = compute_quality_score(portfolio, returns_df, betas, risk_free_rate)
    st.session_state["quality_score"] = qs  # cache for charts

    cols = st.columns(11, gap="small")

    with cols[0]:
        if port_vol is not None:
            st.metric("Port Vol", f"{port_vol:.1%}")
        else:
            st.metric("Port Vol", "—")

    with cols[1]:
        st.metric("Net Beta", f"{net_beta:+.3f}")

    with cols[2]:
        if sharpe is not None:
            st.metric("Sharpe", f"{sharpe:.2f}")
        else:
            st.metric("Sharpe", "—")

    with cols[3]:
        if rsi_long is not None:
            st.metric("RSI (L)", f"{rsi_long:.0f}")
        else:
            st.metric("RSI (L)", "—")

    with cols[4]:
        if rsi_short is not None:
            st.metric("RSI (S)", f"{rsi_short:.0f}")
        else:
            st.metric("RSI (S)", "—")

    with cols[5]:
        st.metric("Gross", f"{portfolio.gross_exposure:.0%}")

    with cols[6]:
        st.metric("Net", f"{portfolio.net_exposure:+.0%}")

    with cols[7]:
        if portfolio.cash:
            cash_label = "Cash ⚠️" if portfolio.cash_pct < portfolio.min_cash_pct else "Cash"
            st.metric(cash_label, f"{portfolio.cash_pct:.1%}")
        else:
            st.metric("Cash", "—")

    with cols[8]:
        if time_dd is not None:
            # Cap at 100% — formula 1/(2·SR²) exceeds 100% for low Sharpe
            capped_dd = min(time_dd, 100.0)
            st.metric("Time in DD", f"{capped_dd:.0f}%")
        else:
            st.metric("Time in DD", "—")

    with cols[9]:
        st.metric("Positions", f"{portfolio.long_count}L/{portfolio.short_count}S")

    with cols[10]:
        st.metric(
            "Quality", f"{qs.total_score:.0f} ({qs.grade})",
            help=(
                "Portfolio Quality Score (0–100). "
                "Weighted composite of 6 dimensions: "
                "Risk-Adj Return (Sharpe, 20%), "
                "DD Resilience (expected drawdown + time in DD, 18%), "
                "Alpha Quality (Deflated Sharpe Ratio, 18%), "
                "Diversification (avg pairwise ρ, 16%), "
                "Tail Risk (CVaR/VaR ratio, 14%), "
                "Exposure Balance (net β, L/S ratio, 14%). "
                "Select 'Quality Score' in Charts for full breakdown."
            ),
        )

    # --- Risk Parameter Dials (expander) ---
    with st.expander("⚙️ Risk Limits", expanded=False):
        dial_cols = st.columns(2)
        alert_cfg = st.session_state.get("alerts", {})

        with dial_cols[0]:
            current_beta_pct = int(alert_cfg.get("max_net_beta", 0.08) * 100)
            new_beta = st.slider(
                "Max Net Beta MV %",
                min_value=-25,
                max_value=25,
                value=current_beta_pct,
                step=1,
                key="dial_max_net_beta",
                help="Maximum net beta-adjusted market value exposure (±25%)",
            )
            if "alerts" in st.session_state:
                st.session_state.alerts["max_net_beta"] = new_beta / 100.0

        with dial_cols[1]:
            current_vol_pct = int(alert_cfg.get("max_ann_vol", 0.06) * 100)
            new_vol = st.slider(
                "Target Ann Vol %",
                min_value=3,
                max_value=15,
                value=current_vol_pct,
                step=1,
                key="dial_max_ann_vol",
                help="Maximum annualized portfolio volatility (3-15%)",
            )
            if "alerts" in st.session_state:
                st.session_state.alerts["max_ann_vol"] = new_vol / 100.0

        st.divider()
        if st.button("⚖️ Rebalance to Targets", use_container_width=True,
                      help="Optimize position sizes to hit the Net Beta and Vol targets above"):
            st.session_state.rebalance_requested = True
            st.rerun()

    st.caption(
        "Portfolio Vol (annualized) • Net Beta • Sharpe Ratio "
        "• Notional-weighted RSI (Long/Short book) "
        "• Gross/Net Exposure • Cash % "
        "• Expected Time in Drawdown • Position Count"
    )


# ---------------------------------------------------------------------------
# Detail Metrics Panel (right column)
# ---------------------------------------------------------------------------

def _fmt_dollar(val: float) -> str:
    """Format a dollar amount compactly (e.g. $1.315B, $450M)."""
    if abs(val) >= 1e9:
        return f"${val / 1e9:.3f}B"
    if abs(val) >= 1e6:
        return f"${val / 1e6:,.1f}M"
    if abs(val) >= 1e3:
        return f"${val / 1e3:,.0f}K"
    return f"${val:,.0f}"


def _build_table_rows(label: str, value: str) -> str:
    """Build a single HTML table row."""
    return (
        f'<tr><td style="padding:3px 8px;color:#888;white-space:nowrap;">'
        f'{label}</td><td style="padding:3px 8px;font-weight:600;'
        f'white-space:nowrap;">{value}</td></tr>'
    )


def render_metrics_detail(
    portfolio: Portfolio,
    returns_df: pl.DataFrame | None = None,
    betas: dict[str, float] | None = None,
    risk_free_rate: float = 0.05,
) -> None:
    """Render the detailed metrics panel as a bordered HTML table grid.

    Laid out in columns: Summary | Risk | Drawdown | Factors | Correlation
    Plus a Sector Exposure chart below.
    Sits between the top metrics bar and the portfolio table.
    """
    if betas is None:
        betas = {}

    # Pre-compute all values so we know which sections are available
    weights = None
    port_rets = None
    benchmark = st.session_state.settings.get("benchmark", "SPY")
    has_returns = returns_df is not None and returns_df.height > 20

    if has_returns:
        weights = portfolio.weight_vector()
        port_rets = return_metrics.portfolio_daily_returns(weights, returns_df)
        if port_rets.len() <= 20:
            port_rets = None

    # --- Build each column's rows ---

    # COL 1: Summary (with Market Value)
    summary_rows = [
        _build_table_rows("NAV", _fmt_dollar(portfolio.nav)),
        _build_table_rows("Long MV", _fmt_dollar(portfolio.long_notional)),
        _build_table_rows("Short MV", _fmt_dollar(portfolio.short_notional)),
        _build_table_rows("Gross MV", _fmt_dollar(portfolio.gross_notional)),
    ]
    ls_ratio = (
        f"{portfolio.long_short_ratio:.2f}"
        if portfolio.short_notional > 0 else "∞"
    )
    summary_rows.append(_build_table_rows("L/S Ratio", ls_ratio))
    hhi = exposure_metrics.concentration_hhi(portfolio)
    top5 = exposure_metrics.top_n_concentration(portfolio, 5)
    summary_rows.append(_build_table_rows("HHI", f"{hhi:.4f}"))
    summary_rows.append(_build_table_rows("Top 5", f"{top5:.1%}"))

    # COL 2: Risk
    risk_rows: list[str] = []
    if port_rets is not None:
        sortino = return_metrics.sortino_ratio(port_rets, risk_free_rate)
        risk_rows.append(_build_table_rows("Sortino", f"{sortino:.2f}"))
        var = risk_metrics.var_historical(port_rets)
        risk_rows.append(_build_table_rows("VaR 95%", f"{var:.2%}"))
        cvar = risk_metrics.cvar_historical(port_rets)
        risk_rows.append(_build_table_rows("CVaR 95%", f"{cvar:.2%}"))
        dsr = return_metrics.deflated_sharpe_ratio(port_rets, risk_free_rate)
        risk_rows.append(_build_table_rows("DSR", f"{dsr:.1%}"))

        # Sharpe inference: PSR, CI, MinTRL
        sr_ci = return_metrics.sharpe_confidence_interval(port_rets, risk_free_rate)
        risk_rows.append(_build_table_rows("PSR", f"{sr_ci['psr']:.1%}"))
        risk_rows.append(_build_table_rows(
            "Sharpe CI",
            f"[{sr_ci['ci_lower']:.2f}, {sr_ci['ci_upper']:.2f}]",
        ))
        min_trl = sr_ci["min_track_record"]
        n_obs = port_rets.len()
        if not math.isinf(min_trl) and min_trl > n_obs:
            extra = int(min_trl - n_obs)
            risk_rows.append(_build_table_rows("MinTRL", f"Need {extra}+ days"))
        elif not math.isinf(min_trl):
            risk_rows.append(_build_table_rows("MinTRL", "✓ Sufficient"))

        if has_returns and benchmark in returns_df.columns:
            te = return_metrics.tracking_error(port_rets, returns_df[benchmark])
            risk_rows.append(_build_table_rows("Track Err", f"{te:.1%}"))
    else:
        risk_rows.append(_build_table_rows("—", "Need returns data"))

    # COL 3: Drawdown
    dd_rows: list[str] = []
    if port_rets is not None and port_rets.len() > 5:
        cum_ret = return_metrics.cumulative_return(port_rets)
        max_dd = drawdown_metrics.max_drawdown_from_returns(port_rets)
        curr_dd = drawdown_metrics.current_drawdown(cum_ret)
        calmar = return_metrics.calmar_ratio(port_rets)
        dd_rows.append(_build_table_rows("Max DD", f"{max_dd:.1%}"))
        dd_rows.append(_build_table_rows("Curr DD", f"{curr_dd:.1%}"))
        dd_rows.append(_build_table_rows("Calmar", f"{calmar:.2f}"))

        sharpe = return_metrics.sharpe_ratio(port_rets, risk_free_rate)
        ann_vol = return_metrics.annualized_volatility(port_rets)
        if sharpe > 0 and ann_vol > 0:
            dd_info = drawdown_table(
                sharpe=sharpe, ann_vol=ann_vol,
                current_drawdown=abs(curr_dd),
            )
            edd = _safe_fmt(dd_info["expected_dd_pct"], "{:.1%}")
            pdd = _safe_fmt(dd_info["dd_prob_10pct"], "{:.1%}")
            dd_rows.append(_build_table_rows("E[DD]", edd))
            dd_rows.append(_build_table_rows("P(DD≥10%)", pdd))
    else:
        dd_rows.append(_build_table_rows("—", "Need returns data"))

    # COL 4: Factors
    factor_rows: list[str] = []
    if port_rets is not None and port_rets.len() > 30 and has_returns:
        if benchmark in returns_df.columns:
            decomp = capm_regression(port_rets, returns_df[benchmark], risk_free_rate)
            factor_rows.append(_build_table_rows("Alpha", f"{decomp.alpha:+.2%}"))
            factor_rows.append(_build_table_rows("Beta (mkt)", f"{decomp.beta_market:.3f}"))
            factor_rows.append(_build_table_rows("Systematic", f"{decomp.systematic_pct:.0f}%"))
            factor_rows.append(_build_table_rows("Resid Vol", f"{decomp.residual_vol:.1%}"))

            factors = build_proxy_factors(returns_df)
            if len(factors) >= 2:
                try:
                    mf = multi_factor_regression(port_rets, factors, risk_free_rate)
                    tilts: list[tuple[str, float]] = []
                    if mf.beta_size is not None:
                        tilts.append(("Size", mf.beta_size))
                    if mf.beta_value is not None:
                        tilts.append(("Value", mf.beta_value))
                    if mf.beta_momentum is not None:
                        tilts.append(("Mom", mf.beta_momentum))
                    quality = mf.alpha - abs(mf.beta_market - 1.0) * 0.5
                    tilts.append(("Qual", quality))
                    tilts.sort(key=lambda x: abs(x[1]), reverse=True)
                    for name, val in tilts[:2]:
                        icon = "+" if val > 0 else ""
                        factor_rows.append(_build_table_rows(name, f"{icon}{val:.3f}"))
                except Exception:
                    pass
        else:
            factor_rows.append(_build_table_rows("—", "No benchmark data"))
    else:
        factor_rows.append(_build_table_rows("—", "Need 30+ days"))

    # COL 5: Correlation & Independence
    corr_rows: list[str] = []
    if has_returns:
        long_tickers = [p.ticker for p in portfolio.long_positions]
        short_tickers = [p.ticker for p in portfolio.short_positions]
        corr_stats = correlation_metrics.correlation_summary(
            returns_df, long_tickers, short_tickers,
        )
        corr_rows.append(_build_table_rows("Avg Pair", f"{corr_stats['avg_pairwise']:.3f}"))
        corr_rows.append(_build_table_rows("Long Corr", f"{corr_stats['avg_long_corr']:.3f}"))
        corr_rows.append(_build_table_rows("Short Corr", f"{corr_stats['avg_short_corr']:.3f}"))
        corr_rows.append(_build_table_rows("L/S Corr", f"{corr_stats['long_short_corr']:.3f}"))

        # Avg idiosyncratic share — what % of position variance is NOT market
        if benchmark in returns_df.columns:
            idio_shares = []
            for t in long_tickers + short_tickers:
                if t in returns_df.columns:
                    t_rets = returns_df[t]
                    if t_rets.len() > 30:
                        decomp = capm_regression(t_rets, returns_df[benchmark], risk_free_rate)
                        idio_shares.append(decomp.idiosyncratic_pct)
            if idio_shares:
                avg_idio = sum(idio_shares) / len(idio_shares)
                corr_rows.append(_build_table_rows("Idio %", f"{avg_idio:.0f}%"))
            else:
                corr_rows.append(_build_table_rows("Idio %", "—"))
        else:
            corr_rows.append(_build_table_rows("Idio %", "—"))
    else:
        corr_rows.append(_build_table_rows("—", "Need returns data"))

    # --- Find max row count to pad shorter columns ---
    all_cols = [summary_rows, risk_rows, dd_rows, factor_rows, corr_rows]
    max_rows = max(len(col) for col in all_cols)
    for col in all_cols:
        while len(col) < max_rows:
            col.append(
                '<tr><td style="padding:3px 8px;">&nbsp;</td>'
                '<td style="padding:3px 8px;">&nbsp;</td></tr>'
            )

    # --- Build the combined HTML table via components.html (iframe) ---
    col_headers = ["Summary", "Risk", "Drawdown", "Factors", "Correlation"]

    # Build inner cell HTML for each column
    cells_html = ""
    for i, col_rows in enumerate(all_cols):
        inner = ''.join(col_rows)
        cells_html += f'<td class="col-block"><table>{inner}</table></td>'

    # Header cells
    header_cells = ''.join(
        f'<th>{h}</th>' for h in col_headers
    )

    # Compute height: row count * ~36px per row + header + padding
    est_height = max_rows * 36 + 65

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: "Source Sans Pro", sans-serif;
    font-size: clamp(11px, 0.85vw, 15px);
    background: transparent;
    color: #e0e0e0;
  }}
  table.grid {{
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
  }}
  table.grid > thead > tr > th {{
    padding: 6px 8px;
    font-weight: 700;
    font-size: clamp(12px, 0.9vw, 17px);
    border-bottom: 2px solid #555;
    background: #1a1a2e;
    color: #e0e0e0;
    text-align: left;
  }}
  table.grid > tbody > tr > td.col-block {{
    border: 1px solid #333;
    border-top: none;
    vertical-align: top;
    padding: 0;
  }}
  .col-block table {{
    width: 100%;
    border-collapse: collapse;
  }}
  .col-block table td {{
    padding: 3px 8px;
    border-bottom: 1px solid #2a2a3a;
    white-space: nowrap;
  }}
  .col-block table tr:last-child td {{
    border-bottom: none;
  }}
  .col-block table td:first-child {{
    color: #999;
  }}
  .col-block table td:last-child {{
    font-weight: 600;
  }}
</style>
</head>
<body>
<table class="grid">
  <thead><tr>{header_cells}</tr></thead>
  <tbody><tr>{cells_html}</tr></tbody>
</table>
</body>
</html>"""

    components.html(html_doc, height=est_height, scrolling=False)

    st.caption(
        "**L/S Corr** — avg correlation between long and short books; "
        "high = both sides share similar systematic exposure, "
        "net P&L driven by stock selection. "
        " · **Idio %** — avg share of position return variance "
        "NOT explained by the market (1 − R² from CAPM); "
        "high = book is driven by stock-specific alpha, low = paying for beta."
    )

    # --- Alerts (full-width below the detail strip) ---
    alert_cfg = st.session_state.get("alerts", {})
    limit_kwargs = {
        "max_sector_net": alert_cfg.get("max_sector_net_exposure", 0.50),
        "max_subsector_net": alert_cfg.get("max_subsector_net_exposure", 0.50),
        "max_single_position": alert_cfg.get("max_single_position_weight", 0.10),
        "max_net_beta": alert_cfg.get("max_net_beta", 0.30),
        "min_net_beta": alert_cfg.get("min_net_beta", -0.10),
        "max_gross": alert_cfg.get("max_gross_exposure", 2.50),
    }
    warnings = exposure_metrics.check_exposure_limits(
        portfolio, betas=betas, **limit_kwargs,
    )

    max_ann_vol = alert_cfg.get("max_ann_vol")
    if max_ann_vol and has_returns:
        w = portfolio.weight_vector()
        pv = risk_metrics.portfolio_volatility(w, returns_df)
        if pv > max_ann_vol:
            warnings.append(f"Portfolio vol {pv:.1%} exceeds limit {max_ann_vol:.1%}")

    if warnings:
        with st.expander(f"⚠️ {len(warnings)} Alert(s)", expanded=False):
            for w in warnings:
                st.warning(w)


# ---------------------------------------------------------------------------
# Sector Exposure Chart (standalone, called from portfolio_view)
# ---------------------------------------------------------------------------

def _sector_annotation(
    sector: str,
    long_pct: float,
    short_pct: float,
    net_pct: float,
    position_count: int,
) -> str:
    """Generate a concise annotation explaining a sector's exposure."""
    direction = "net long" if net_pct > 0 else "net short" if net_pct < 0 else "flat"
    abs_net = abs(net_pct)

    if abs_net < 0.5:
        tilt = "Roughly hedged"
    elif abs_net < 2.0:
        tilt = f"Modest {direction}"
    elif abs_net < 4.0:
        tilt = f"Notable {direction}"
    else:
        tilt = f"Large {direction}"

    if long_pct > 0 and short_pct > 0:
        hedge_ratio = short_pct / long_pct * 100 if long_pct > 0 else 0
        hedge_note = f"{hedge_ratio:.0f}% hedged"
    elif long_pct > 0:
        hedge_note = "unhedged long"
    elif short_pct > 0:
        hedge_note = "pure short"
    else:
        hedge_note = "no exposure"

    return f"{tilt} ({hedge_note}, {position_count} names)"


def render_sector_exposure(portfolio: Portfolio) -> None:
    """Render the Sector Exposure bar chart with per-sector annotations.

    Called separately from render_metrics_detail so it can be placed
    below the portfolio table in the layout.
    """
    sector_exp = portfolio.sector_exposure()
    if not sector_exp:
        return

    st.subheader("📊 Sector Exposure")

    sorted_sectors = sorted(sector_exp.items(), key=lambda x: x[1]["net"])
    names = [s[0] for s in sorted_sectors]
    longs = [s[1]["long"] * 100 for s in sorted_sectors]
    shorts = [-s[1]["short"] * 100 for s in sorted_sectors]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=longs, orientation="h",
        name="Long", marker_color="#2ecc71",
        hovertemplate="%{y}: +%{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=names, x=shorts, orientation="h",
        name="Short", marker_color="#e74c3c",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        barmode="relative",
        height=max(len(names) * 34, 250),
        margin=dict(l=0, r=20, t=10, b=0),
        xaxis=dict(
            title=None, showgrid=True, zeroline=True,
            zerolinecolor="#888", zerolinewidth=1,
            ticksuffix="%", tickfont=dict(size=16),
        ),
        yaxis=dict(
            title=None, tickfont=dict(size=16),
            automargin=True,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=15),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key="sector_bar")

    # --- Per-sector annotations ---
    # Count positions per sector
    sector_counts: dict[str, int] = {}
    for p in portfolio.positions:
        sector_counts[p.sector] = sector_counts.get(p.sector, 0) + 1

    annotations = []
    for sector, exp in sorted_sectors:
        long_pct = exp["long"] * 100
        short_pct = exp["short"] * 100
        net_pct = exp["net"] * 100
        count = sector_counts.get(sector, 0)
        note = _sector_annotation(sector, long_pct, short_pct, net_pct, count)
        annotations.append(f"**{sector}:** {note}")

    st.markdown(
        " &nbsp;|&nbsp; ".join(annotations),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Legacy wrapper (for backward compatibility if called directly)
# ---------------------------------------------------------------------------

def render_metrics_panel(
    portfolio: Portfolio,
    returns_df: pl.DataFrame | None = None,
    betas: dict[str, float] | None = None,
    risk_free_rate: float = 0.05,
) -> None:
    """Legacy wrapper — renders both top bar and detail panel.

    Use render_top_metrics_bar() and render_metrics_detail() separately
    for the new layout.
    """
    render_top_metrics_bar(portfolio, returns_df, betas, risk_free_rate)
    st.divider()
    render_metrics_detail(portfolio, returns_df, betas, risk_free_rate)
