"""Chart gallery — Plotly visualizations for portfolio analytics.

All charts are interactive via Plotly.
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from core.metrics import correlation_metrics, drawdown_metrics, return_metrics, risk_metrics
from core.metrics.attribution import position_attribution, sector_attribution, side_attribution
from core.portfolio import Portfolio


def render_charts(
    portfolio: Portfolio,
    returns_df: pl.DataFrame | None = None,
    betas: dict[str, float] | None = None,
) -> None:
    """Render the chart gallery."""
    st.subheader("📊 Charts")

    if betas is None:
        betas = {}

    # Chart selector
    available_charts = [
        "Sector Exposure",
        "P&L Waterfall",
        "Sector P&L",
        "Beta Scatter",
        "Risk Contribution",
        "RSI Heatmap",
    ]
    if returns_df is not None and returns_df.height > 20:
        available_charts.extend([
            "Correlation Heatmap",
            "Dispersion",
            "Quality Score",
            "NAV Curve",
            "Drawdown",
            "Rolling Metrics",
        ])

    selected = st.multiselect(
        "Select charts to display",
        available_charts,
        default=["Sector Exposure", "Beta Scatter"],
        key="chart_selector",
    )

    if not selected:
        return

    # Render selected charts in a grid
    cols = st.columns(min(len(selected), 2))
    for i, chart_name in enumerate(selected):
        with cols[i % 2]:
            if chart_name == "Sector Exposure":
                _chart_sector_exposure(portfolio)
            elif chart_name == "P&L Waterfall":
                _chart_pnl_waterfall(portfolio)
            elif chart_name == "Sector P&L":
                _chart_sector_pnl(portfolio)
            elif chart_name == "Beta Scatter":
                _chart_beta_scatter(portfolio, betas)
            elif chart_name == "Risk Contribution" and returns_df is not None:
                _chart_risk_contribution(portfolio, returns_df)
            elif chart_name == "RSI Heatmap":
                _chart_rsi_heatmap(portfolio, returns_df)
            elif chart_name == "Correlation Heatmap" and returns_df is not None:
                _chart_correlation_heatmap(portfolio, returns_df)
            elif chart_name == "Dispersion" and returns_df is not None:
                _chart_dispersion(portfolio, returns_df)
            elif chart_name == "Quality Score" and returns_df is not None:
                _chart_quality_score(portfolio, returns_df, betas)
            elif chart_name == "NAV Curve" and returns_df is not None:
                _chart_nav_curve(portfolio, returns_df)
            elif chart_name == "Drawdown" and returns_df is not None:
                _chart_drawdown(portfolio, returns_df)
            elif chart_name == "Rolling Metrics" and returns_df is not None:
                _chart_rolling_metrics(portfolio, returns_df)


def _chart_sector_exposure(portfolio: Portfolio) -> None:
    """Stacked bar chart of long/short exposure by sector."""
    sector_exp = portfolio.sector_exposure()

    sectors = sorted(sector_exp.keys())
    longs = [sector_exp[s]["long"] * 100 for s in sectors]
    shorts = [-sector_exp[s]["short"] * 100 for s in sectors]
    nets = [sector_exp[s]["net"] * 100 for s in sectors]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Long", x=sectors, y=longs, marker_color="#2ecc71"))
    fig.add_trace(go.Bar(name="Short", x=sectors, y=shorts, marker_color="#e74c3c"))
    fig.add_trace(go.Scatter(
        name="Net", x=sectors, y=nets,
        mode="markers+lines", marker=dict(color="#3498db", size=10),
    ))

    fig.update_layout(
        title="Sector Exposure (%)",
        barmode="relative",
        height=400,
        yaxis_title="% of NAV",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_beta_scatter(portfolio: Portfolio, betas: dict[str, float]) -> None:
    """Scatter: x=beta, y=weight, color=sector, size=notional."""
    data = []
    for p in portfolio.positions:
        beta = betas.get(p.ticker, 1.0)
        weight = p.weight_in(portfolio.nav) * 100
        data.append({
            "Ticker": p.ticker,
            "Beta": beta,
            "Weight %": weight,
            "Sector": p.sector,
            "Side": p.side,
            "Notional ($K)": p.notional / 1000,
        })

    if not data:
        return

    df = pl.DataFrame(data).to_pandas()
    fig = px.scatter(
        df, x="Beta", y="Weight %",
        color="Sector", size="Notional ($K)",
        hover_data=["Ticker", "Side"],
        title="Position Beta vs Weight",
        height=400,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)


def _chart_risk_contribution(portfolio: Portfolio, returns_df: pl.DataFrame) -> None:
    """Waterfall chart of marginal contribution to risk."""
    weights = portfolio.weight_vector()
    mcr = risk_metrics.marginal_contribution_to_risk(weights, returns_df)

    if not mcr:
        st.info("Insufficient data for risk contribution chart.")
        return

    # Sort by absolute contribution
    sorted_mcr = sorted(mcr.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    tickers = [t for t, _ in sorted_mcr]
    values = [v * 100 for _, v in sorted_mcr]  # as percentage

    colors = ["#2ecc71" if v < 0 else "#e74c3c" for v in values]

    fig = go.Figure(go.Bar(
        x=tickers, y=values,
        marker_color=colors,
    ))
    fig.update_layout(
        title="Top 15 Risk Contributors (%)",
        yaxis_title="Marginal Contribution to Vol (%)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_rsi_heatmap(portfolio: Portfolio, returns_df: pl.DataFrame | None) -> None:
    """Bar chart of RSI values for all positions."""
    from core.metrics.technical_metrics import rsi_current

    rsi_period = st.session_state.settings.get("rsi_period", 14)
    data = []

    for p in portfolio.positions:
        if returns_df is not None and p.ticker in returns_df.columns:
            rets = returns_df[p.ticker]
            price_series = (1 + rets).cum_prod() * 100
            rsi_val = rsi_current(price_series, rsi_period)
        else:
            rsi_val = 50.0

        data.append({
            "Ticker": p.ticker,
            "RSI": rsi_val,
            "Side": p.side,
        })

    if not data:
        return

    df = pl.DataFrame(data).sort("RSI")

    colors = []
    for row in df.iter_rows(named=True):
        rsi = row["RSI"]
        if rsi >= 70:
            colors.append("#e74c3c")  # overbought
        elif rsi <= 30:
            colors.append("#2ecc71")  # oversold
        else:
            colors.append("#3498db")  # neutral

    fig = go.Figure(go.Bar(
        x=df["Ticker"].to_list(),
        y=df["RSI"].to_list(),
        marker_color=colors,
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.update_layout(title=f"RSI ({rsi_period})", height=400, yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)


def _chart_correlation_heatmap(portfolio: Portfolio, returns_df: pl.DataFrame) -> None:
    """Heatmap of pairwise correlations + most/least correlated tables."""
    tickers = [t for t in portfolio.tickers if t in returns_df.columns]
    if len(tickers) < 3:
        return

    corr = correlation_metrics.pairwise_correlation_matrix(returns_df, tickers)
    if corr.size == 0:
        return

    # --- Heatmap left, tables right ---
    hmap_col, table_col = st.columns([3, 2])

    with hmap_col:
        fig = go.Figure(go.Heatmap(
            z=corr, x=tickers, y=tickers,
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        ))
        fig.update_layout(title="Pairwise Correlation", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with table_col:
        # --- Pairwise: 3 most & 3 least correlated pairs ---
        most = correlation_metrics.most_correlated_pairs(returns_df, tickers, top_n=3)
        least = correlation_metrics.least_correlated_pairs(returns_df, tickers, top_n=3)

        st.markdown("**Most Correlated Pairs**")
        if most:
            rows = [{"Pair": f"{a} / {b}", "ρ": f"{c:+.3f}"} for a, b, c in most]
            st.dataframe(pl.DataFrame(rows), use_container_width=True, hide_index=True, height=140)

        st.markdown("**Least Correlated Pairs**")
        if least:
            rows = [{"Pair": f"{a} / {b}", "ρ": f"{c:+.3f}"} for a, b, c in least]
            st.dataframe(pl.DataFrame(rows), use_container_width=True, hide_index=True, height=140)

        # --- Correlation to SPX ---
        benchmark = st.session_state.get("settings", {}).get("benchmark", "SPY")
        if benchmark in returns_df.columns:
            spy_rets = returns_df[benchmark].to_numpy()
            spy_corrs: list[tuple[str, float]] = []
            for t in tickers:
                if t == benchmark:
                    continue
                t_rets = returns_df[t].to_numpy()
                min_len = min(len(spy_rets), len(t_rets))
                if min_len < 20:
                    continue
                c = float(np.corrcoef(t_rets[-min_len:], spy_rets[-min_len:])[0, 1])
                spy_corrs.append((t, c))

            if spy_corrs:
                spy_corrs.sort(key=lambda x: x[1], reverse=True)
                top3 = spy_corrs[:3]
                bot3 = spy_corrs[-3:]

                st.markdown(f"**Most Correlated to {benchmark}**")
                rows = [{"Ticker": t, "ρ": f"{c:+.3f}"} for t, c in top3]
                st.dataframe(
                    pl.DataFrame(rows),
                    use_container_width=True, hide_index=True, height=140,
                )

                st.markdown(f"**Least Correlated to {benchmark}**")
                rows = [{"Ticker": t, "ρ": f"{c:+.3f}"} for t, c in bot3]
                st.dataframe(
                    pl.DataFrame(rows),
                    use_container_width=True, hide_index=True, height=140,
                )


def _chart_nav_curve(portfolio: Portfolio, returns_df: pl.DataFrame) -> None:
    """Cumulative NAV curve."""
    weights = portfolio.weight_vector()
    port_rets = return_metrics.portfolio_daily_returns(weights, returns_df)

    if port_rets.len() < 5:
        return

    cum_ret = return_metrics.cumulative_return(port_rets)
    if "date" in returns_df.columns:
        dates = returns_df["date"].tail(cum_ret.len())
    else:
        dates = list(range(cum_ret.len()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates.to_list() if hasattr(dates, "to_list") else dates,
        y=(cum_ret * portfolio.nav).to_list(),
        mode="lines",
        name="Portfolio NAV",
        line=dict(color="#3498db"),
    ))
    fig.update_layout(title="NAV Curve (Historical)", height=400, yaxis_title="NAV ($)")
    st.plotly_chart(fig, use_container_width=True)


def _chart_drawdown(portfolio: Portfolio, returns_df: pl.DataFrame) -> None:
    """Underwater (drawdown) chart."""
    weights = portfolio.weight_vector()
    port_rets = return_metrics.portfolio_daily_returns(weights, returns_df)

    if port_rets.len() < 5:
        return

    cum_ret = return_metrics.cumulative_return(port_rets)
    dd = drawdown_metrics.drawdown_series(cum_ret)
    if "date" in returns_df.columns:
        dates = returns_df["date"].tail(dd.len())
    else:
        dates = list(range(dd.len()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates.to_list() if hasattr(dates, "to_list") else dates,
        y=(dd * 100).to_list(),
        fill="tozeroy",
        mode="lines",
        name="Drawdown",
        line=dict(color="#e74c3c"),
        fillcolor="rgba(231, 76, 60, 0.3)",
    ))
    fig.update_layout(title="Drawdown (%)", height=300, yaxis_title="Drawdown (%)")
    st.plotly_chart(fig, use_container_width=True)


def _chart_rolling_metrics(portfolio: Portfolio, returns_df: pl.DataFrame) -> None:
    """Rolling Sharpe and volatility."""
    weights = portfolio.weight_vector()
    port_rets = return_metrics.portfolio_daily_returns(weights, returns_df)

    if port_rets.len() < 63:
        st.info("Need at least 63 days for rolling metrics.")
        return

    rf = st.session_state.get("risk_free_rate_value", 0.05)
    rolling_sharpe = return_metrics.rolling_sharpe(port_rets, window=63, risk_free_rate=rf)

    if "date" in returns_df.columns:
        dates = returns_df["date"].tail(rolling_sharpe.len())
    else:
        dates = list(range(rolling_sharpe.len()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates.to_list() if hasattr(dates, "to_list") else dates,
        y=rolling_sharpe.to_list(),
        name="Rolling Sharpe (63d)",
        line=dict(color="#3498db"),
    ))
    fig.update_layout(title="Rolling Sharpe Ratio (63-day)", height=350)
    st.plotly_chart(fig, use_container_width=True)


def _chart_pnl_waterfall(portfolio: Portfolio) -> None:
    """P&L waterfall — top winners and losers by contribution to portfolio P&L."""
    attribs = position_attribution(portfolio)
    if not attribs:
        st.info("No positions for P&L waterfall.")
        return

    # Take top 10 winners + top 10 losers
    winners = [a for a in attribs if a.contribution_bps > 0]
    losers = [a for a in attribs if a.contribution_bps < 0]

    # Sort: losers ascending (worst first), winners descending
    losers.sort(key=lambda x: x.contribution_bps)
    winners.sort(key=lambda x: x.contribution_bps, reverse=True)

    # Interleave: top winners then top losers
    display = winners[:10] + losers[:10]
    # Sort for display: largest positive → largest negative
    display.sort(key=lambda x: x.contribution_bps, reverse=True)

    tickers = [a.ticker for a in display]
    bps_values = [a.contribution_bps for a in display]
    pnl_dollars = [a.pnl_dollars for a in display]

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in bps_values]

    # Build hover text
    hover = [
        f"{t}<br>P&L: ${pnl:+,.0f}<br>Contribution: {bps:+.1f} bps<br>{a.side} | {a.sector}"
        for t, pnl, bps, a in zip(tickers, pnl_dollars, bps_values, display)
    ]

    fig = go.Figure(go.Bar(
        x=tickers,
        y=bps_values,
        marker_color=colors,
        hovertext=hover,
        hoverinfo="text",
    ))

    fig.add_hline(y=0, line_color="gray", line_dash="solid", line_width=1)

    # Annotate total P&L
    total_pnl = portfolio.total_pnl_dollars
    total_bps = sum(a.contribution_bps for a in attribs)
    fig.update_layout(
        title=(
            f"P&L Waterfall — Top Winners & Losers "
            f"(Total: {total_bps:+.0f} bps / ${total_pnl:+,.0f})"
        ),
        yaxis_title="Contribution (bps)",
        height=450,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_sector_pnl(portfolio: Portfolio) -> None:
    """Sector P&L attribution — bar chart of P&L contribution by sector."""
    attribs = sector_attribution(portfolio)
    if not attribs:
        st.info("No sector data for P&L attribution.")
        return

    sectors = [a.sector for a in attribs]
    long_pnl = [a.long_pnl for a in attribs]
    short_pnl = [a.short_pnl for a in attribs]
    net_bps = [a.contribution_bps for a in attribs]

    nav = portfolio.nav

    fig = go.Figure()

    # Long P&L bars (green, positive)
    fig.add_trace(go.Bar(
        name="Long P&L",
        x=sectors,
        y=[(p / nav * 10_000) if nav else 0 for p in long_pnl],
        marker_color="#2ecc71",
    ))

    # Short P&L bars (red, can be positive or negative)
    fig.add_trace(go.Bar(
        name="Short P&L",
        x=sectors,
        y=[(p / nav * 10_000) if nav else 0 for p in short_pnl],
        marker_color="#e74c3c",
    ))

    # Net P&L line
    fig.add_trace(go.Scatter(
        name="Net",
        x=sectors,
        y=net_bps,
        mode="markers+lines",
        marker=dict(color="#3498db", size=10),
        line=dict(color="#3498db", width=2),
    ))

    # Side attribution summary
    sa = side_attribution(portfolio)
    long_total = (sa.long_pnl / nav * 10_000) if nav else 0
    short_total = (sa.short_pnl / nav * 10_000) if nav else 0

    fig.add_hline(y=0, line_color="gray", line_dash="solid", line_width=1)
    fig.update_layout(
        title=(
            f"Sector P&L Attribution (Long: {long_total:+.0f} bps | "
            f"Short: {short_total:+.0f} bps)"
        ),
        barmode="relative",
        yaxis_title="Contribution (bps)",
        height=450,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_dispersion(portfolio: Portfolio, returns_df: pl.DataFrame) -> None:
    """Rolling cross-sectional dispersion of position returns.

    Dispersion = daily cross-sectional standard deviation of individual
    position returns. High dispersion = more stock-picking opportunity;
    low dispersion = everything moving together (correlated market).
    """
    tickers = [t for t in portfolio.tickers if t in returns_df.columns]
    if len(tickers) < 5:
        st.info("Need at least 5 positions with return data for dispersion.")
        return

    mat = returns_df.select(tickers).to_numpy()  # (T, N)
    if mat.shape[0] < 30:
        st.info("Need at least 30 days for dispersion chart.")
        return

    # Daily cross-sectional std (dispersion)
    daily_disp = np.nanstd(mat, axis=1) * 100  # as percentage

    # 21-day rolling average for smoother line
    window = 21
    if len(daily_disp) >= window:
        kernel = np.ones(window) / window
        rolling_disp = np.convolve(daily_disp, kernel, mode="valid")
    else:
        rolling_disp = daily_disp

    if "date" in returns_df.columns:
        dates = returns_df["date"].tail(len(rolling_disp))
    else:
        dates = list(range(len(rolling_disp)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates.to_list() if hasattr(dates, "to_list") else dates,
        y=rolling_disp.tolist(),
        mode="lines",
        name="Dispersion (21d avg)",
        line=dict(color="#9b59b6", width=2),
        fill="tozeroy",
        fillcolor="rgba(155, 89, 182, 0.2)",
    ))

    avg_disp = float(np.mean(rolling_disp))
    fig.add_hline(y=avg_disp, line_dash="dash", line_color="#888",
                  annotation_text=f"Avg {avg_disp:.2f}%")

    fig.update_layout(
        title=f"Return Dispersion — Cross-Sectional Vol ({len(tickers)} names)",
        yaxis_title="Daily Dispersion (%)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_quality_score(
    portfolio: Portfolio,
    returns_df: pl.DataFrame,
    betas: dict[str, float] | None = None,
) -> None:
    """Quality Score radar chart + dimension breakdown."""
    from core.metrics.quality_score import compute_quality_score

    qs = st.session_state.get("quality_score")
    if qs is None:
        qs = compute_quality_score(portfolio, returns_df, betas)

    # --- Radar chart left, table right ---
    radar_col, detail_col = st.columns([3, 2])

    with radar_col:
        names = [d.name for d in qs.dimensions]
        scores = [d.score for d in qs.dimensions]
        # Close the radar polygon
        names_closed = names + [names[0]]
        scores_closed = scores + [scores[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores_closed,
            theta=names_closed,
            fill="toself",
            fillcolor="rgba(46, 204, 113, 0.25)",
            line=dict(color="#2ecc71", width=2),
            name="Portfolio",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=11)),
                angularaxis=dict(tickfont=dict(size=13)),
            ),
            title=f"Quality Score: {qs.total_score:.0f} ({qs.grade})",
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with detail_col:
        st.markdown(f"### {qs.total_score:.0f}/100 — Grade: **{qs.grade}**")
        for d in qs.dimensions:
            # Color-coded indicator
            if d.score >= 75:
                color = "🟢"
            elif d.score >= 50:
                color = "🟡"
            else:
                color = "🔴"
            st.markdown(
                f"{color} **{d.name}** — {d.score:.0f}/100  \n"
                f"<span style='color:#888;font-size:0.9em'>{d.detail}</span>",
                unsafe_allow_html=True,
            )
