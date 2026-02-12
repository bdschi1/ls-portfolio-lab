"""PM Scorecard — Comprehensive PM performance analytics.

The full PM evaluation dashboard. Requires paper mode with sufficient history.
"""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from app.state.session import get_snapshot_store, get_trade_log, is_paper_mode
from core.metrics.pm_performance import PMScorecard
from history.performance import generate_pm_scorecard


def render() -> None:
    """Render the PM scorecard page."""
    st.header("🏆 PM Scorecard")

    if not is_paper_mode():
        st.info(
            "**Paper Mode is OFF.** The PM Scorecard requires trade history to compute "
            "hit rate, slugging %, and sector attribution. Toggle **📝 Paper Portfolio Mode** "
            "ON in the sidebar, then apply trades from the Trade Simulator."
        )
        return

    trade_log = get_trade_log()
    snapshot_store = get_snapshot_store()

    if trade_log.trade_count == 0:
        st.info(
            "**No trades recorded yet.** Go to the **Trade Simulator**, enter trades, "
            "click **Preview Impact**, then **Apply Trades** (with Paper Mode ON). "
            "The scorecard populates automatically once trades are logged."
        )
        return

    # Get market returns for alpha calculation
    returns_df = st.session_state.get("returns_df")
    benchmark = st.session_state.settings.get("benchmark", "SPY")
    market_returns = None
    if returns_df is not None and benchmark in returns_df.columns:
        market_returns = returns_df[benchmark]

    rf = st.session_state.get("risk_free_rate_value", 0.05)

    # Generate scorecard
    with st.spinner("Computing PM scorecard..."):
        scorecard = generate_pm_scorecard(
            snapshot_store=snapshot_store,
            trade_log=trade_log,
            market_returns=market_returns,
            risk_free_rate=rf,
        )

    # --- Headline Numbers ---
    st.subheader("📊 Headline Numbers")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Return", f"{scorecard.total_return_pct:+.1f}%")
        st.metric("Ann. Return", f"{scorecard.annualized_return_pct:+.1f}%")

    with col2:
        st.metric("Sharpe", f"{scorecard.sharpe:.2f}")
        st.metric("Alpha", f"{scorecard.alpha:+.1f}%")

    with col3:
        st.metric("Hit Rate", f"{scorecard.hit_rate:.0%}")
        st.metric("⭐ Slugging %", f"{scorecard.slugging_pct:.2f}x")

    with col4:
        st.metric("EV / Trade", f"${scorecard.expected_value_per_trade:+,.0f}")
        st.metric("Total Trades", scorecard.total_trades)

    # Slugging context
    if scorecard.slugging_pct > 0:
        if scorecard.slugging_pct >= 3.0:
            st.success("⭐ Elite slugging (≥3.0x) — winners significantly larger than losers")
        elif scorecard.slugging_pct >= 2.0:
            st.success("✅ Strong slugging (≥2.0x) — good risk management")
        elif scorecard.slugging_pct >= 1.5:
            st.info("📊 Solid slugging (≥1.5x) — above average")
        elif scorecard.slugging_pct >= 1.0:
            st.warning("⚠️ Mediocre slugging (1.0-1.5x) — winners barely exceed losers")
        else:
            st.error("🔴 Poor slugging (<1.0x) — losers bigger than winners")

    st.divider()

    # --- Trading Stats ---
    st.subheader("📈 Trading Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Wins & Losses**")
        st.text(f"Winners: {scorecard.total_winners}")
        st.text(f"Losers: {scorecard.total_losers}")
        st.text(f"Win/Loss $: {scorecard.win_loss_ratio:.2f}x")
        st.text(f"Avg Win: ${scorecard.avg_win_dollars:+,.0f} ({scorecard.avg_win_pct:+.1%})")
        st.text(f"Avg Loss: ${scorecard.avg_loss_dollars:,.0f} ({scorecard.avg_loss_pct:.1%})")

    with col2:
        st.markdown("**Best & Worst**")
        st.text(f"Best: {scorecard.best_trade_ticker} (${scorecard.best_trade_pnl:+,.0f})")
        st.text(f"Worst: {scorecard.worst_trade_ticker} (${scorecard.worst_trade_pnl:+,.0f})")
        st.text(f"Avg Hold: {scorecard.avg_holding_period_days:.0f} days")
        st.text(f"Trades/Month: {scorecard.trades_per_month:.1f}")
        st.text(f"Turnover: {scorecard.turnover_pct:.0f}%")

    with col3:
        st.markdown("**Long vs Short**")
        st.text(f"Long Hit Rate: {scorecard.long_hit_rate:.0%}")
        st.text(f"Long Slugging: {scorecard.long_slugging:.2f}x")
        st.text(f"Short Hit Rate: {scorecard.short_hit_rate:.0%}")
        st.text(f"Short Slugging: {scorecard.short_slugging:.2f}x")

    st.divider()

    # --- Sector Skill ---
    st.subheader("🏭 Sector Skill")

    if scorecard.sector_stats:
        rows = []
        for sector, stats in sorted(
            scorecard.sector_stats.items(),
            key=lambda x: x[1].get("total_pnl", 0),
            reverse=True,
        ):
            rows.append({
                "Sector": sector,
                "Hit Rate": f"{stats['hit_rate']:.0%}",
                "Slugging": f"{stats['slugging']:.2f}x",
                "Total P&L": f"${stats['total_pnl']:+,.0f}",
                "Avg P&L %": f"{stats['avg_pnl_pct']:.1%}",
                "# Trades": f"{stats['num_trades']:.0f}",
            })

        st.dataframe(
            pl.DataFrame(rows),
            use_container_width=True,
            height=min(35 * len(rows) + 50, 400),
        )

        # Sector P&L bar chart
        sectors = [r["Sector"] for r in rows]
        pnls = [scorecard.sector_stats[s]["total_pnl"] for s in sectors]
        colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in pnls]

        fig = go.Figure(go.Bar(x=sectors, y=pnls, marker_color=colors))
        fig.update_layout(
            title="P&L by Sector ($)",
            yaxis_title="Total P&L ($)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "**No sector data yet.** Sector skill breakdown appears after you "
            "close trades (SELL, COVER, or EXIT) across multiple sectors."
        )

    st.divider()

    # --- Drawdown Analysis ---
    st.subheader("📉 Drawdown Behavior")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Max Drawdown", f"{scorecard.max_drawdown:.1f}%")
    with col2:
        st.metric("Avg DD Depth", f"{scorecard.avg_drawdown_depth:.1f}%")
    with col3:
        st.metric("Avg Recovery", f"{scorecard.avg_recovery_days:.0f} days")
    with col4:
        st.metric("# Drawdowns", scorecard.num_drawdowns)

    # NAV curve with drawdown shading from snapshots
    snapshots = snapshot_store.read_all()
    if len(snapshots) >= 3:
        dates = [s.date for s in snapshots]
        navs = [s.nav for s in snapshots]

        # Compute drawdown from NAV
        peak = navs[0]
        dd_pcts = []
        for nav in navs:
            peak = max(peak, nav)
            dd_pcts.append((nav - peak) / peak * 100)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=navs,
            name="NAV",
            line=dict(color="#3498db", width=2),
        ))
        fig.update_layout(title="NAV Curve", height=400, yaxis_title="NAV ($)")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dates, y=dd_pcts,
            fill="tozeroy",
            name="Drawdown %",
            line=dict(color="#e74c3c"),
            fillcolor="rgba(231, 76, 60, 0.3)",
        ))
        fig2.update_layout(title="Drawdown (%)", height=300, yaxis_title="%")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # --- Key Takeaways ---
    st.subheader("🔑 Key Takeaways")

    takeaways = []

    if scorecard.slugging_pct >= 2.0 and scorecard.hit_rate >= 0.45:
        takeaways.append("✅ Strong edge: good hit rate with excellent slugging — asymmetric P&L profile")
    elif scorecard.slugging_pct >= 2.0:
        takeaways.append("✅ Great slugging overcomes modest hit rate — winners are much larger than losers")
    elif scorecard.hit_rate >= 0.55:
        takeaways.append("⚠️ High hit rate but needs better slugging — winners should be larger than losers")

    if scorecard.long_slugging > scorecard.short_slugging * 1.5:
        takeaways.append("📊 Stronger on the long side — consider reducing short book complexity")
    elif scorecard.short_slugging > scorecard.long_slugging * 1.5:
        takeaways.append("📊 Stronger on the short side — short book adding significant alpha")

    if scorecard.sector_stats:
        best_sector = max(scorecard.sector_stats.items(), key=lambda x: x[1].get("total_pnl", 0))
        worst_sector = min(scorecard.sector_stats.items(), key=lambda x: x[1].get("total_pnl", 0))
        takeaways.append(f"🏆 Best sector: {best_sector[0]} (${best_sector[1]['total_pnl']:+,.0f})")
        if worst_sector[1]["total_pnl"] < 0:
            takeaways.append(f"⚠️ Worst sector: {worst_sector[0]} (${worst_sector[1]['total_pnl']:+,.0f})")

    if scorecard.avg_recovery_days > 30:
        takeaways.append("⚠️ Slow drawdown recovery — consider tighter stop-loss discipline")

    if not takeaways:
        takeaways.append("📊 Insufficient data for meaningful takeaways. Keep trading!")

    for t in takeaways:
        st.markdown(t)
