"""Paper Portfolio — History view of tracked trades and portfolio state.

Shows trade journal, daily snapshots, NAV curve, and exposure evolution.
Only available when paper mode is ON.
"""

from __future__ import annotations

from datetime import datetime

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from app.state.session import (
    get_portfolio,
    get_snapshot_store,
    get_trade_log,
    is_paper_mode,
)
from history.snapshot import DailySnapshot, PositionSnapshot


def _take_snapshot() -> None:
    """Take a daily snapshot of the current portfolio state."""
    portfolio = get_portfolio()
    if portfolio is None:
        return

    snapshot_store = get_snapshot_store()
    betas = st.session_state.get("betas", {})

    # Build position snapshots
    pos_snaps = []
    for p in portfolio.positions:
        pos_snaps.append(PositionSnapshot(
            ticker=p.ticker,
            side=p.side,
            shares=p.shares,
            price=p.current_price,
            notional=p.notional,
            weight=p.weight_in(portfolio.nav),
            pnl_dollars=p.pnl_dollars,
            pnl_pct=p.pnl_pct,
            beta=betas.get(p.ticker, 1.0),
            sector=p.sector,
            subsector=p.subsector,
            asset_type=p.asset_type,
        ))

    # Sector exposures
    sector_exp = portfolio.sector_exposure()
    sector_nets = {s: exp["net"] for s, exp in sector_exp.items()}

    from core.metrics import exposure_metrics
    net_beta = exposure_metrics.net_beta_exposure(portfolio, betas) if betas else 0.0

    snapshot = DailySnapshot(
        date=datetime.now().date(),
        nav=portfolio.nav + portfolio.total_pnl_dollars,
        cash=portfolio.cash or 0.0,
        gross_exposure=portfolio.gross_exposure,
        net_exposure=portfolio.net_exposure,
        net_beta=net_beta,
        long_count=portfolio.long_count,
        short_count=portfolio.short_count,
        total_pnl_dollars=portfolio.total_pnl_dollars,
        total_pnl_pct=portfolio.total_pnl_pct,
        positions=pos_snaps,
        sector_exposures=sector_nets,
    )

    snapshot_store.save_snapshot(snapshot)


def render() -> None:
    """Render the paper portfolio page."""
    st.header("📝 Paper Portfolio")

    if not is_paper_mode():
        st.info(
            "**Paper Mode is OFF.** Toggle **📝 Paper Portfolio Mode** ON in the sidebar "
            "to start tracking trades and building a performance record. "
            "Once enabled, trades applied from the Trade Simulator will be logged here."
        )
        return

    portfolio = get_portfolio()
    if portfolio is None:
        st.info(
            "**No portfolio loaded.** Go to the **Portfolio** page to load or generate one, "
            "then return here to start tracking trades and snapshots."
        )
        return

    # Top summary row — including Slugging % prominently
    trade_log = get_trade_log()
    closed = trade_log.identify_closed_trades()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📸 Take Daily Snapshot", use_container_width=True):
            _take_snapshot()
            st.success("Snapshot taken!")

    with col2:
        snap_count = get_snapshot_store().snapshot_count
        st.metric("Snapshots", snap_count)

    with col3:
        trade_count = trade_log.trade_count
        st.metric("Trades Logged", trade_count)

    with col4:
        # Promote Slugging % to top row
        if closed:
            winners = [t for t in closed if t["pnl_dollars"] > 0]
            losers = [t for t in closed if t["pnl_dollars"] <= 0]
            avg_win = sum(t["pnl_dollars"] for t in winners) / len(winners) if winners else 0
            avg_loss = sum(abs(t["pnl_dollars"]) for t in losers) / len(losers) if losers else 1
            slugging = avg_win / avg_loss if avg_loss > 0 else 0
            st.metric("Slugging %", f"{slugging:.2f}x")
        else:
            st.metric("Slugging %", "—")

    st.caption(
        "Slugging % = avg win size / avg loss size. "
        ">1.0x means your winners are larger than your losers on average."
    )

    st.divider()

    # --- Trade Journal ---
    st.subheader("📖 Trade Journal")
    st.caption(
        "Most recent 50 trades, newest first. "
        "Trades are logged when applied from the Trade Simulator."
    )
    records = trade_log.read_all()

    if records:
        # Show last 50 trades
        display_records = list(reversed(records[-50:]))

        rows = []
        for r in display_records:
            rows.append({
                "Date": r.timestamp.strftime("%Y-%m-%d %H:%M"),
                "Ticker": r.ticker,
                "Action": r.action,
                "Shares": f"{r.shares:,.0f}",
                "Price": f"${r.price:,.2f}",
                "Notional": f"${r.notional:,.0f}",
                "After": f"{r.side_after or 'CLOSED'} ({r.shares_after:,.0f})",
                "Notes": r.notes,
            })

        st.dataframe(
            pl.DataFrame(rows),
            use_container_width=True,
            height=min(35 * len(rows) + 50, 500),
        )
    else:
        st.info(
            "**No trades logged yet.** Go to the **Trade Simulator** page, "
            "enter a trade, click **Preview Impact**, then **Apply Trades**. "
            "Applied trades will appear here automatically."
        )

    st.divider()

    # --- Snapshot History ---
    st.subheader("📊 Portfolio History")
    snapshot_store = get_snapshot_store()
    snapshots = snapshot_store.read_all()

    if len(snapshots) >= 2:
        # NAV Curve
        dates = [s.date for s in snapshots]
        navs = [s.nav for s in snapshots]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=navs,
            mode="lines+markers",
            name="NAV",
            line=dict(color="#3498db", width=2),
        ))
        fig.update_layout(
            title="Portfolio NAV Over Time",
            yaxis_title="NAV ($)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Exposure Evolution
        gross_exp = [s.gross_exposure * 100 for s in snapshots]
        net_exp = [s.net_exposure * 100 for s in snapshots]
        net_betas = [s.net_beta for s in snapshots]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dates, y=gross_exp, name="Gross %", line=dict(color="#e74c3c")))
        fig2.add_trace(go.Scatter(x=dates, y=net_exp, name="Net %", line=dict(color="#3498db")))
        fig2.update_layout(title="Exposure Over Time", height=350, yaxis_title="%")
        st.plotly_chart(fig2, use_container_width=True)

        # Beta evolution
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=dates, y=net_betas, name="Net Beta",
            line=dict(color="#9b59b6"),
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="gray")
        fig3.update_layout(title="Net Beta Over Time", height=300)
        st.plotly_chart(fig3, use_container_width=True)

    elif len(snapshots) == 1:
        st.info(
            "**One snapshot recorded.** Take at least one more on a different day "
            "to see NAV and exposure history charts."
        )
    else:
        st.info(
            "**No snapshots yet.** Click **📸 Take Daily Snapshot** above to "
            "capture today's NAV, exposure, and positions. Take snapshots daily "
            "to build a history for NAV and exposure charts."
        )

    # --- Closed Trades Summary ---
    st.divider()
    st.subheader("📊 Closed Trade Summary")
    st.caption(
        "Round-trip trades (opened and closed). "
        "Hit Rate = % winners. Slugging = avg win / avg loss."
    )

    if closed:
        winners = [t for t in closed if t["pnl_dollars"] > 0]
        losers = [t for t in closed if t["pnl_dollars"] <= 0]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Closed Trades", len(closed))
        with col2:
            hit_rate = len(winners) / len(closed) * 100 if closed else 0
            st.metric("Hit Rate", f"{hit_rate:.0f}%")
        with col3:
            avg_win = sum(t["pnl_dollars"] for t in winners) / len(winners) if winners else 0
            avg_loss = sum(abs(t["pnl_dollars"]) for t in losers) / len(losers) if losers else 1
            slugging_closed = avg_win / avg_loss if avg_loss > 0 else 0
            st.metric("Slugging %", f"{slugging_closed:.2f}x")
        with col4:
            total_pnl = sum(t["pnl_dollars"] for t in closed)
            st.metric("Total P&L", f"${total_pnl:+,.0f}")
    else:
        st.info(
            "**No closed trades yet.** Closed trades appear when you fully exit a position "
            "(SELL, COVER, or EXIT in the Trade Simulator). Hit rate and slugging "
            "are computed from round-trip trades only."
        )
