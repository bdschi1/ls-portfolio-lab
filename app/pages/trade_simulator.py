"""Trade Simulator — What-if trade entry + impact preview.

Enter up to 10 trades, preview how all metrics change, then apply or discard.
"""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from app.state.session import (
    get_cache,
    get_portfolio,
    get_trade_log,
    is_paper_mode,
    set_portfolio,
)
from core.portfolio import ProposedTrade, TradeBasket
from core.trade_impact import TradeImpactResult, simulate_impact
from history.trade_log import TradeRecord


def render() -> None:
    """Render the trade simulator page."""
    st.header("🔄 Trade Simulator")

    portfolio = get_portfolio()
    if portfolio is None:
        st.info(
            "**No portfolio loaded.** Go to the **Portfolio** page and either "
            "upload a CSV/Excel file or click **Generate Mock Portfolio**. "
            "Once a portfolio is loaded, return here to model trades and preview their impact."
        )
        return

    st.caption(
        f"Current: {portfolio.long_count}L / {portfolio.short_count}S | "
        f"Gross: {portfolio.gross_exposure:.1%} | Net: {portfolio.net_exposure:+.1%}"
    )

    # --- Trade Input Form ---
    st.subheader("Propose Trades (max 10)")
    st.caption(
        "Enter ticker, action (BUY/SHORT/ADD/REDUCE/SELL/COVER/EXIT), "
        "and shares or dollar amount. Preview the impact on all portfolio metrics before applying."
    )

    num_trades = st.number_input(
        "Number of trades", min_value=1, max_value=10, value=1, key="num_trades"
    )

    trades: list[dict] = []

    for i in range(int(num_trades)):
        with st.container():
            cols = st.columns([2, 1.5, 2, 1.5, 1.5, 2])

            with cols[0]:
                ticker = st.text_input("Ticker", key=f"trade_ticker_{i}", placeholder="AAPL")

            with cols[1]:
                asset_type = st.selectbox(
                    "Type",
                    options=["EQUITY", "ETF", "OPTION"],
                    key=f"trade_asset_type_{i}",
                )

            with cols[2]:
                action = st.selectbox(
                    "Action",
                    options=["BUY", "SHORT", "ADD", "REDUCE", "SELL", "COVER", "EXIT"],
                    key=f"trade_action_{i}",
                )

            with cols[3]:
                shares = st.number_input(
                    "Shares", min_value=0, value=0, key=f"trade_shares_{i}",
                )

            with cols[4]:
                dollar = st.number_input(
                    "$ Amount", min_value=0.0, value=0.0, key=f"trade_dollar_{i}",
                    format="%.0f",
                )

            with cols[5]:
                notes = st.text_input("Notes", key=f"trade_notes_{i}", placeholder="Optional")

            # Option-specific fields (conditionally shown)
            option_kwargs: dict = {}
            if asset_type == "OPTION":
                opt_cols = st.columns([1.5, 1.5, 1.5, 1.5])
                with opt_cols[0]:
                    strike = st.number_input(
                        "Strike", min_value=0.0, value=0.0,
                        key=f"trade_strike_{i}", format="%.2f",
                    )
                with opt_cols[1]:
                    option_type = st.selectbox(
                        "Call/Put", options=["CALL", "PUT"],
                        key=f"trade_opttype_{i}",
                    )
                with opt_cols[2]:
                    expiry = st.date_input(
                        "Expiry",
                        value=date.today() + timedelta(days=30),
                        key=f"trade_expiry_{i}",
                    )
                with opt_cols[3]:
                    delta_val = st.number_input(
                        "Delta", min_value=-1.0, max_value=1.0, value=0.50,
                        key=f"trade_delta_{i}", format="%.2f",
                    )

                if strike > 0:
                    option_kwargs = {
                        "strike": strike,
                        "option_type": option_type,
                        "expiry": expiry,
                        "delta": delta_val if delta_val != 0 else None,
                    }

            if ticker.strip():
                trade_data = {
                    "ticker": ticker.strip().upper(),
                    "action": action,
                    "asset_type": asset_type,
                    "shares": shares if shares > 0 else None,
                    "dollar_amount": dollar if dollar > 0 else None,
                    "notes": notes,
                    **option_kwargs,
                }
                trades.append(trade_data)

    st.divider()

    # --- Preview Impact ---
    if st.button("📊 Preview Impact", type="primary", use_container_width=True):
        if not trades:
            st.warning("Enter at least one trade above.")
            return

        # Validate and build basket
        try:
            proposed = []
            for t in trades:
                if t["action"] == "EXIT":
                    proposed.append(ProposedTrade(
                        ticker=t["ticker"], action=t["action"],
                        shares=1,  # placeholder for EXIT
                        asset_type=t.get("asset_type", "EQUITY"),
                        notes=t.get("notes", ""),
                    ))
                elif t["shares"] is None and t["dollar_amount"] is None:
                    st.error(f"Trade for {t['ticker']}: provide shares or dollar amount.")
                    return
                else:
                    proposed.append(ProposedTrade(**t))

            basket = TradeBasket(trades=proposed)
        except Exception as e:
            st.error(f"Invalid trade input: {e}")
            return

        # Get current prices for trade tickers
        cache = get_cache()
        trade_tickers = [t.ticker for t in basket.trades]
        existing_tickers = portfolio.tickers
        all_needed = list(set(trade_tickers + existing_tickers))
        current_prices = cache.get_current_prices(all_needed)

        # Run simulation
        try:
            returns_df = st.session_state.get("returns_df")
            betas = st.session_state.get("betas", {})
            rf = st.session_state.get("risk_free_rate_value", 0.05)

            result = simulate_impact(
                portfolio=portfolio,
                basket=basket,
                current_prices=current_prices,
                returns_df=returns_df,
                betas=betas,
                risk_free_rate=rf,
            )

            st.session_state.impact_result = result
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            return

    # --- Display Impact Result ---
    result: TradeImpactResult | None = st.session_state.get("impact_result")
    if result is not None:
        st.subheader("Impact Preview")

        # Position changes
        if result.new_positions:
            st.success(f"🆕 New positions: {', '.join(result.new_positions)}")
        if result.removed_positions:
            st.info(f"❌ Removed positions: {', '.join(result.removed_positions)}")
        if result.modified_positions:
            st.info(f"✏️ Modified positions: {', '.join(result.modified_positions)}")

        # Metrics diff table
        st.markdown("**Metric Changes**")

        for diff in result.metric_diffs:
            # Skip tiny changes
            if abs(diff.change) < 0.0001 and diff.category != "exposure":
                continue

            # Color and icon
            if diff.improved is True:
                icon = "✅"
            elif diff.improved is False:
                icon = "⚠️"
            else:
                icon = "➖"

            # Format values based on category
            if diff.category == "exposure" and diff.name.startswith("Sector"):
                before_str = f"{diff.before:+.1%}"
                after_str = f"{diff.after:+.1%}"
                change_str = f"{diff.change:+.1%}"
            elif "Count" in diff.name:
                before_str = f"{diff.before:.0f}"
                after_str = f"{diff.after:.0f}"
                change_str = f"{diff.change:+.0f}"
            elif "HHI" in diff.name:
                before_str = f"{diff.before:.4f}"
                after_str = f"{diff.after:.4f}"
                change_str = f"{diff.change:+.4f}"
            else:
                before_str = f"{diff.before:.3f}"
                after_str = f"{diff.after:.3f}"
                change_str = f"{diff.change:+.3f}"

            cols = st.columns([3, 1.5, 1.5, 1.5, 0.5])
            with cols[0]:
                st.text(diff.name)
            with cols[1]:
                st.text(before_str)
            with cols[2]:
                st.text(after_str)
            with cols[3]:
                st.text(f"{change_str} {diff.direction_str}")
            with cols[4]:
                st.text(icon)

        # Warnings
        if result.warnings:
            st.divider()
            st.markdown("**⚠️ Limit Warnings**")
            for w in result.warnings:
                st.warning(w)

        # Apply / Discard buttons
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Apply Trades", type="primary", use_container_width=True):
                # Apply to portfolio
                set_portfolio(result.portfolio_after)

                # Log trades if in paper mode
                if is_paper_mode():
                    trade_log = get_trade_log()
                    for trade in result.proposed_trades:
                        cache = get_cache()
                        prices = cache.get_current_prices([trade.ticker])
                        price = prices.get(trade.ticker, 0.0)
                        shares = trade.shares or (
                            trade.dollar_amount / price if trade.dollar_amount and price > 0 else 0
                        )

                        after_pos = result.portfolio_after.get_position(trade.ticker)
                        record = TradeRecord(
                            ticker=trade.ticker,
                            action=trade.action,
                            shares=shares,
                            price=price,
                            notional=shares * price,
                            asset_type=trade.asset_type,
                            side_after=after_pos.side if after_pos else None,
                            shares_after=after_pos.shares if after_pos else 0,
                            portfolio_nav_after=result.portfolio_after.nav,
                            sector=after_pos.sector if after_pos else "",
                            subsector=after_pos.subsector if after_pos else "",
                            notes=trade.notes,
                        )
                        trade_log.append(record)

                    st.success("Trades applied and logged to paper portfolio ✅")
                else:
                    st.success("Trades applied ✅")

                # Clear impact result
                st.session_state.impact_result = None
                st.rerun()

        with col2:
            if st.button("❌ Discard", use_container_width=True):
                st.session_state.impact_result = None
                st.rerun()
