"""Trade impact simulator: apply proposed trades and diff all metrics.

The core "what-if" engine. Takes current portfolio + proposed trades,
creates a hypothetical portfolio, recalculates all metrics, and returns
a structured diff showing exactly what changes and by how much.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl

from core.metrics import (
    correlation_metrics,
    drawdown_metrics,
    exposure_metrics,
    return_metrics,
    risk_metrics,
    technical_metrics,
)
from core.portfolio import Portfolio, Position, ProposedTrade, TradeBasket


@dataclass
class MetricDiff:
    """Before/after comparison for a single metric."""

    name: str
    before: float
    after: float
    change: float
    change_pct: float | None  # None if before was 0
    improved: bool | None  # True=better, False=worse, None=neutral/context-dependent
    category: str = ""  # "risk", "exposure", "return", "correlation"

    @property
    def direction_str(self) -> str:
        if self.change > 0:
            return "▲"
        elif self.change < 0:
            return "▼"
        return "─"


@dataclass
class TradeImpactResult:
    """Complete result of a trade impact simulation."""

    proposed_trades: list[ProposedTrade]
    portfolio_before: Portfolio
    portfolio_after: Portfolio
    metric_diffs: list[MetricDiff] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    new_positions: list[str] = field(default_factory=list)
    removed_positions: list[str] = field(default_factory=list)
    modified_positions: list[str] = field(default_factory=list)


def apply_trades(
    portfolio: Portfolio,
    basket: TradeBasket,
    current_prices: dict[str, float],
) -> Portfolio:
    """
    Apply a basket of proposed trades to create a new portfolio state.

    Does NOT mutate the original portfolio — returns a new one.
    """
    new_portfolio = portfolio.model_copy(deep=True)

    for trade in basket.trades:
        ticker = trade.ticker.upper()
        price = current_prices.get(ticker, 0.0)
        if price == 0:
            msg = f"No price available for {ticker}. Cannot simulate trade."
            raise ValueError(msg)

        # Resolve shares from dollar amount if needed
        shares = trade.shares
        if shares is None and trade.dollar_amount is not None:
            shares = trade.dollar_amount / price

        existing = new_portfolio.get_position(ticker)

        if trade.action == "EXIT":
            if existing is None:
                msg = f"Cannot EXIT {ticker} — not in portfolio."
                raise ValueError(msg)
            new_portfolio = new_portfolio.remove_position(ticker)

        elif trade.action in ("BUY", "ADD"):
            if existing is not None:
                # Add to existing long position
                if existing.side != "LONG":
                    msg = f"Cannot BUY/ADD {ticker} — currently SHORT. Use COVER first."
                    raise ValueError(msg)
                new_shares = existing.shares + (shares or 0)
                # Weighted average entry price
                new_entry = (
                    (existing.shares * existing.entry_price + (shares or 0) * price)
                    / new_shares
                )
                new_portfolio = new_portfolio.update_position(
                    ticker,
                    shares=new_shares,
                    entry_price=new_entry,
                    current_price=price,
                )
            else:
                # New long position
                from data.sector_map import classify_ticker
                sector, subsector = classify_ticker(ticker)
                pos_kwargs: dict = dict(
                    ticker=ticker,
                    side="LONG",
                    shares=shares or 0,
                    entry_price=price,
                    entry_date=date.today(),
                    current_price=price,
                    sector=sector,
                    subsector=subsector,
                    asset_type=trade.asset_type,
                )
                # Propagate option fields
                if trade.asset_type == "OPTION":
                    pos_kwargs.update(
                        strike=trade.strike,
                        expiry=trade.expiry,
                        option_type=trade.option_type,
                        contract_multiplier=trade.contract_multiplier,
                        delta=trade.delta,
                    )
                new_position = Position(**pos_kwargs)
                new_portfolio = new_portfolio.add_position(new_position)

        elif trade.action in ("SHORT",):
            # Options cannot be shorted
            if trade.asset_type == "OPTION":
                msg = f"Cannot SHORT options. Use BUY/SELL for {ticker}."
                raise ValueError(msg)

            if existing is not None:
                if existing.side != "SHORT":
                    msg = f"Cannot SHORT {ticker} — currently LONG. Use SELL first."
                    raise ValueError(msg)
                new_shares = existing.shares + (shares or 0)
                new_entry = (
                    (existing.shares * existing.entry_price + (shares or 0) * price)
                    / new_shares
                )
                new_portfolio = new_portfolio.update_position(
                    ticker,
                    shares=new_shares,
                    entry_price=new_entry,
                    current_price=price,
                )
            else:
                from data.sector_map import classify_ticker
                sector, subsector = classify_ticker(ticker)
                new_position = Position(
                    ticker=ticker,
                    side="SHORT",
                    shares=shares or 0,
                    entry_price=price,
                    entry_date=date.today(),
                    current_price=price,
                    sector=sector,
                    subsector=subsector,
                    asset_type=trade.asset_type,
                )
                new_portfolio = new_portfolio.add_position(new_position)

        elif trade.action in ("SELL", "REDUCE"):
            if existing is None:
                msg = f"Cannot SELL/REDUCE {ticker} — not in portfolio."
                raise ValueError(msg)
            if existing.side != "LONG":
                msg = f"Cannot SELL/REDUCE {ticker} — it's a SHORT. Use COVER."
                raise ValueError(msg)
            remaining = existing.shares - (shares or 0)
            if remaining <= 0:
                new_portfolio = new_portfolio.remove_position(ticker)
            else:
                new_portfolio = new_portfolio.update_position(
                    ticker, shares=remaining, current_price=price,
                )

        elif trade.action == "COVER":
            if existing is None:
                msg = f"Cannot COVER {ticker} — not in portfolio."
                raise ValueError(msg)
            if existing.side != "SHORT":
                msg = f"Cannot COVER {ticker} — it's a LONG. Use SELL."
                raise ValueError(msg)
            remaining = existing.shares - (shares or 0)
            if remaining <= 0:
                new_portfolio = new_portfolio.remove_position(ticker)
            else:
                new_portfolio = new_portfolio.update_position(
                    ticker, shares=remaining, current_price=price,
                )

    return new_portfolio


def _make_diff(
    name: str,
    before: float,
    after: float,
    category: str,
    lower_is_better: bool = False,
) -> MetricDiff:
    """Create a MetricDiff with improvement detection."""
    change = after - before
    change_pct = (change / abs(before) * 100) if before != 0 else None

    if abs(change) < 1e-10:
        improved = None
    elif lower_is_better:
        improved = change < 0
    else:
        improved = change > 0

    return MetricDiff(
        name=name,
        before=before,
        after=after,
        change=change,
        change_pct=change_pct,
        improved=improved,
        category=category,
    )


def simulate_impact(
    portfolio: Portfolio,
    basket: TradeBasket,
    current_prices: dict[str, float],
    returns_df: pl.DataFrame,
    market_returns: pl.Series | None = None,
    betas: dict[str, float] | None = None,
    risk_free_rate: float = 0.05,
) -> TradeImpactResult:
    """
    Full trade impact simulation.

    Takes the current portfolio and proposed trades, computes all metrics
    before and after, and returns a structured diff.
    """
    # Apply trades
    portfolio_after = apply_trades(portfolio, basket, current_prices)

    # Track position changes
    before_tickers = set(portfolio.tickers)
    after_tickers = set(portfolio_after.tickers)
    new_positions = list(after_tickers - before_tickers)
    removed_positions = list(before_tickers - after_tickers)
    modified_positions = [
        t for t in (before_tickers & after_tickers)
        if portfolio.get_position(t) != portfolio_after.get_position(t)
    ]

    # Compute metrics before and after
    diffs: list[MetricDiff] = []
    warnings: list[str] = []

    # --- Exposure metrics ---
    diffs.append(_make_diff(
        "Gross Exposure", portfolio.gross_exposure, portfolio_after.gross_exposure,
        "exposure", lower_is_better=False,
    ))
    diffs.append(_make_diff(
        "Net Exposure", portfolio.net_exposure, portfolio_after.net_exposure,
        "exposure",
    ))
    diffs.append(_make_diff(
        "Long Count", float(portfolio.long_count), float(portfolio_after.long_count),
        "exposure",
    ))
    diffs.append(_make_diff(
        "Short Count", float(portfolio.short_count), float(portfolio_after.short_count),
        "exposure",
    ))
    diffs.append(_make_diff(
        "HHI Concentration",
        exposure_metrics.concentration_hhi(portfolio),
        exposure_metrics.concentration_hhi(portfolio_after),
        "exposure", lower_is_better=True,
    ))

    # --- Beta ---
    if betas:
        before_beta = exposure_metrics.net_beta_exposure(portfolio, betas)
        after_beta = exposure_metrics.net_beta_exposure(portfolio_after, betas)
        diffs.append(_make_diff("Net Beta", before_beta, after_beta, "risk"))

    # --- Portfolio volatility ---
    w_before = portfolio.weight_vector()
    w_after = portfolio_after.weight_vector()

    vol_before = risk_metrics.portfolio_volatility(w_before, returns_df)
    vol_after = risk_metrics.portfolio_volatility(w_after, returns_df)
    diffs.append(_make_diff("Portfolio Vol", vol_before, vol_after, "risk", lower_is_better=True))

    # --- Sharpe (historical estimate) ---
    port_rets_before = return_metrics.portfolio_daily_returns(w_before, returns_df)
    port_rets_after = return_metrics.portfolio_daily_returns(w_after, returns_df)

    if port_rets_before.len() > 20 and port_rets_after.len() > 20:
        sharpe_before = return_metrics.sharpe_ratio(port_rets_before, risk_free_rate)
        sharpe_after = return_metrics.sharpe_ratio(port_rets_after, risk_free_rate)
        diffs.append(_make_diff("Sharpe (hist.)", sharpe_before, sharpe_after, "return"))

        sortino_before = return_metrics.sortino_ratio(port_rets_before, risk_free_rate)
        sortino_after = return_metrics.sortino_ratio(port_rets_after, risk_free_rate)
        diffs.append(_make_diff("Sortino (hist.)", sortino_before, sortino_after, "return"))

    # --- VaR ---
    if port_rets_before.len() > 20 and port_rets_after.len() > 20:
        var_before = risk_metrics.var_historical(port_rets_before)
        var_after = risk_metrics.var_historical(port_rets_after)
        diffs.append(_make_diff("VaR (95%)", var_before, var_after, "risk", lower_is_better=True))

        cvar_before = risk_metrics.cvar_historical(port_rets_before)
        cvar_after = risk_metrics.cvar_historical(port_rets_after)
        diffs.append(_make_diff("CVaR (95%)", cvar_before, cvar_after, "risk", lower_is_better=True))

    # --- Correlation ---
    long_tickers_before = [p.ticker for p in portfolio.long_positions]
    short_tickers_before = [p.ticker for p in portfolio.short_positions]
    long_tickers_after = [p.ticker for p in portfolio_after.long_positions]
    short_tickers_after = [p.ticker for p in portfolio_after.short_positions]

    corr_before = correlation_metrics.average_pairwise_correlation(
        returns_df, portfolio.tickers
    )
    corr_after = correlation_metrics.average_pairwise_correlation(
        returns_df, portfolio_after.tickers
    )
    diffs.append(_make_diff(
        "Avg Pairwise Corr", corr_before, corr_after, "correlation", lower_is_better=True,
    ))

    # --- Sector exposure changes ---
    sectors_before = exposure_metrics.sector_net_exposure(portfolio)
    sectors_after = exposure_metrics.sector_net_exposure(portfolio_after)
    all_sectors = set(list(sectors_before.keys()) + list(sectors_after.keys()))

    for sector in sorted(all_sectors):
        net_before = sectors_before.get(sector, {}).get("net", 0.0)
        net_after = sectors_after.get(sector, {}).get("net", 0.0)
        if abs(net_before - net_after) > 0.001:
            diffs.append(_make_diff(
                f"Sector: {sector}", net_before, net_after, "exposure",
            ))

    # --- Check limits on the after portfolio ---
    limit_warnings = exposure_metrics.check_exposure_limits(
        portfolio_after,
        betas=betas,
    )
    warnings.extend(limit_warnings)

    return TradeImpactResult(
        proposed_trades=basket.trades,
        portfolio_before=portfolio,
        portfolio_after=portfolio_after,
        metric_diffs=diffs,
        warnings=warnings,
        new_positions=new_positions,
        removed_positions=removed_positions,
        modified_positions=modified_positions,
    )
