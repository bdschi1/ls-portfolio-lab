"""Portfolio data model — Pydantic models for positions and portfolio state.

Pure data structures with basic computed properties. Heavy calculations
(covariance, factor regressions) live in core/metrics/ modules.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


class Position(BaseModel):
    """A single long or short position (equity, ETF, or option)."""

    ticker: str
    side: Literal["LONG", "SHORT"]
    shares: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    entry_date: date = Field(default_factory=date.today)
    current_price: float = 0.0
    sector: str = ""
    subsector: str = ""
    market_cap: float = 0.0  # in USD

    # Asset type classification
    asset_type: Literal["EQUITY", "ETF", "OPTION"] = "EQUITY"

    # Option-specific fields (required when asset_type == "OPTION")
    strike: float | None = None
    expiry: date | None = None
    option_type: Literal["CALL", "PUT"] | None = None
    contract_multiplier: int = 100
    delta: float | None = None
    underlying_price: float | None = None  # for delta-adjusted calcs

    @model_validator(mode="after")
    def validate_option_fields(self) -> Position:
        """If asset_type is OPTION, require strike, expiry, and option_type."""
        if self.asset_type == "OPTION":
            missing = []
            if self.strike is None:
                missing.append("strike")
            if self.expiry is None:
                missing.append("expiry")
            if self.option_type is None:
                missing.append("option_type")
            if missing:
                msg = f"Option positions require: {', '.join(missing)}"
                raise ValueError(msg)
        return self

    @property
    def direction(self) -> int:
        """+1 for LONG, -1 for SHORT."""
        return 1 if self.side == "LONG" else -1

    @property
    def notional(self) -> float:
        """Absolute dollar value of position at current price."""
        if self.asset_type == "OPTION":
            return self.shares * self.contract_multiplier * self.current_price
        return self.shares * self.current_price

    @property
    def signed_notional(self) -> float:
        """Signed dollar value: positive for longs, negative for shorts."""
        return self.direction * self.notional

    @property
    def entry_notional(self) -> float:
        """Dollar value at entry."""
        if self.asset_type == "OPTION":
            return self.shares * self.contract_multiplier * self.entry_price
        return self.shares * self.entry_price

    @property
    def pnl_dollars(self) -> float:
        """Unrealized P&L in dollars."""
        multiplier = self.contract_multiplier if self.asset_type == "OPTION" else 1
        return self.direction * self.shares * multiplier * (self.current_price - self.entry_price)

    @property
    def pnl_pct(self) -> float:
        """Unrealized P&L as percentage of entry value."""
        if self.entry_price == 0:
            return 0.0
        return self.direction * (self.current_price - self.entry_price) / self.entry_price

    @property
    def delta_adjusted_notional(self) -> float | None:
        """Delta-adjusted notional for options. None if not an option or delta not set."""
        if self.asset_type != "OPTION" or self.delta is None:
            return None
        underlying = self.underlying_price or self.current_price
        return self.shares * self.contract_multiplier * self.delta * underlying

    def weight_in(self, nav: float) -> float:
        """Position weight as fraction of NAV (signed: positive=long, negative=short)."""
        if nav == 0:
            return 0.0
        return self.signed_notional / nav

    def abs_weight_in(self, nav: float) -> float:
        """Absolute position weight as fraction of NAV."""
        if nav == 0:
            return 0.0
        return self.notional / nav


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


class Portfolio(BaseModel):
    """A long/short equity portfolio with up to 80 positions."""

    name: str = "Untitled"
    positions: list[Position] = Field(default_factory=list, max_length=80)
    benchmark: str = "SPY"
    inception_date: date = Field(default_factory=date.today)
    nav: float = Field(default=3_000_000_000.0, gt=0)  # starting capital
    last_updated: datetime | None = None

    # Cash management
    cash: float | None = None  # actual cash balance; None = auto-computed
    min_cash_pct: float = 0.05  # 5% soft guideline (warning, not a blocker)

    # ETF constraints
    max_etf_positions: int = 3

    @model_validator(mode="after")
    def no_duplicate_tickers(self) -> Portfolio:
        """Ensure no ticker appears more than once."""
        tickers = [p.ticker for p in self.positions]
        dupes = [t for t in set(tickers) if tickers.count(t) > 1]
        if dupes:
            msg = f"Duplicate tickers not allowed: {dupes}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def initialize_cash_and_validate(self) -> Portfolio:
        """Auto-initialize cash from NAV minus invested. Validate ETF count."""
        # Auto-compute cash if not explicitly provided
        if self.cash is None:
            if self.positions:
                total_invested = sum(p.notional for p in self.positions)
                self.cash = self.nav - total_invested
            else:
                self.cash = self.nav

        # Validate ETF count
        etf_count = sum(1 for p in self.positions if p.asset_type == "ETF")
        if etf_count > self.max_etf_positions:
            msg = f"Too many ETF positions: {etf_count} exceeds max {self.max_etf_positions}"
            raise ValueError(msg)

        return self

    # --- Position accessors ---

    @property
    def long_positions(self) -> list[Position]:
        return [p for p in self.positions if p.side == "LONG"]

    @property
    def short_positions(self) -> list[Position]:
        return [p for p in self.positions if p.side == "SHORT"]

    @property
    def long_count(self) -> int:
        return len(self.long_positions)

    @property
    def short_count(self) -> int:
        return len(self.short_positions)

    @property
    def total_count(self) -> int:
        return len(self.positions)

    @property
    def tickers(self) -> list[str]:
        return [p.ticker for p in self.positions]

    def get_position(self, ticker: str) -> Position | None:
        """Look up a position by ticker."""
        for p in self.positions:
            if p.ticker == ticker:
                return p
        return None

    # --- Asset-type accessors ---

    @property
    def equity_positions(self) -> list[Position]:
        """All plain equity (non-ETF, non-option) positions."""
        return [p for p in self.positions if p.asset_type == "EQUITY"]

    @property
    def etf_positions(self) -> list[Position]:
        """All ETF positions."""
        return [p for p in self.positions if p.asset_type == "ETF"]

    @property
    def etf_count(self) -> int:
        return len(self.etf_positions)

    @property
    def option_positions(self) -> list[Position]:
        """All option positions."""
        return [p for p in self.positions if p.asset_type == "OPTION"]

    # --- Cash accessors ---

    @property
    def cash_pct(self) -> float:
        """Cash as a fraction of NAV."""
        if self.nav == 0:
            return 0.0
        return (self.cash or 0.0) / self.nav

    @property
    def investable_cash(self) -> float:
        """Cash available above the soft 5% guideline floor."""
        floor = self.nav * self.min_cash_pct
        return max(0.0, (self.cash or 0.0) - floor)

    # --- Exposure calculations ---

    @property
    def long_notional(self) -> float:
        """Total dollar value of long positions."""
        return sum(p.notional for p in self.long_positions)

    @property
    def short_notional(self) -> float:
        """Total dollar value of short positions."""
        return sum(p.notional for p in self.short_positions)

    @property
    def gross_notional(self) -> float:
        """Sum of absolute notional values (long + short)."""
        return self.long_notional + self.short_notional

    @property
    def net_notional(self) -> float:
        """Long notional minus short notional."""
        return self.long_notional - self.short_notional

    @property
    def gross_exposure(self) -> float:
        """Gross exposure as fraction of NAV."""
        if self.nav == 0:
            return 0.0
        return self.gross_notional / self.nav

    @property
    def net_exposure(self) -> float:
        """Net exposure as fraction of NAV."""
        if self.nav == 0:
            return 0.0
        return self.net_notional / self.nav

    @property
    def long_short_ratio(self) -> float:
        """Long notional / short notional. Inf if no shorts."""
        if self.short_notional == 0:
            return float("inf")
        return self.long_notional / self.short_notional

    # --- P&L ---

    @property
    def total_pnl_dollars(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(p.pnl_dollars for p in self.positions)

    @property
    def total_pnl_pct(self) -> float:
        """Total unrealized P&L as percentage of NAV."""
        if self.nav == 0:
            return 0.0
        return self.total_pnl_dollars / self.nav

    # --- Weight vectors ---

    def weight_vector(self) -> dict[str, float]:
        """Signed weight for each position (positive=long, negative=short)."""
        return {p.ticker: p.weight_in(self.nav) for p in self.positions}

    def abs_weight_vector(self) -> dict[str, float]:
        """Absolute weight for each position."""
        return {p.ticker: p.abs_weight_in(self.nav) for p in self.positions}

    # --- Sector exposure ---

    def sector_exposure(self) -> dict[str, dict[str, float]]:
        """
        Net, long, and short exposure per sector as fraction of NAV.

        Returns: {sector: {"long": float, "short": float, "net": float}}
        """
        sectors: dict[str, dict[str, float]] = {}
        for p in self.positions:
            s = p.sector or "Unknown"
            if s not in sectors:
                sectors[s] = {"long": 0.0, "short": 0.0, "net": 0.0}
            w = p.abs_weight_in(self.nav)
            if p.side == "LONG":
                sectors[s]["long"] += w
            else:
                sectors[s]["short"] += w
            sectors[s]["net"] = sectors[s]["long"] - sectors[s]["short"]
        return sectors

    def subsector_exposure(self) -> dict[str, dict[str, float]]:
        """
        Net, long, and short exposure per subsector as fraction of NAV.

        Returns: {subsector: {"long": float, "short": float, "net": float}}
        """
        subsectors: dict[str, dict[str, float]] = {}
        for p in self.positions:
            s = p.subsector or p.sector or "Unknown"
            if s not in subsectors:
                subsectors[s] = {"long": 0.0, "short": 0.0, "net": 0.0}
            w = p.abs_weight_in(self.nav)
            if p.side == "LONG":
                subsectors[s]["long"] += w
            else:
                subsectors[s]["short"] += w
            subsectors[s]["net"] = subsectors[s]["long"] - subsectors[s]["short"]
        return subsectors

    # --- Mutation helpers (all return new Portfolio, never mutate) ---

    def add_position(self, position: Position) -> Portfolio:
        """Return a new portfolio with the position added.

        Warns (does not block) if cash drops below the soft 5% guideline.
        """
        if self.get_position(position.ticker) is not None:
            msg = f"Position {position.ticker} already exists. Use update_position instead."
            raise ValueError(msg)
        if len(self.positions) >= 80:
            msg = "Portfolio already at max 80 positions."
            raise ValueError(msg)

        # ETF limit check
        if position.asset_type == "ETF":
            if self.etf_count >= self.max_etf_positions:
                msg = (
                    f"Cannot add ETF {position.ticker}: "
                    f"already at max {self.max_etf_positions} ETF positions."
                )
                raise ValueError(msg)

        # Cash tracking
        position_cost = position.notional
        new_cash = (self.cash or self.nav) - position_cost
        cash_floor = self.nav * self.min_cash_pct

        if new_cash < cash_floor:
            logger.warning(
                "Cash will drop to $%s (%.1f%% of NAV) — below %.0f%% guideline "
                "after adding %s ($%s cost).",
                f"{new_cash:,.0f}",
                (new_cash / self.nav) * 100 if self.nav else 0,
                self.min_cash_pct * 100,
                position.ticker,
                f"{position_cost:,.0f}",
            )

        new_positions = [*self.positions, position]
        return self.model_copy(update={
            "positions": new_positions,
            "cash": new_cash,
        })

    def remove_position(self, ticker: str) -> Portfolio:
        """Return a new portfolio with the position removed. Cash is returned."""
        removed = self.get_position(ticker)
        new_positions = [p for p in self.positions if p.ticker != ticker]
        if len(new_positions) == len(self.positions):
            msg = f"Position {ticker} not found."
            raise ValueError(msg)
        returned_cash = removed.notional if removed else 0.0
        return self.model_copy(update={
            "positions": new_positions,
            "cash": (self.cash or 0.0) + returned_cash,
        })

    def update_position(self, ticker: str, **kwargs: object) -> Portfolio:
        """Return a new portfolio with the specified position fields updated."""
        new_positions = []
        found = False
        for p in self.positions:
            if p.ticker == ticker:
                found = True
                new_positions.append(p.model_copy(update=kwargs))
            else:
                new_positions.append(p)
        if not found:
            msg = f"Position {ticker} not found."
            raise ValueError(msg)
        return self.model_copy(update={"positions": new_positions})

    def update_prices(self, prices: dict[str, float]) -> Portfolio:
        """Return a new portfolio with current prices updated from a {ticker: price} dict."""
        new_positions = []
        for p in self.positions:
            if p.ticker in prices:
                new_positions.append(p.model_copy(update={"current_price": prices[p.ticker]}))
            else:
                new_positions.append(p)
        return self.model_copy(
            update={
                "positions": new_positions,
                "last_updated": datetime.now(),
            }
        )


# ---------------------------------------------------------------------------
# Proposed Trade & Trade Basket
# ---------------------------------------------------------------------------


class ProposedTrade(BaseModel):
    """A single proposed trade for the what-if simulator."""

    ticker: str
    action: Literal["BUY", "SELL", "SHORT", "COVER", "ADD", "REDUCE", "EXIT"]
    shares: float | None = Field(default=None, gt=0)
    dollar_amount: float | None = Field(default=None, gt=0)
    notes: str = ""

    # Asset type for the trade
    asset_type: Literal["EQUITY", "ETF", "OPTION"] = "EQUITY"

    # Option-specific fields (required when asset_type == "OPTION")
    strike: float | None = None
    expiry: date | None = None
    option_type: Literal["CALL", "PUT"] | None = None
    contract_multiplier: int = 100
    delta: float | None = None

    @model_validator(mode="after")
    def has_sizing(self) -> ProposedTrade:
        """At least one of shares or dollar_amount must be provided (unless EXIT)."""
        if self.action != "EXIT" and self.shares is None and self.dollar_amount is None:
            msg = "Must provide shares or dollar_amount (unless action is EXIT)."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_option_actions(self) -> ProposedTrade:
        """Options only allow BUY, SELL, EXIT actions (no SHORT/COVER)."""
        if self.asset_type == "OPTION" and self.action in ("SHORT", "COVER"):
            msg = (
                f"Options do not support {self.action}. "
                f"Use BUY/SELL for options (PUT/CALL determines directionality)."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_option_fields(self) -> ProposedTrade:
        """If asset_type is OPTION, require strike, expiry, option_type."""
        if self.asset_type == "OPTION":
            missing = []
            if self.strike is None:
                missing.append("strike")
            if self.expiry is None:
                missing.append("expiry")
            if self.option_type is None:
                missing.append("option_type")
            if missing:
                msg = f"Option trades require: {', '.join(missing)}"
                raise ValueError(msg)
        return self


class TradeBasket(BaseModel):
    """A batch of proposed trades (up to 10)."""

    trades: list[ProposedTrade] = Field(max_length=10)
    description: str = ""
