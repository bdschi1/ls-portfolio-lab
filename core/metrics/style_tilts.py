"""Style/sector tilts and crowding diagnostics.

Per-name factor regressions roll up to portfolio loadings on a set of style
factors (MKT, SMB, HML, MOM, QMJ, BAB or any user-supplied set). Sector tilts
group by Position.sector. Crowding is wired as a passthrough hook: real
implementations require HF 13F ownership and short-interest panels, neither of
which lives in this repo, so the default is a clearly-labelled placeholder.

Sector weight convention (documented to avoid ambiguity):
- ``long_weight`` is the sum of the absolute weights of long positions in the
  sector (always >= 0).
- ``short_weight`` is the sum of the absolute weights of short positions in the
  sector reported as a positive number (always >= 0).
- ``portfolio_weight = long_weight - short_weight``. This is the net signed
  sector weight; longs contribute positive, shorts contribute negative,
  matching the spec ("net signed weight in sector (long+, short-)").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from core.portfolio import Portfolio

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class StyleTilt:
    """Portfolio loading on a single style factor, vs an optional target."""

    factor: str  # e.g. "MKT", "SMB", "HML", "MOM", "QMJ", "BAB"
    portfolio_loading: float  # Σ w_i β_ik using signed weights
    target_loading: float | None  # user-supplied target (None if not given)
    drift: float | None  # portfolio_loading - target_loading (None if no target)


@dataclass
class SectorTilt:
    """Net sector exposure broken into long/short legs vs an optional benchmark."""

    sector: str
    portfolio_weight: float  # net signed weight in sector (long+, short-)
    long_weight: float  # sum of abs weights of long legs in sector (>= 0)
    short_weight: float  # sum of abs weights of short legs in sector (>= 0)
    benchmark_weight: float | None  # benchmark (e.g. SPY) weight if supplied
    active_weight: float | None  # portfolio_weight - benchmark_weight (None if no bench)


@dataclass
class CrowdingScore:
    """Per-name crowding score in [0, 1]. Higher = more crowded.

    ``source`` is one of:
    - ``"external"`` -- score was looked up from a user-supplied dict
    - ``"placeholder"`` -- no data source wired; score is a 0.0 stub. To make
      this meaningful, plug in HF 13F ownership data (e.g. WhaleWisdom,
      Goldman VIP basket weights) or short-interest panels.
    """

    ticker: str
    score: float
    source: str


@dataclass
class StyleTiltsReport:
    """Bundle of style, sector and crowding diagnostics for one portfolio."""

    style: list[StyleTilt]
    sectors: list[SectorTilt]
    crowding: list[CrowdingScore]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_MIN_OBS_FOR_REGRESSION = 30


def _ols_betas(
    y: np.ndarray,
    factor_matrix: np.ndarray,
) -> np.ndarray:
    """OLS slope vector (no intercept returned). Adds an intercept column.

    Returns the K-vector of factor betas (intercept dropped).
    Falls back to a vector of zeros on a singular design matrix.
    """
    n = y.shape[0]
    x_with_const = np.column_stack([np.ones(n), factor_matrix])
    try:
        betas, *_ = np.linalg.lstsq(x_with_const, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(factor_matrix.shape[1])
    return betas[1:]


def _per_name_betas(
    returns_df: pl.DataFrame,
    factor_returns: dict[str, pl.Series],
    tickers: list[str],
) -> dict[str, dict[str, float]]:
    """Run a multi-factor OLS for each ticker. Returns {ticker: {factor: beta}}.

    Missing tickers (not in returns_df) or short histories get all-zero betas
    so they contribute nothing to portfolio loading rather than NaN-poisoning
    the aggregate.
    """
    factor_names = list(factor_returns.keys())
    if not factor_names:
        return {t: {} for t in tickers}

    # Trim every factor series to the shortest length so they align.
    min_factor_len = min(s.len() for s in factor_returns.values())
    factor_cols = [factor_returns[name].tail(min_factor_len).to_numpy() for name in factor_names]
    factor_matrix_full = np.column_stack(factor_cols)

    out: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        if ticker not in returns_df.columns:
            out[ticker] = dict.fromkeys(factor_names, 0.0)
            continue
        r = returns_df[ticker]
        n = min(r.len(), min_factor_len)
        if n < _MIN_OBS_FOR_REGRESSION:
            out[ticker] = dict.fromkeys(factor_names, 0.0)
            continue
        y = r.tail(n).to_numpy()
        x = factor_matrix_full[-n:, :]
        betas = _ols_betas(y, x)
        out[ticker] = {name: float(betas[i]) for i, name in enumerate(factor_names)}
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def style_tilts(
    portfolio: Portfolio,
    returns_df: pl.DataFrame,
    factor_returns: dict[str, pl.Series],
    targets: dict[str, float] | None = None,
    benchmark_weights: dict[str, float] | None = None,
    crowding_data: dict[str, float] | None = None,
) -> StyleTiltsReport:
    """Compute style / sector / crowding tilts for a long-short portfolio.

    Per-name factor betas are estimated via OLS of each ticker's return series
    on the supplied factor returns. Portfolio loading on factor k is the
    signed-weight-weighted sum: ``Σ w_i × β_ik``. Signed weights mean longs
    contribute their weight as a positive number and shorts as negative, so a
    well-hedged L/S book lands near zero on broad factors like MKT.

    Args:
        portfolio: the Portfolio to analyse.
        returns_df: Polars DataFrame with one column per ticker (daily simple
            returns). Tickers not in the DataFrame contribute zero beta.
        factor_returns: dict mapping factor name (e.g. ``"MKT"``, ``"SMB"``,
            ``"HML"``, ``"MOM"``, ``"QMJ"``, ``"BAB"``) to a Polars Series of
            factor returns. Factor names are passed through verbatim into the
            output -- callers pick the labels.
        targets: optional dict ``{factor_name: target_loading}``. Drift is only
            computed for factors that appear here.
        benchmark_weights: optional dict ``{sector: benchmark_weight}`` (e.g.
            SPY sector weights). Active weight is only computed when a sector
            is in this dict.
        crowding_data: optional dict ``{ticker: score in [0, 1]}``. When None,
            crowding rows are emitted as zero-score placeholders -- see
            CrowdingScore docstring for how to wire real data.

    Returns:
        StyleTiltsReport bundling style, sector and crowding lists.
    """
    weights = portfolio.weight_vector()  # signed: longs +, shorts -
    tickers = list(weights.keys())

    # ---- style ----
    betas_per_name = _per_name_betas(returns_df, factor_returns, tickers)
    style: list[StyleTilt] = []
    for factor in factor_returns:
        loading = sum(weights[t] * betas_per_name[t].get(factor, 0.0) for t in tickers)
        target = targets.get(factor) if targets else None
        drift = loading - target if target is not None else None
        style.append(
            StyleTilt(
                factor=factor,
                portfolio_loading=float(loading),
                target_loading=target,
                drift=drift,
            )
        )

    # ---- sectors ----
    nav = portfolio.nav
    sector_long: dict[str, float] = {}
    sector_short: dict[str, float] = {}
    for p in portfolio.positions:
        sector = p.sector or "Unknown"
        w_abs = p.abs_weight_in(nav)
        if p.side == "LONG":
            sector_long[sector] = sector_long.get(sector, 0.0) + w_abs
        else:
            sector_short[sector] = sector_short.get(sector, 0.0) + w_abs

    all_sectors = sorted(set(sector_long) | set(sector_short))
    sectors: list[SectorTilt] = []
    for sector in all_sectors:
        long_w = sector_long.get(sector, 0.0)
        short_w = sector_short.get(sector, 0.0)
        net = long_w - short_w
        bench = benchmark_weights.get(sector) if benchmark_weights else None
        active = net - bench if bench is not None else None
        sectors.append(
            SectorTilt(
                sector=sector,
                portfolio_weight=net,
                long_weight=long_w,
                short_weight=short_w,
                benchmark_weight=bench,
                active_weight=active,
            )
        )

    # ---- crowding ----
    # No HF 13F ownership panel or short-interest feed lives in this repo, so
    # the default mode emits placeholders. Wire real data in via
    # ``crowding_data`` once a source is available.
    crowding: list[CrowdingScore] = []
    if crowding_data is None:
        for t in tickers:
            crowding.append(CrowdingScore(ticker=t, score=0.0, source="placeholder"))
    else:
        for t in tickers:
            score = crowding_data.get(t, 0.0)
            source = "external" if t in crowding_data else "placeholder"
            crowding.append(CrowdingScore(ticker=t, score=float(score), source=source))

    return StyleTiltsReport(style=style, sectors=sectors, crowding=crowding)
