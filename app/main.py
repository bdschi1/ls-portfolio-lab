"""LS Portfolio Lab — Main Streamlit Application.

Entry point: streamlit run app/main.py

Layout orchestration, navigation, and top-level controls.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="LS Portfolio Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.pages import paper_portfolio, pm_scorecard, portfolio_view, trade_simulator  # noqa: E402
from app.state.session import init_session_state  # noqa: E402

# Initialize session state
init_session_state()


def main() -> None:
    """Main app entry point."""

    # --- Sidebar Navigation ---
    with st.sidebar:
        # --- Sidebar CSS: bigger nav fonts (+3), bigger body fonts (+2),
        #     compressed spacing between sections ---
        st.markdown("""<style>
        /* --- Navigation radio: bigger font --- */
        section[data-testid="stSidebar"] div[role="radiogroup"] label p {
            font-size: clamp(13px, 1.0vw, 19px) !important;
            font-weight: 600 !important;
        }
        /* --- All sidebar labels, text, captions --- */
        section[data-testid="stSidebar"] .stSelectbox label p,
        section[data-testid="stSidebar"] .stTextInput label p,
        section[data-testid="stSidebar"] .stNumberInput label p,
        section[data-testid="stSidebar"] .stSlider label p,
        section[data-testid="stSidebar"] .stRadio label p,
        section[data-testid="stSidebar"] .stToggle label p,
        section[data-testid="stSidebar"] .stCheckbox label p {
            font-size: clamp(12px, 0.9vw, 17px) !important;
        }
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
        section[data-testid="stSidebar"] .stTextInput input,
        section[data-testid="stSidebar"] .stNumberInput input {
            font-size: clamp(12px, 0.9vw, 17px) !important;
        }
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] .stMarkdown p {
            font-size: clamp(11px, 0.85vw, 16px) !important;
        }
        /* --- Compress spacing: tighter gaps --- */
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0 !important;
        }
        section[data-testid="stSidebar"] hr {
            margin-top: 0.4rem !important;
            margin-bottom: 0.4rem !important;
        }
        section[data-testid="stSidebar"] .stSubheader {
            margin-top: 0 !important;
            padding-top: 0 !important;
            margin-bottom: 0.2rem !important;
        }
        section[data-testid="stSidebar"] .element-container {
            margin-bottom: 0.15rem !important;
        }
        section[data-testid="stSidebar"] .stRadio > div {
            gap: 0.15rem !important;
        }
        section[data-testid="stSidebar"] .stExpander {
            margin-top: 0.2rem !important;
            margin-bottom: 0.2rem !important;
        }
        /* Nav description text under each radio option */
        .nav-desc {
            font-size: clamp(10px, 0.75vw, 14px) !important;
            color: #888 !important;
            margin-top: -0.1rem;
            margin-bottom: 0.3rem;
            line-height: 1.3;
        }
        </style>""", unsafe_allow_html=True)

        st.title("📊 LS Portfolio Lab")
        st.caption("Long/Short Equity Risk Workbench")

        # Instructions dialog
        @st.dialog("LS Portfolio Lab — Instructions", width="large")
        def _show_instructions():
            st.markdown("""
### Why This Was Built

LS Portfolio Lab is a **risk cockpit** for long/short equity portfolio managers.
It is *not* an alpha generator — it helps you **monitor, stress-test, and track**
your portfolio's risk/return profile in real time.

Built for PMs who need a lightweight, local-first tool to answer:
*"What happens to my risk if I add this trade?"*

---

### Pages

**1. Portfolio Dashboard**
- Upload an Excel/CSV portfolio or generate a mock one.
- **Top bar:** Portfolio Vol, Net Beta, Sharpe, RSI, exposure, Cash %.
- **Detail strip:** Summary, Risk, Drawdown, Factors, Correlation.
- **Position table:** Sortable, filterable. Per-name vol, beta, RSI.
- **Charts:** Sector exposure, correlation heatmap, NAV curve, beta scatter.

**2. Trade Simulator**
- Enter up to **10 proposed trades** (BUY, SHORT, ADD, REDUCE, SELL, COVER, EXIT).
- Supports equities, ETFs, and **options with delta adjustment**.
- Size by shares or dollar amount.
- See a **before/after comparison** of every metric — then apply or discard.

**3. Paper Portfolio**
- Toggle **Paper Mode** in the sidebar to start tracking.
- Every applied trade is logged to an **immutable JSONL trade journal**.
- Daily snapshots capture NAV, positions, sector exposure, P&L.
- View your full trade history with entry/exit prices and realized P&L.

**4. PM Scorecard**
- Available when Paper Mode is ON with trade history.
- **Hit rate** (% winning trades), **Slugging %** (avg winner / avg loser).
- Expected value per trade, sector attribution, annualized alpha.

---

### Sidebar Controls

- **Lookback:** 3M to 3Y of historical data for metric calculations.
- **Benchmark:** Default SPY — change to any ticker.
- **RSI Period:** 7, 14, or 21 days.
- **Factor Model:** CAPM, Fama-French 3, or 4-factor.
- **Risk-Free Rate:** Auto (13-week T-bill) or manual override.
- **Alert Thresholds:** Max sector net exposure, beta bounds, gross exposure cap.
- **Risk Limits (dashboard):** Max net beta MV %, target annualized vol %.

---

### Data

- **Source:** yfinance (free, no API key required). EOD data only.
- **Cache:** SQLite local cache — prices refresh every 18 hours, fundamentals every 7 days.
- **Universe:** ~440 liquid Russell 1000 names (market cap > $5B).
""")

        # Glossary dialog
        @st.dialog("📘 Glossary — Metrics & Methodology", width="large")
        def _show_glossary():
            st.markdown(r"""
### Portfolio-Level Risk

**Portfolio Volatility (annualized)**
$$\sigma_p = \sqrt{\mathbf{w}^\top \Sigma \, \mathbf{w}} \;\times\; \sqrt{252}$$
Covariance matrix $\Sigma$ estimated from daily returns over the selected lookback window
(default 1Y / 252 trading days). Weights $\mathbf{w}$ are signed (+long, −short) as fractions
of NAV.

**Sharpe Ratio**
$$SR = \frac{R_p - R_f}{\sigma_p}$$
$R_p$ = geometric annualized return: $\bigl(\prod(1+r_i)\bigr)^{252/n} - 1$.
$R_f$ = annualized risk-free rate (default: 13-week T-bill via yfinance, or manual override).
$\sigma_p$ = annualized portfolio volatility.

**Sortino Ratio**
$$\text{Sortino} = \frac{R_p - R_f}{\sigma_{\text{down}}}$$
Same as Sharpe but denominator uses **downside deviation** — standard deviation computed only
on negative daily returns, annualized by $\sqrt{252}$.

**Calmar Ratio**
$$\text{Calmar} = \frac{R_p}{|\text{Max DD}|}$$
Annualized return divided by the absolute value of the worst peak-to-trough drawdown.

**Deflated Sharpe Ratio (DSR)** — Bailey & Lopez de Prado (2014)
$$DSR = \Phi\!\left[\frac{\widehat{SR} - SR_0}{\sigma(\widehat{SR})}\right]$$
where the standard error of the Sharpe ratio accounts for non-normality:
$$\sigma(\widehat{SR}) = \sqrt{\frac{1 - \gamma_3 \cdot SR + \frac{\gamma_4 - 1}{4} \cdot SR^2}{T}}$$
$\gamma_3$ = skewness, $\gamma_4$ = excess kurtosis, $T$ = number of observations.
$SR_0$ adjusts for multiple testing (number of strategy variants tried).
Reported as a **probability** — DSR > 95% means the Sharpe is statistically significant at the 5% level.

---

### Position-Level Volatility

**Annualized Vol (per name)**
$$\sigma_i = \text{std}(r_i) \times \sqrt{252}$$
Standard deviation of daily returns for a single position, annualized. Measures total risk
(systematic + idiosyncratic combined).

**Idiosyncratic Vol (per name)**
$$\sigma_{\varepsilon,i} = \text{std}(r_i - \beta_i \cdot r_m) \times \sqrt{252}$$
Annualized standard deviation of the CAPM residual — the portion of a stock's volatility
**not** explained by its market beta. High idiosyncratic vol = stock-specific risk dominates;
low = the position mostly moves with the market.

---

### Value at Risk & Expected Shortfall

**VaR 95% (Historical)**
The 5th percentile of the empirical daily return distribution. Non-parametric — no
distributional assumption. Interpretation: on 95% of days, the portfolio daily loss
should not exceed this amount.

**CVaR 95% (Conditional VaR / Expected Shortfall)**
$$\text{CVaR}_{95} = \mathbb{E}\bigl[R \;\big|\; R \le \text{VaR}_{95}\bigr]$$
Average of all daily returns **below** the VaR threshold. Captures tail risk more
completely than VaR alone.

---

### Drawdown Analytics

Empirical drawdowns are computed from the cumulative return series:
$$\text{DD}_t = \frac{V_t - V_{\text{peak}}}{V_{\text{peak}}}$$

**Max Drawdown** — worst peak-to-trough decline observed in the lookback window.

**Current Drawdown** — distance from the most recent cumulative-return peak to today.

Analytical drawdowns follow **Bailey & Lopez de Prado (2014)** assuming P&L is arithmetic
Brownian motion $dX_t = \mu\,dt + \sigma\,dB_t$.

**Expected Drawdown**
$$\mathbb{E}[D] = \frac{\sigma}{2 \cdot SR}$$

**P(DD ≥ 10%)** — stationary probability of ever reaching a 10% drawdown:
$$P(D \ge b) = \exp\!\bigl(-2\,b_\sigma \cdot SR\bigr), \quad b_\sigma = b / \sigma$$

**Time in Drawdown** — expected fraction of time spent below the previous high-water mark:
$$\mathbb{E}[\text{time in DD}] = \frac{1}{2 \cdot SR^2}$$

---

### Exposure & Concentration

**Gross Exposure**
$$\text{Gross} = \frac{\sum |w_i| \cdot P_i \cdot \text{shares}_i}{\text{NAV}}$$
Total dollars deployed (longs + shorts) as a fraction of NAV. E.g., 180% = $1.80 deployed
per $1 of capital.

**Net Exposure** — long notional minus short notional, divided by NAV. Positive = net long.

**Net Beta**
$$\beta_{\text{net}} = \sum_i w_i \cdot \beta_i$$
Signed weights times per-position betas. Measures the portfolio's net directional market
sensitivity. Per-position $\beta_i$ is computed via OLS regression of daily returns against the
benchmark (default SPY) over the full lookback window.

**HHI (Herfindahl-Hirschman Index)**
$$\text{HHI} = \sum_i w_i^2$$
Sum of squared absolute position weights. Measures portfolio concentration.
HHI ≈ 0 = highly diversified across many positions.
HHI = 1.0 = entirely in a single position.
For an equal-weight portfolio of $N$ names, $\text{HHI} = 1/N$.

**Top 5 Concentration** — sum of the five largest absolute position weights.

**L/S Ratio** — long notional divided by short notional.

---

### Factor Models

Single-factor CAPM:
$$R_i - R_f = \alpha + \beta_{\text{mkt}}(R_m - R_f) + \varepsilon$$

**FF3 — Fama & French (1993)** three-factor:
$$R_i - R_f = \alpha + \beta_{\text{mkt}}(R_m - R_f) + \beta_{\text{smb}} \cdot SMB + \beta_{\text{hml}} \cdot HML + \varepsilon$$
- **SMB (Small Minus Big):** size factor, proxied by IWM − SPY
- **HML (High Minus Low):** value factor, proxied by IWD − IWF

**FF4 — Carhart (1997)** four-factor adds momentum:
$$R_i - R_f = \alpha + \beta_{\text{mkt}}(R_m - R_f) + \beta_{\text{smb}} \cdot SMB + \beta_{\text{hml}} \cdot HML + \beta_{\text{mom}} \cdot UMD + \varepsilon$$
- **UMD (Up Minus Down):** momentum factor, proxied by MTUM − SPY

All regressions are OLS, requiring ≥ 30 days of data. Factor loadings use ETF proxies
(not Kenneth French library data).

**Alpha (Jensen's α)** — the intercept of the factor regression, annualized ($\alpha_{\text{daily}} \times 252$).
Represents return not explained by factor exposures.

**Systematic %** — $R^2$ of the regression. Fraction of return variance explained by factor exposures.

**Residual Vol** — $\sigma_\varepsilon \times \sqrt{252}$ — annualized standard deviation of regression residuals
(idiosyncratic risk).

---

### Correlation Metrics

All correlations use the Pearson correlation coefficient from daily returns over the lookback window.

**Avg Pairwise** — mean of all off-diagonal entries in the $N \times N$ correlation matrix.

**Long Corr / Short Corr** — average pairwise correlation within the long book (or short book) only.

**L/S Corr** — correlation between the equal-weighted long-book return series and the
equal-weighted short-book return series. High positive = shorts move with longs (good hedge).

---

### Technical Indicators

**RSI (Relative Strength Index)** — Wilder's exponential smoothing (default 14-day, configurable 7/14/21):
$$RSI = 100 - \frac{100}{1 + RS}, \quad RS = \frac{\text{EMA}(\text{gains}, n)}{\text{EMA}(\text{losses}, n)}$$
The dashboard shows **notional-weighted** RSI for the long and short books separately.
RSI > 70 = overbought, RSI < 30 = oversold.

**Tracking Error**
$$TE = \text{std}(R_p - R_b) \times \sqrt{252}$$
Annualized standard deviation of excess returns vs. the benchmark.

---

### PM Scorecard (Paper Mode)

**Hit Rate** — percentage of closed trades that were profitable.

**Slugging %** — average winning trade P&L / average losing trade P&L (absolute values).
A slugging of 2.0× means winners are twice as large as losers on average.
A PM with 40% hit rate and 3.0× slugging beats 60% hit rate with 1.0× slugging:
$$\text{EV} = (0.40 \times 3W) - (0.60 \times W) = +0.60W$$

**Expected Value per Trade**
$$EV = (\text{hit rate} \times \text{avg win}) - (\text{miss rate} \times \text{avg loss})$$

---

### Data & Caching

- **Prices:** End-of-day adjusted close from Yahoo Finance via yfinance. Refreshed every **18 hours**.
- **Fundamentals** (sector, market cap): refreshed every **7 days**.
- **Risk-free rate (auto):** 13-week US Treasury bill yield (^IRX), converted to annualized rate.
- **Cache:** local SQLite database. No external API keys required.

---

### References

Bailey, D.H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*, 40(5), 94-107. — DSR, σ(SR) with skewness/kurtosis correction.

Bailey, D.H. & Lopez de Prado, M. (2014). "The Sharpe Ratio Efficient Frontier." *Algorithmic Finance*, 3(1-2), 99-109. DOI: [10.3233/AF-140035](https://doi.org/10.3233/AF-140035) — E[DD], P(DD≥b), time in DD.

Lo, A. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts Journal*, 58(4), 36-52. — SR standard error, PSR, MinTRL.

Paleologo, G. (2024). *The Elements of Quantitative Investing*, Insight 4.2. — Partial correlations via precision matrix.

Fama, E.F. & French, K.R. (1993). *Journal of Financial Economics*, 33(1), 3-56. — FF3 model.

Carhart, M.M. (1997). *The Journal of Finance*, 52(1), 57-82. — Momentum (FF4).
""")

        btn_cols = st.columns(2)
        with btn_cols[0]:
            if st.button("📖 Instructions", use_container_width=True):
                _show_instructions()
        with btn_cols[1]:
            if st.button("📘 Glossary", use_container_width=True):
                _show_glossary()

        st.divider()

        # Navigation with descriptions
        _PAGE_INFO = {
            "Portfolio": "Live risk dashboard — metrics, charts, positions",
            "Trade Simulator": "Model trades and preview impact before executing",
            "Paper Portfolio": "Track trades and build a performance record",
            "PM Scorecard": "Hit rate, slugging %, attribution from paper trades",
        }

        page = st.radio(
            "Navigate",
            options=list(_PAGE_INFO.keys()),
            index=0,
            key="nav_radio",
        )
        st.session_state.active_page = page
        # Show description for the selected page
        st.markdown(
            f'<p class="nav-desc">{_PAGE_INFO[page]}</p>',
            unsafe_allow_html=True,
        )

        st.divider()

        # Paper mode toggle
        paper_mode = st.toggle(
            "📝 Paper Portfolio Mode",
            value=st.session_state.get("paper_mode", False),
            help="Track all trades and build PM performance history",
        )
        st.session_state.paper_mode = paper_mode

        if paper_mode:
            trade_count = st.session_state.trade_log.trade_count
            snap_count = st.session_state.snapshot_store.snapshot_count
            st.success(f"📝 Paper mode ON — {trade_count} trades, {snap_count} snapshots")

        st.divider()

        # Quick settings
        st.subheader("⚙️ Quick Settings")

        settings = st.session_state.settings

        settings["lookback_days"] = st.selectbox(
            "Lookback (trading days)",
            options=[63, 126, 252, 504, 756],
            index=[63, 126, 252, 504, 756].index(settings.get("lookback_days", 252)),
            format_func=lambda x: {63: "3M (63d)", 126: "6M (126d)", 252: "1Y (252d)",
                                     504: "2Y (504d)", 756: "3Y (756d)"}[x],
        )

        settings["benchmark"] = st.text_input(
            "Benchmark",
            value=settings.get("benchmark", "SPY"),
        )

        settings["rsi_period"] = st.selectbox(
            "RSI Period",
            options=[7, 14, 21],
            index=[7, 14, 21].index(settings.get("rsi_period", 14)),
        )

        settings["factor_model"] = st.selectbox(
            "Factor Model",
            options=["CAPM", "FF3", "FF4"],
            index=["CAPM", "FF3", "FF4"].index(settings.get("factor_model", "CAPM")),
            format_func=lambda x: {
                "CAPM": "CAPM — Market only",
                "FF3": "FF3 — Fama-French (Mkt + SMB + HML)",
                "FF4": "FF4 — Carhart (Mkt + SMB + HML + Mom)",
            }[x],
            help=(
                "CAPM: single-factor market model. "
                "FF3: Fama & French (1993) three-factor — adds Size (SMB) and Value (HML). "
                "FF4: Carhart (1997) four-factor — adds Momentum (UMD)."
            ),
        )

        rf = settings.get("risk_free_rate", "auto")
        rf_mode = st.radio(
            "Risk-Free Rate",
            options=["Auto (T-Bill)", "Manual"],
            index=0 if rf == "auto" else 1,
            horizontal=True,
        )
        if rf_mode == "Manual":
            settings["risk_free_rate"] = st.number_input(
                "Rate (%)", min_value=0.0, max_value=20.0,
                value=5.0, step=0.25,
            ) / 100
        else:
            settings["risk_free_rate"] = "auto"

        st.divider()

        # Data source selector
        with st.expander("🔌 Data Source"):
            from data.provider_factory import (
                PROVIDER_DESCRIPTIONS,
                available_providers,
                get_provider_safe,
            )

            avail = available_providers()
            current_source = st.session_state.get("data_source", "Yahoo Finance")
            if current_source not in avail:
                current_source = avail[0] if avail else "Yahoo Finance"

            source = st.selectbox(
                "Provider",
                options=avail,
                index=avail.index(current_source) if current_source in avail else 0,
                key="data_source_select",
                help="Switch the market data provider. Bloomberg and IB require "
                     "their respective software running locally.",
            )
            # Show description
            desc = PROVIDER_DESCRIPTIONS.get(source, "")
            if desc:
                st.caption(desc)

            # If the user changed the provider, update cache
            if source != st.session_state.get("data_source"):
                st.session_state.data_source = source
                new_provider = get_provider_safe(source)
                st.session_state.data_cache.set_provider(new_provider)
                # Clear cached returns so data refreshes from new source
                st.session_state.returns_df = None
                st.rerun()

        st.divider()

        # Alert thresholds (expandable)
        with st.expander("🚨 Alert Thresholds"):
            alerts = st.session_state.alerts
            alerts["max_sector_net_exposure"] = st.slider(
                "Max Sector Net Exposure", 0.1, 1.0,
                value=alerts.get("max_sector_net_exposure", 0.50),
                step=0.05, format="%.0f%%",
            )
            alerts["max_net_beta"] = st.slider(
                "Max Net Beta", 0.0, 0.5,
                value=alerts.get("max_net_beta", 0.30),
                step=0.05,
            )
            alerts["min_net_beta"] = st.slider(
                "Min Net Beta", -0.5, 0.0,
                value=alerts.get("min_net_beta", -0.10),
                step=0.05,
            )
            alerts["max_gross_exposure"] = st.slider(
                "Max Gross Exposure", 1.0, 4.0,
                value=alerts.get("max_gross_exposure", 2.50),
                step=0.1,
            )

    # --- Main Content ---
    if page == "Portfolio":
        portfolio_view.render()
    elif page == "Trade Simulator":
        trade_simulator.render()
    elif page == "Paper Portfolio":
        paper_portfolio.render()
    elif page == "PM Scorecard":
        pm_scorecard.render()


main()
