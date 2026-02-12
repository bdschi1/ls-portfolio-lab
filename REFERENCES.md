# References

Academic papers and methodologies implemented in LS Portfolio Lab.

---

## Drawdown Analytics & Deflated Sharpe Ratio

**Bailey, D.H. & Lopez de Prado, M. (2014).** "The Strategy Approval Decision: A Sharpe Ratio Indifference Curve Approach." *Algorithmic Finance*, 3(1-2), 99-109. DOI: [10.3233/AF-140035](https://doi.org/10.3233/AF-140035)

Implemented in [`core/metrics/drawdown_analytics.py`](core/metrics/drawdown_analytics.py) and [`core/metrics/return_metrics.py`](core/metrics/return_metrics.py).

**Drawdown results used:**
| Formula | Equation | Description |
|---------|----------|-------------|
| `E[D] = σ / (2·SR)` | Eq. 7 | Expected drawdown depth |
| `P(D ≥ b) = exp(−2·b_σ·SR)` | Eq. 6 | Stationary probability of drawdown ≥ b |
| `P(b, T)` via inverse Gaussian CDF | Eq. 4-5 | Finite-horizon drawdown probability |
| `E[τ] = b_σ / SR` | Eq. 8 | Expected recovery time from drawdown b |
| `std(τ) = √(b_σ/SR) / SR` | Eq. 9 | Recovery time standard deviation |
| `E[time in DD] = 1 / (2·SR²)` | Eq. 10 | Expected fraction of time in drawdown |

All formulas assume strategy P&L follows arithmetic Brownian motion: `dX_t = μdt + σdB_t`.

**Deflated Sharpe Ratio (DSR):**
Adjusts the observed Sharpe ratio for non-normality (skewness, kurtosis) and sample size:
`DSR = Φ[(SR − SR₀) / σ(SR)]` where `σ(SR) = √[(1 − γ₃·SR + (γ₄−1)/4·SR²) / T]`.
Reported as a probability — DSR > 95% indicates statistical significance at the 5% level.
Also supports multiple-testing correction via the expected maximum SR under the null.

---

## Factor Models

**Fama, E.F. & French, K.R. (1993).** "Common Risk Factors in the Returns on Stocks and Bonds." *Journal of Financial Economics*, 33(1), 3-56. DOI: [10.1016/0304-405X(93)90023-5](https://doi.org/10.1016/0304-405X(93)90023-5)

Three-factor model (**FF3**): `Ri − Rf = α + β_mkt(Rm−Rf) + β_smb(SMB) + β_hml(HML) + ε`
- **Market (Rm−Rf):** Excess return of the market portfolio
- **SMB (Small Minus Big):** Size factor — return spread between small-cap and large-cap stocks
- **HML (High Minus Low):** Value factor — return spread between high book-to-market and low book-to-market stocks

**Carhart, M.M. (1997).** "On Persistence in Mutual Fund Performance." *The Journal of Finance*, 52(1), 57-82. DOI: [10.1111/j.1540-6261.1997.tb03808.x](https://doi.org/10.1111/j.1540-6261.1997.tb03808.x)

Four-factor model (**FF4** / Carhart): Extends FF3 with momentum:
`Ri − Rf = α + β_mkt(Rm−Rf) + β_smb(SMB) + β_hml(HML) + β_mom(UMD) + ε`
- **UMD (Up Minus Down):** Momentum factor — return spread between past winners and past losers

Implemented in [`core/factor_model.py`](core/factor_model.py) using ETF proxies:
| Factor | Long ETF | Short ETF | Academic Source |
|--------|----------|-----------|-----------------|
| Market | SPY | — | CAPM / Fama-French |
| Size (SMB) | IWM | SPY | Fama & French (1993) |
| Value (HML) | IWD | IWF | Fama & French (1993) |
| Momentum (UMD) | MTUM | SPY | Carhart (1997) |

---

## Risk Metrics

Standard implementations of:
- **Sharpe Ratio:** Sharpe, W.F. (1966). "Mutual Fund Performance." *Journal of Business*, 39(S1), 119-138.
- **Sortino Ratio:** Sortino, F.A. & van der Meer, R. (1991). "Downside Risk." *Journal of Portfolio Management*, 17(4), 27-31.
- **Value at Risk / CVaR:** Historical simulation approach (non-parametric).
- **Calmar Ratio:** Young, T.W. (1991). "Calmar Ratio: A Smoother Tool." *Futures Magazine*.
- **HHI (Herfindahl-Hirschman Index):** Concentration measure — sum of squared portfolio weights. HHI ≈ 0 = highly diversified, HHI → 1 = concentrated.
