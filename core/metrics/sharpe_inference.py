"""Sharpe ratio inference — delegates to shared bds_data_providers package.

All core inference functions (variance, PSR, MinTRL, critical SR, power,
FDR/FWER corrections, expected maximum SR) are implemented in
bds_data_providers.sharpe_inference and re-exported here for backward
compatibility.

References:
    Bailey & Lopez de Prado (2014). The Deflated Sharpe Ratio.
    Bailey & Lopez de Prado (2012). The Sharpe Ratio Efficient Frontier.
    Lo (2002). The Statistics of Sharpe Ratios.
"""

from bds_data_providers.sharpe_inference import (  # noqa: F401
    _make_expectation_gh,
    _moments_mk,
    adjusted_p_values_bonferroni,
    adjusted_p_values_holm,
    adjusted_p_values_sidak,
    control_for_fdr,
    critical_sharpe_ratio,
    expected_maximum_sharpe_ratio,
    minimum_track_record_length,
    observed_fdr,
    posterior_fdr,
    probabilistic_sharpe_ratio,
    sharpe_ratio_power,
    sharpe_ratio_variance,
)

__all__ = [
    "sharpe_ratio_variance",
    "probabilistic_sharpe_ratio",
    "minimum_track_record_length",
    "critical_sharpe_ratio",
    "sharpe_ratio_power",
    "posterior_fdr",
    "observed_fdr",
    "control_for_fdr",
    "adjusted_p_values_bonferroni",
    "adjusted_p_values_sidak",
    "adjusted_p_values_holm",
    "expected_maximum_sharpe_ratio",
]
