"""
Risk metrics for single stock analysis.

This module includes functions for computing Value at Risk (VaR) using
multiple methods, the Sharpe ratio, and the CAPM beta of an asset relative
to a benchmark index.  These metrics help quantify downside risk and
risk‑adjusted return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Literal, Optional, Dict

from scipy.stats import norm


def compute_var(returns: pd.Series,
                confidence_level: float = 0.95,
                method: Literal['historical', 'parametric', 'monte_carlo'] = 'historical',
                num_simulations: int = 10000) -> float:
    """Compute Value at Risk (VaR) of a time series of returns.

    Parameters
    ----------
    returns : pd.Series
        Series of returns (in decimal form, e.g., 0.01 for 1%).
    confidence_level : float, default 0.95
        The desired confidence level for VaR (e.g., 0.95 for 95% VaR).
    method : {'historical', 'parametric', 'monte_carlo'}, default 'historical'
        Method to compute VaR:
        - 'historical': uses the empirical distribution of past returns【983904180201883†L266-L315】.
        - 'parametric': assumes returns are normally distributed and uses mean & std【983904180201883†L266-L315】.
        - 'monte_carlo': simulates price paths under a log‑normal GBM model【569262248736669†L287-L385】.
    num_simulations : int, default 10000
        Number of simulated paths for Monte Carlo method.

    Returns
    -------
    float
        The Value at Risk at the specified confidence level (a negative number representing loss).
    """
    # Convert returns to numpy array
    r = returns.dropna().values

    if method == 'historical':
        # Sort returns ascending; VaR is the quantile at (1 - confidence)
        var = np.quantile(r, 1 - confidence_level)
        return var
    elif method == 'parametric':
        mu = np.mean(r)
        sigma = np.std(r, ddof=1)
        # Compute z‑score for the confidence level
        z = norm.ppf(1 - confidence_level)
        var = mu + z * sigma
        return var
    elif method == 'monte_carlo':
        # Estimate drift and volatility from historical returns (log returns)
        mu = np.mean(r)
        sigma = np.std(r, ddof=1)
        # Time step = 1 day
        dt = 1
        # Starting price assumed to be 1; we only need relative changes
        price0 = 1.0
        # Generate random shocks
        # Simulate final return after one period using GBM formula: S_t = S0 * exp((mu - 0.5*sigma^2)dt + sigma * sqrt(dt) * Z)
        z = np.random.normal(size=num_simulations)
        st = price0 * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        simulated_returns = (st - price0) / price0
        var = np.quantile(simulated_returns, 1 - confidence_level)
        return var
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_sharpe_ratio(returns: pd.Series,
                         risk_free_rate: float = 0.0,
                         freq: int = 252) -> float:
    """Calculate the annualised Sharpe ratio of a series of returns.

    The Sharpe ratio measures risk‑adjusted return and is defined as
    (mean return − risk free rate) divided by the standard deviation of returns【719733448842304†L209-L230】.

    Parameters
    ----------
    returns : pd.Series
        Series of daily (or periodic) returns in decimal form.
    risk_free_rate : float, default 0.0
        Risk‑free rate per period (e.g., daily risk‑free rate).  By default
        zero.
    freq : int, default 252
        Number of periods per year (252 for trading days).  Used to
        annualise the Sharpe ratio.

    Returns
    -------
    float
        Annualised Sharpe ratio.
    """
    excess_returns = returns - risk_free_rate
    mean_excess = excess_returns.mean() * freq
    std_dev = returns.std(ddof=1) * np.sqrt(freq)
    if std_dev == 0:
        return np.nan
    return mean_excess / std_dev


def compute_beta(asset_returns: pd.Series,
                 benchmark_returns: pd.Series) -> float:
    """Compute the CAPM beta of an asset relative to a benchmark index.

    Beta measures the systematic risk of an asset: how much the asset’s
    returns co‑move with the market returns【985233146337732†L160-L207】.  A beta
    greater than 1 indicates higher volatility relative to the market.

    Parameters
    ----------
    asset_returns : pd.Series
        Returns of the asset.
    benchmark_returns : pd.Series
        Returns of the benchmark index.

    Returns
    -------
    float
        The beta coefficient.
    """
    # Align on common dates
    combined = pd.concat([asset_returns, benchmark_returns], axis=1, join='inner').dropna()
    asset_r = combined.iloc[:, 0]
    bench_r = combined.iloc[:, 1]
    covariance = np.cov(asset_r, bench_r, ddof=1)[0, 1]
    variance_bench = np.var(bench_r, ddof=1)
    if variance_bench == 0:
        return np.nan
    beta = covariance / variance_bench
    return beta
