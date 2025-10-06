"""
Portfolio risk analysis tools.

This module extends the single‑stock risk metrics to portfolios by combining
asset returns with portfolio weights.  Functions include portfolio Value at
Risk (VaR), Sharpe ratio, beta and Monte Carlo simulation of portfolio
value paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Literal, Optional

from scipy.stats import norm

from ..single_stock import risk_metrics as smr


def compute_portfolio_returns(returns_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Compute portfolio returns given individual asset returns and weights.

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame of returns for each asset (columns = tickers).
    weights : np.ndarray
        Weight vector (sum to 1).  Order must match columns of returns_df.

    Returns
    -------
    pd.Series
        Series of portfolio returns.
    """
    # Ensure weight vector matches the number of assets
    weights = np.array(weights)
    if len(weights) != returns_df.shape[1]:
        raise ValueError("weights length must match number of assets")
    # Compute weighted returns
    port_returns = returns_df.values @ weights
    return pd.Series(port_returns, index=returns_df.index)


def portfolio_var(returns_df: pd.DataFrame,
                 weights: np.ndarray,
                 confidence_level: float = 0.95,
                 method: Literal['historical', 'parametric', 'monte_carlo'] = 'historical',
                 num_simulations: int = 10000) -> float:
    """Compute portfolio Value at Risk (VaR).

    This function first aggregates individual returns into a portfolio return
    series using the provided weights and then applies the single‑stock
    VaR computation.
    """
    port_returns = compute_portfolio_returns(returns_df, weights)
    return smr.compute_var(port_returns, confidence_level=confidence_level,
                           method=method, num_simulations=num_simulations)


def portfolio_sharpe(returns_df: pd.DataFrame,
                     weights: np.ndarray,
                     risk_free_rate: float = 0.0,
                     freq: int = 252) -> float:
    """Compute the Sharpe ratio of the portfolio.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Individual asset returns.
    weights : np.ndarray
        Portfolio weights.
    risk_free_rate : float, default 0.0
        Risk free rate per period.
    freq : int, default 252
        Periods per year.
    """
    port_returns = compute_portfolio_returns(returns_df, weights)
    return smr.compute_sharpe_ratio(port_returns, risk_free_rate=risk_free_rate, freq=freq)


def portfolio_beta(returns_df: pd.DataFrame,
                   weights: np.ndarray,
                   benchmark_returns: pd.Series) -> float:
    """Compute portfolio beta relative to a benchmark index.

    The portfolio beta is the weighted sum of individual betas of the assets.
    This function aggregates the asset returns to a portfolio and then
    computes beta using the CAPM formula.
    """
    port_returns = compute_portfolio_returns(returns_df, weights)
    return smr.compute_beta(port_returns, benchmark_returns)


def simulate_portfolio_paths(price_df: pd.DataFrame,
                             weights: np.ndarray,
                             forecast_days: int = 252,
                             num_simulations: int = 10000) -> np.ndarray:
    """Simulate future portfolio values using Monte Carlo based on GBM.

    Parameters
    ----------
    price_df : pd.DataFrame
        DataFrame of historical prices for each asset (columns = tickers).
    weights : np.ndarray
        Portfolio weights.
    forecast_days : int, default 252
        Number of trading days to simulate.
    num_simulations : int, default 10000
        Number of Monte Carlo simulations.

    Returns
    -------
    np.ndarray
        Simulated final portfolio values for each simulation.
    """
    # Compute log returns to estimate drift and volatility for each asset
    log_returns = np.log(price_df / price_df.shift(1)).dropna()
    mu = log_returns.mean().values
    sigma = log_returns.cov().values
    # Cholesky decomposition to introduce correlations
    chol = np.linalg.cholesky(sigma)
    n_assets = price_df.shape[1]
    # Starting prices
    last_prices = price_df.iloc[-1].values
    # Simulate correlated random normal numbers
    random_normals = np.random.normal(size=(num_simulations, forecast_days, n_assets))
    correlated_normals = random_normals @ chol.T
    # Initialize array to store portfolio values
    portfolio_values = np.zeros(num_simulations)
    for i in range(num_simulations):
        # For each simulation, compute price paths for each asset
        prices = last_prices.copy()
        for t in range(forecast_days):
            # Update prices using GBM: P(t+1) = P(t) * exp((mu - 0.5*diag(sigma)) + chol*Z)
            drift = (mu - 0.5 * np.diag(sigma))
            shocks = correlated_normals[i, t]
            prices = prices * np.exp(drift + shocks)
        # Compute portfolio value at final time
        portfolio_values[i] = np.dot(prices, weights)
    return portfolio_values
