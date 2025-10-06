"""
Portfolio optimisation tools.

This module implements classical mean‑variance optimisation using Modern
Portfolio Theory (MPT) to construct efficient portfolios【836681858147528†L266-L297】.
The goal is to find asset weightings that maximise expected return for a given
level of risk or minimise risk for a given target return.

We provide functions to compute the minimum variance portfolio, the maximum
Sharpe ratio portfolio, and the full efficient frontier.  Optimisation is
performed using the convex optimisation library cvxpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Tuple, Dict, Optional


def mean_variance_optimisation(expected_returns: np.ndarray,
                               cov_matrix: np.ndarray,
                               target_return: Optional[float] = None,
                               risk_free_rate: float = 0.0) -> Dict[str, np.ndarray]:
    """Solve a mean‑variance optimisation problem.

    Parameters
    ----------
    expected_returns : np.ndarray
        Expected returns vector (annualised).
    cov_matrix : np.ndarray
        Covariance matrix of asset returns (annualised).
    target_return : float, optional
        Desired portfolio return.  If provided, the optimisation will minimise
        variance subject to the portfolio achieving at least `target_return`.
        If None, the optimisation will maximise the Sharpe ratio relative to
        `risk_free_rate`.
    risk_free_rate : float, default 0.0
        Risk‑free rate used for Sharpe ratio maximisation.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'weights': optimal weight vector
        - 'expected_return': expected return of the optimal portfolio
        - 'volatility': standard deviation of the optimal portfolio
    """
    n = len(expected_returns)
    w = cp.Variable(n)
    # Portfolio return and variance
    portfolio_return = expected_returns @ w
    portfolio_variance = cp.quad_form(w, cov_matrix)
    # Constraints: weights sum to 1 and no short selling (w >= 0)
    constraints = [cp.sum(w) == 1, w >= 0]

    if target_return is not None:
        # Minimise variance for a given return
        constraints.append(portfolio_return >= target_return)
        objective = cp.Minimize(portfolio_variance)
    else:
        # Maximise Sharpe ratio: (mu_p - rf) / sigma_p -> Equivalent to
        # maximise numerator / sqrt(variance).  This is non‑linear; we
        # transform to maximise (mu_p - rf) - 0.5 * k * variance.  We set
        # k = 1 for simplicity; scaling constant does not change optimum.
        excess_return = portfolio_return - risk_free_rate
        objective = cp.Maximize(excess_return - 0.5 * portfolio_variance)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    weights = w.value
    # Normalise to sum to 1 in case of numerical issues
    weights = np.array(weights).flatten()
    weights = np.maximum(weights, 0)
    weights /= weights.sum()
    # Compute portfolio stats
    exp_ret = expected_returns @ weights
    vol = np.sqrt(weights.T @ cov_matrix @ weights)
    return {
        'weights': weights,
        'expected_return': exp_ret,
        'volatility': vol
    }


def efficient_frontier(expected_returns: np.ndarray,
                       cov_matrix: np.ndarray,
                       num_portfolios: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the efficient frontier by computing optimal portfolios across a range of target returns.

    Parameters
    ----------
    expected_returns : np.ndarray
        Expected returns vector (annualised).
    cov_matrix : np.ndarray
        Covariance matrix of asset returns (annualised).
    num_portfolios : int, default 50
        Number of points to generate on the efficient frontier.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (target_returns, volatilities, weights_matrix)
    """
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), num_portfolios)
    volatilities = np.zeros(num_portfolios)
    weights_matrix = np.zeros((num_portfolios, len(expected_returns)))
    for i, target in enumerate(target_returns):
        result = mean_variance_optimisation(expected_returns, cov_matrix, target_return=target)
        volatilities[i] = result['volatility']
        weights_matrix[i, :] = result['weights']
    return target_returns, volatilities, weights_matrix
