"""
Option & Derivative Metrics
---------------------------
Comprehensive Black-Scholes-Merton option pricing, Greeks computation,
implied volatility analysis, and scenario modeling.

Fully integrated with the data fetching and processing pipeline.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from utils.data_processing import (
        load_current_session_data,
        get_current_session_tickers,
    )
except ImportError as e:
    print(f"Error: Could not import data processing utilities: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_RF_RATE = 0.02  # 2% annual
DEFAULT_SIGMA = 0.25    # 25% volatility
TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BSMResult:
    """Black-Scholes-Merton option price and Greeks."""
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    lambda_: float  # elasticity/leverage


@dataclass
class OptionPosition:
    """Specification for an option position."""
    option_type: str  # 'call' or 'put'
    strike: float
    maturity: float
    quantity: float = 1.0


# ---------------------------------------------------------------------------
# Black-Scholes-Merton implementation
# ---------------------------------------------------------------------------

def bsm_price_and_greeks(S: float, K: float, T: float, r: float, sigma: float,
                         q: float = 0.0, option_type: str = "call") -> BSMResult:
    """
    Calculate Black-Scholes-Merton option price and Greeks.
    
    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate (continuous)
    sigma : float
        Volatility (annualized)
    q : float
        Dividend yield (continuous)
    option_type : str
        'call' or 'put'
    
    Returns
    -------
    BSMResult
    """
    # Handle edge cases
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return BSMResult(intrinsic, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    if sigma <= 0 or S <= 0 or K <= 0:
        return BSMResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Discount factors
    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)
    
    # Calculate price and Greeks
    if option_type == "call":
        price = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
        delta = df_q * norm.cdf(d1)
        rho = K * T * df_r * norm.cdf(d2)
        theta = (-(S * norm.pdf(d1) * sigma * df_q) / (2 * np.sqrt(T))
                 - r * K * df_r * norm.cdf(d2)
                 + q * S * df_q * norm.cdf(d1))
    else:  # put
        price = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
        delta = -df_q * norm.cdf(-d1)
        rho = -K * T * df_r * norm.cdf(-d2)
        theta = (-(S * norm.pdf(d1) * sigma * df_q) / (2 * np.sqrt(T))
                 + r * K * df_r * norm.cdf(-d2)
                 - q * S * df_q * norm.cdf(-d1))
    
    # Greeks common to both
    gamma = (norm.pdf(d1) * df_q) / (S * sigma * np.sqrt(T))
    vega = S * df_q * norm.pdf(d1) * np.sqrt(T)
    
    # Theta is per year; convert to per day
    theta_per_day = theta / 365.25
    
    # Lambda (elasticity)
    lambda_ = delta * S / price if price > 0 else 0
    
    return BSMResult(
        price=float(price),
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega / 100),  # vega per 1% vol change
        theta=float(theta_per_day),
        rho=float(rho / 100),  # rho per 1% rate change
        lambda_=float(lambda_)
    )


def implied_volatility(market_price: float, S: float, K: float, T: float,
                      r: float, q: float = 0.0, option_type: str = "call",
                      bounds: Tuple[float, float] = (0.001, 5.0)) -> float:
    """
    Calculate implied volatility using Brent's method.
    
    Returns
    -------
    float
        Implied volatility or np.nan if not found
    """
    def objective(sigma):
        return bsm_price_and_greeks(S, K, T, r, sigma, q, option_type).price - market_price
    
    try:
        return brentq(objective, bounds[0], bounds[1], xtol=1e-6, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan


def put_call_parity_check(call_price: float, put_price: float,
                          S: float, K: float, T: float,
                          r: float, q: float = 0.0) -> float:
    """
    Check put-call parity: C - P = S*e^(-qT) - K*e^(-rT)
    
    Returns
    -------
    float
        Parity violation (should be ~0)
    """
    lhs = call_price - put_price
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    return lhs - rhs


# ---------------------------------------------------------------------------
# Historical volatility estimation
# ---------------------------------------------------------------------------

def historical_volatility(returns: pd.Series, window: int = 30,
                         annualize: bool = True) -> pd.Series:
    """Calculate rolling historical volatility."""
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol


def realized_volatility(prices: pd.Series, horizon: int = 30) -> float:
    """Calculate realized volatility over a specific period."""
    returns = prices.pct_change().dropna()
    if len(returns) < horizon:
        return np.nan
    recent_returns = returns.iloc[-horizon:]
    return float(recent_returns.std() * np.sqrt(TRADING_DAYS))


# ---------------------------------------------------------------------------
# Option strategies
# ---------------------------------------------------------------------------

def bull_call_spread(S: float, K_long: float, K_short: float, T: float,
                    r: float, sigma: float, q: float = 0.0) -> Dict:
    """Price a bull call spread (long lower strike, short higher strike)."""
    long_call = bsm_price_and_greeks(S, K_long, T, r, sigma, q, "call")
    short_call = bsm_price_and_greeks(S, K_short, T, r, sigma, q, "call")
    
    cost = long_call.price - short_call.price
    max_profit = K_short - K_long - cost
    max_loss = cost
    
    return {
        'cost': cost,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'delta': long_call.delta - short_call.delta,
        'gamma': long_call.gamma - short_call.gamma,
        'vega': long_call.vega - short_call.vega,
        'theta': long_call.theta - short_call.theta
    }


def straddle(S: float, K: float, T: float, r: float, sigma: float,
            q: float = 0.0) -> Dict:
    """Price a straddle (long call + long put at same strike)."""
    call = bsm_price_and_greeks(S, K, T, r, sigma, q, "call")
    put = bsm_price_and_greeks(S, K, T, r, sigma, q, "put")
    
    cost = call.price + put.price
    
    return {
        'cost': cost,
        'delta': call.delta + put.delta,
        'gamma': call.gamma + put.gamma,
        'vega': call.vega + put.vega,
        'theta': call.theta + put.theta,
        'breakeven_up': K + cost,
        'breakeven_down': K - cost
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stock_data(base_name: str = "last_fetch") -> Tuple[pd.Series, pd.Series, str]:
    """
    Load stock prices and returns for analysis.
    
    Returns
    -------
    tuple
        (prices, returns, ticker)
    """
    prices_df, bundle = load_current_session_data(base_name=base_name)
    tickers = bundle['meta'].get('tickers', [])
    is_single = bundle.get('is_single_ticker', False)
    
    if not tickers:
        raise ValueError("No tickers found in session data")
    
    ticker = tickers[0]
    
    # Extract prices
    if is_single or not isinstance(prices_df.columns, pd.MultiIndex):
        prices = prices_df['Adj Close'] if 'Adj Close' in prices_df.columns else prices_df['Close']
    else:
        prices = prices_df[ticker]['Adj Close']
    
    prices = prices.ffill()
    returns = prices.pct_change().dropna()
    
    return prices, returns, ticker


def estimate_dividend_yield(bundle: Dict, ticker: str) -> float:
    """Estimate annual dividend yield from dividend history."""
    try:
        dividends = bundle.get('dividends', {}).get(ticker, pd.Series(dtype=float))
        if len(dividends) == 0:
            return 0.0
        
        # Get last year of dividends
        recent_divs = dividends[dividends.index > (dividends.index.max() - pd.Timedelta(days=365))]
        annual_div = recent_divs.sum()
        
        # Get current price
        prices = bundle['prices']
        is_single = bundle.get('is_single_ticker', False)
        
        if is_single or not isinstance(prices.columns, pd.MultiIndex):
            current_price = prices['Adj Close'].iloc[-1] if 'Adj Close' in prices.columns else prices['Close'].iloc[-1]
        else:
            current_price = prices[ticker]['Adj Close'].iloc[-1]
        
        yield_ = annual_div / current_price if current_price > 0 else 0.0
        return float(yield_)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Scenario analysis
# ---------------------------------------------------------------------------

def create_price_surface(S: float, r: float, sigma: float, q: float,
                        strikes: np.ndarray, maturities: np.ndarray,
                        option_type: str = "call") -> pd.DataFrame:
    """Create option price surface across strikes and maturities."""
    surface = np.zeros((len(strikes), len(maturities)))
    
    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            result = bsm_price_and_greeks(S, K, T, r, sigma, q, option_type)
            surface[i, j] = result.price
    
    return pd.DataFrame(surface, index=strikes, columns=maturities)


def create_greeks_surfaces(S: float, r: float, sigma: float, q: float,
                          strikes: np.ndarray, maturities: np.ndarray,
                          option_type: str = "call") -> Dict[str, pd.DataFrame]:
    """Create surfaces for all Greeks."""
    greeks = {'Delta': [], 'Gamma': [], 'Vega': [], 'Theta': [], 'Rho': []}
    
    for K in strikes:
        row_delta, row_gamma, row_vega, row_theta, row_rho = [], [], [], [], []
        for T in maturities:
            result = bsm_price_and_greeks(S, K, T, r, sigma, q, option_type)
            row_delta.append(result.delta)
            row_gamma.append(result.gamma)
            row_vega.append(result.vega)
            row_theta.append(result.theta)
            row_rho.append(result.rho)
        
        greeks['Delta'].append(row_delta)
        greeks['Gamma'].append(row_gamma)
        greeks['Vega'].append(row_vega)
        greeks['Theta'].append(row_theta)
        greeks['Rho'].append(row_rho)
    
    return {
        name: pd.DataFrame(data, index=strikes, columns=maturities)
        for name, data in greeks.items()
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_payoff_diagram(S: float, positions: List[Tuple[str, float, float]],
                       title: str = "Option Payoff") -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot payoff diagram for option strategy.
    
    Parameters
    ----------
    S : float
        Current stock price
    positions : List[Tuple[str, float, float]]
        List of (type, strike, premium) tuples
    """
    spot_range = np.linspace(S * 0.5, S * 1.5, 200)
    total_payoff = np.zeros_like(spot_range)
    total_cost = 0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for opt_type, K, premium in positions:
        if opt_type == 'call':
            payoff = np.maximum(spot_range - K, 0) - premium
        else:  # put
            payoff = np.maximum(K - spot_range, 0) - premium
        
        total_payoff += payoff
        total_cost += premium
        ax.plot(spot_range, payoff, '--', alpha=0.5, label=f'{opt_type.capitalize()} K={K:.0f}')
    
    ax.plot(spot_range, total_payoff, linewidth=2.5, color='darkblue', label='Total P&L')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(S, color='red', linestyle='--', linewidth=1, label=f'Current: {S:.2f}')
    
    ax.set_xlabel('Stock Price at Expiry', fontsize=12)
    ax.set_ylabel('Profit / Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, ax


def plot_greeks_by_strike(S: float, r: float, sigma: float, q: float,
                         strikes: np.ndarray, T: float,
                         option_type: str = "call") -> Tuple[plt.Figure, plt.Axes]:
    """Plot all Greeks as function of strike for fixed maturity."""
    deltas, gammas, vegas, thetas = [], [], [], []
    
    for K in strikes:
        result = bsm_price_and_greeks(S, K, T, r, sigma, q, option_type)
        deltas.append(result.delta)
        gammas.append(result.gamma)
        vegas.append(result.vega)
        thetas.append(result.theta)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(strikes, deltas, linewidth=2, color='blue')
    axes[0, 0].set_title('Delta')
    axes[0, 0].axvline(S, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(strikes, gammas, linewidth=2, color='green')
    axes[0, 1].set_title('Gamma')
    axes[0, 1].axvline(S, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(strikes, vegas, linewidth=2, color='purple')
    axes[1, 0].set_title('Vega')
    axes[1, 0].axvline(S, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(strikes, thetas, linewidth=2, color='orange')
    axes[1, 1].set_title('Theta (per day)')
    axes[1, 1].axvline(S, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.set_xlabel('Strike Price')
    
    fig.suptitle(f'Greeks vs Strike (T={T:.2f}y, S={S:.2f})', 
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig, axes


def plot_volatility_surface(strikes: np.ndarray, maturities: np.ndarray,
                           iv_surface: np.ndarray, title: str) -> Tuple[plt.Figure, plt.Axes]:
    """Plot implied volatility surface."""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(maturities, strikes)
    surf = ax.plot_surface(X, Y, iv_surface, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('Time to Maturity (years)', fontsize=10)
    ax.set_ylabel('Strike Price', fontsize=10)
    ax.set_zlabel('Implied Volatility', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    fig.colorbar(surf, ax=ax, shrink=0.5)
    return fig, ax


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def export_to_excel(output_path: Path,
                    summary: pd.DataFrame,
                    surfaces: Dict[str, pd.DataFrame],
                    images: List[Tuple[str, Path]]) -> None:
    """Export comprehensive results to Excel."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Formats
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'})
        number_fmt = workbook.add_format({'num_format': '0.0000'})
        
        # Summary
        summary.to_excel(writer, sheet_name='Summary', index=False)
        ws = writer.sheets['Summary']
        for col_num in range(len(summary.columns)):
            ws.write(0, col_num, summary.columns[col_num], header_fmt)
        
        # Surfaces
        for name, df in surfaces.items():
            df.to_excel(writer, sheet_name=name[:31])
        
        # Charts
        charts_ws = workbook.add_worksheet('Charts')
        row = 1
        for label, img_path in images:
            try:
                charts_ws.write(row, 1, label, header_fmt)
                charts_ws.insert_image(row + 1, 1, str(img_path),
                                      {'x_scale': 0.8, 'y_scale': 0.8})
                row += 30
            except Exception:
                row += 2
    
    print(f"✓ Excel report saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Option & Derivative Metrics")
    parser.add_argument("--base", type=str, default="last_fetch")
    parser.add_argument("--S", type=float, default=None, 
                       help="Override spot price")
    parser.add_argument("--r", type=float, default=DEFAULT_RF_RATE,
                       help="Risk-free rate (annual)")
    parser.add_argument("--sigma", type=float, default=None,
                       help="Volatility (if None, uses realized vol)")
    parser.add_argument("--strikes", type=str, default="0.8,0.9,1.0,1.1,1.2",
                       help="Strikes as multiples of spot")
    parser.add_argument("--maturities", type=str, default="0.25,0.5,1.0",
                       help="Maturities in years")
    parser.add_argument("--option-type", type=str, default="call",
                       choices=["call", "put"])
    parser.add_argument("--save-dir", type=str, default="results/options")
    parser.add_argument("--export-xlsx", type=str, default="reports/options_analysis.xlsx")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("OPTION & DERIVATIVE METRICS")
    print(f"{'='*60}\n")
    
    # Load data
    try:
        prices, returns, ticker = load_stock_data(args.base)
        prices_df, bundle = load_current_session_data(args.base)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Get parameters
    S = args.S if args.S is not None else float(prices.iloc[-1])
    q = estimate_dividend_yield(bundle, ticker)
    r = np.log(1 + args.r)  # Convert to continuous
    q_cc = np.log(1 + q) if q > 0 else 0
    
    if args.sigma is not None:
        sigma = args.sigma
    else:
        sigma = realized_volatility(prices, horizon=30)
        if np.isnan(sigma):
            sigma = DEFAULT_SIGMA
    
    print(f"Ticker: {ticker}")
    print(f"Spot Price: ${S:.2f}")
    print(f"Volatility: {sigma*100:.2f}%")
    print(f"Dividend Yield: {q*100:.2f}%")
    print(f"Risk-Free Rate: {args.r*100:.2f}%\n")
    
    # Parse strikes and maturities
    strike_multiples = [float(x) for x in args.strikes.split(',')]
    strikes = np.array(strike_multiples) * S
    maturities = np.array([float(x) for x in args.maturities.split(',')])
    
    # Create surfaces
    print("Generating option surfaces...")
    price_surface = create_price_surface(S, r, sigma, q_cc, strikes, maturities, args.option_type)
    greeks_surfaces = create_greeks_surfaces(S, r, sigma, q_cc, strikes, maturities, args.option_type)
    
    # Calculate sample options
    summary_rows = []
    for K in strikes:
        for T in maturities:
            result = bsm_price_and_greeks(S, K, T, r, sigma, q_cc, args.option_type)
            summary_rows.append({
                'Strike': K,
                'Maturity': T,
                'Price': result.price,
                'Delta': result.delta,
                'Gamma': result.gamma,
                'Vega': result.vega,
                'Theta': result.theta,
                'Rho': result.rho
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Generate plots
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    images = []
    
    # Greeks by strike
    fig, _ = plot_greeks_by_strike(S, r, sigma, q_cc, strikes, maturities[0], args.option_type)
    img_path = outdir / f"{ticker}_greeks_by_strike.png"
    fig.savefig(img_path, dpi=120, bbox_inches='tight')
    images.append((f"{ticker} Greeks by Strike", img_path))
    plt.close(fig)
    
    # Payoff diagram
    positions = [(args.option_type, strikes[len(strikes)//2], price_surface.iloc[len(strikes)//2, 0])]
    fig, _ = plot_payoff_diagram(S, positions, f"{ticker} {args.option_type.capitalize()} Payoff")
    img_path = outdir / f"{ticker}_payoff.png"
    fig.savefig(img_path, dpi=120, bbox_inches='tight')
    images.append((f"{ticker} Payoff", img_path))
    plt.close(fig)
    
    # Export
    surfaces = {'Prices': price_surface}
    surfaces.update(greeks_surfaces)
    
    export_path = Path(args.export_xlsx)
    export_to_excel(export_path, summary_df, surfaces, images)
    
    print(f"\n{'='*60}")
    print("SAMPLE OPTION PRICES")
    print(f"{'='*60}")
    print(summary_df.head(10).to_string(index=False))
    print(f"\n{'='*60}")
    print(f"✓ Analysis complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()