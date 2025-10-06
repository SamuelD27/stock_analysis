"""
Event & Factor Sensitivity Analysis
------------------------------------
Comprehensive single-stock factor exposure analysis and event studies.
Fully integrated with the data fetching and processing pipeline.

Produces:
- CAPM and multi-factor regression results
- Rolling beta analysis
- Event study (market model abnormal returns)
- Comprehensive Excel report with embedded charts
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings

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
    print("Make sure you're running from the project root directory.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_BENCHMARK: str = "^GSPC"  # S&P 500
DEFAULT_RF_ANNUAL: float = 0.02   # 2% annual risk-free rate
DEFAULT_ROLLING_WINDOW: int = 252  # 1 year

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RegressionResult:
    """Results from a factor regression."""
    alpha: float
    alpha_tstat: float
    alpha_pvalue: float
    betas: Dict[str, float]
    beta_tstats: Dict[str, float]
    beta_pvalues: Dict[str, float]
    r_squared: float
    adj_r_squared: float
    n_obs: int
    residuals: pd.Series
    fitted: pd.Series


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_asset_data(base_name: str = "last_fetch") -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load prices and returns for current session tickers.
    
    Returns
    -------
    tuple
        (prices_df, returns_df, ticker_list)
    """
    prices, bundle = load_current_session_data(base_name=base_name)
    tickers = bundle['meta'].get('tickers', [])
    is_single = bundle.get('is_single_ticker', False)
    
    # Extract adjusted close prices
    if is_single or not isinstance(prices.columns, pd.MultiIndex):
        px = prices['Adj Close'] if 'Adj Close' in prices.columns else prices['Close']
        px = px.to_frame(name=tickers[0] if tickers else 'Asset')
    else:
        px = prices.xs('Adj Close', level=1, axis=1)
    
    # Compute returns
    px = px.ffill()
    rets = px.pct_change().dropna(how='all')
    
    return px, rets, list(rets.columns)


def fetch_benchmark_data(ticker: str, start: str, end: str) -> Tuple[pd.Series, pd.Series]:
    """
    Fetch benchmark price and return data.
    
    Returns
    -------
    tuple
        (prices, returns)
    """
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            raise ValueError(f"No data returned for {ticker}")
        
        prices = data['Adj Close']
        returns = prices.pct_change().dropna()
        
        print(f"✓ Loaded benchmark {ticker}: {len(returns)} observations")
        return prices, returns
    except Exception as e:
        print(f"✗ Error loading benchmark {ticker}: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float)


def create_rf_series(index: pd.DatetimeIndex, annual_rate: float = DEFAULT_RF_ANNUAL) -> pd.Series:
    """Create daily risk-free rate series from annual rate."""
    daily_rf = (1.0 + annual_rate) ** (1/252) - 1.0
    return pd.Series(daily_rf, index=index, name='RF')


# ---------------------------------------------------------------------------
# Regression functions
# ---------------------------------------------------------------------------

def ols_regression(y: pd.Series, X: pd.DataFrame, add_constant: bool = True) -> RegressionResult:
    """
    Perform OLS regression with proper statistics.
    
    Parameters
    ----------
    y : pd.Series
        Dependent variable (excess returns)
    X : pd.DataFrame
        Independent variables (factors)
    add_constant : bool
        Whether to add intercept
    
    Returns
    -------
    RegressionResult
    """
    # Align data
    data = pd.concat([y, X], axis=1, join='inner').dropna()
    y_clean = data.iloc[:, 0].values
    X_clean = data.iloc[:, 1:].values
    
    if add_constant:
        X_clean = np.column_stack([np.ones(len(X_clean)), X_clean])
        param_names = ['alpha'] + list(X.columns)
    else:
        param_names = list(X.columns)
    
    # OLS estimation
    beta_hat = np.linalg.lstsq(X_clean, y_clean, rcond=None)[0]
    y_fitted = X_clean @ beta_hat
    residuals = y_clean - y_fitted
    
    # Statistics
    n = len(y_clean)
    k = X_clean.shape[1]
    df_resid = n - k
    
    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r2 = 1 - (1 - r2) * (n - 1) / df_resid if df_resid > 0 else r2
    
    # Standard errors (homoskedastic)
    mse = ss_res / df_resid if df_resid > 0 else np.nan
    var_covar = mse * np.linalg.inv(X_clean.T @ X_clean)
    se = np.sqrt(np.diag(var_covar))
    
    # t-statistics and p-values
    t_stats = beta_hat / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))
    
    # Package results
    if add_constant:
        alpha = beta_hat[0]
        alpha_tstat = t_stats[0]
        alpha_pval = p_values[0]
        betas = {name: beta_hat[i] for i, name in enumerate(param_names[1:], 1)}
        beta_tstats = {name: t_stats[i] for i, name in enumerate(param_names[1:], 1)}
        beta_pvals = {name: p_values[i] for i, name in enumerate(param_names[1:], 1)}
    else:
        alpha = 0.0
        alpha_tstat = 0.0
        alpha_pval = 1.0
        betas = {name: beta_hat[i] for i, name in enumerate(param_names)}
        beta_tstats = {name: t_stats[i] for i, name in enumerate(param_names)}
        beta_pvals = {name: p_values[i] for i, name in enumerate(param_names)}
    
    return RegressionResult(
        alpha=alpha,
        alpha_tstat=alpha_tstat,
        alpha_pvalue=alpha_pval,
        betas=betas,
        beta_tstats=beta_tstats,
        beta_pvalues=beta_pvals,
        r_squared=r2,
        adj_r_squared=adj_r2,
        n_obs=n,
        residuals=pd.Series(residuals, index=data.index, name='residuals'),
        fitted=pd.Series(y_fitted, index=data.index, name='fitted')
    )


def run_capm(asset_returns: pd.Series, 
             market_returns: pd.Series, 
             rf_series: pd.Series) -> RegressionResult:
    """Run CAPM regression: r_i - rf = alpha + beta * (r_m - rf)."""
    # Align data
    data = pd.concat([asset_returns, market_returns, rf_series], axis=1, join='inner').dropna()
    data.columns = ['asset', 'market', 'rf']
    
    y = data['asset'] - data['rf']
    X = pd.DataFrame({'Mkt-RF': data['market'] - data['rf']})
    
    return ols_regression(y, X)


def run_rolling_beta(asset_returns: pd.Series,
                     market_returns: pd.Series,
                     rf_series: pd.Series,
                     window: int = 252) -> pd.DataFrame:
    """
    Calculate rolling beta over time.
    
    Returns
    -------
    pd.DataFrame
        With columns: beta, alpha, r_squared
    """
    data = pd.concat([asset_returns, market_returns, rf_series], axis=1, join='inner').dropna()
    data.columns = ['asset', 'market', 'rf']
    
    results = []
    
    for i in range(window, len(data) + 1):
        window_data = data.iloc[i-window:i]
        y = window_data['asset'] - window_data['rf']
        x = window_data['market'] - window_data['rf']
        X_mat = np.column_stack([np.ones(len(x)), x.values])
        
        try:
            beta_hat = np.linalg.lstsq(X_mat, y.values, rcond=None)[0]
            y_fit = X_mat @ beta_hat
            ss_res = np.sum((y.values - y_fit) ** 2)
            ss_tot = np.sum((y.values - y.values.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            results.append({
                'date': window_data.index[-1],
                'beta': beta_hat[1],
                'alpha': beta_hat[0],
                'r_squared': r2
            })
        except Exception:
            continue
    
    return pd.DataFrame(results).set_index('date')


# ---------------------------------------------------------------------------
# Event study
# ---------------------------------------------------------------------------

def event_study(asset_returns: pd.Series,
                market_returns: pd.Series,
                rf_series: pd.Series,
                event_dates: List[pd.Timestamp],
                estimation_window: Tuple[int, int] = (-250, -30),
                event_window: Tuple[int, int] = (-10, 10)) -> pd.DataFrame:
    """
    Conduct event study using market model.
    
    Returns
    -------
    pd.DataFrame
        With columns: event_id, rel_day, AR, CAR
    """
    data = pd.concat([asset_returns, market_returns, rf_series], axis=1, join='inner').dropna()
    data.columns = ['asset', 'market', 'rf']
    
    results = []
    
    for event_id, event_date in enumerate(event_dates):
        if event_date not in data.index:
            print(f"Warning: Event date {event_date} not in data")
            continue
        
        # Get event position
        event_loc = data.index.get_loc(event_date)
        
        # Estimation window
        est_start = max(0, event_loc + estimation_window[0])
        est_end = max(0, event_loc + estimation_window[1])
        
        if est_end <= est_start:
            continue
        
        est_data = data.iloc[est_start:est_end+1]
        
        # Estimate parameters
        y_est = est_data['asset'] - est_data['rf']
        x_est = est_data['market'] - est_data['rf']
        X_est = np.column_stack([np.ones(len(x_est)), x_est.values])
        
        try:
            params = np.linalg.lstsq(X_est, y_est.values, rcond=None)[0]
            alpha, beta = params[0], params[1]
        except Exception:
            continue
        
        # Event window
        ew_start = max(0, event_loc + event_window[0])
        ew_end = min(len(data) - 1, event_loc + event_window[1])
        ew_data = data.iloc[ew_start:ew_end+1]
        
        # Calculate abnormal returns
        expected_ret = alpha + beta * (ew_data['market'] - ew_data['rf']) + ew_data['rf']
        ar = ew_data['asset'] - expected_ret
        car = ar.cumsum()
        
        # Store results with relative days
        rel_days = np.arange(event_window[0], event_window[0] + len(ar))
        
        for rel_day, (date, ar_val, car_val) in zip(rel_days, zip(ar.index, ar.values, car.values)):
            results.append({
                'event_id': event_id,
                'event_date': event_date,
                'date': date,
                'rel_day': rel_day,
                'AR': ar_val,
                'CAR': car_val
            })
    
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_capm_scatter(asset_returns: pd.Series,
                      market_returns: pd.Series,
                      rf_series: pd.Series,
                      ticker: str,
                      result: RegressionResult) -> Tuple[plt.Figure, plt.Axes]:
    """Plot CAPM scatter with regression line."""
    data = pd.concat([asset_returns, market_returns, rf_series], axis=1, join='inner').dropna()
    data.columns = ['asset', 'market', 'rf']
    
    x = (data['market'] - data['rf']).values
    y = (data['asset'] - data['rf']).values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x * 100, y * 100, alpha=0.5, s=10, label='Daily observations')
    
    # Regression line
    x_line = np.array([x.min(), x.max()])
    y_line = result.alpha + result.betas['Mkt-RF'] * x_line
    ax.plot(x_line * 100, y_line * 100, 'r-', linewidth=2, 
            label=f'CAPM: α={result.alpha*252*100:.2f}% (annual), β={result.betas["Mkt-RF"]:.3f}')
    
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel('Market Excess Return (%)', fontsize=12)
    ax.set_ylabel(f'{ticker} Excess Return (%)', fontsize=12)
    ax.set_title(f'CAPM Regression: {ticker}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add R² annotation
    ax.text(0.02, 0.98, f'R² = {result.r_squared:.3f}\nN = {result.n_obs}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    return fig, ax


def plot_rolling_beta(rolling_results: pd.DataFrame, ticker: str) -> Tuple[plt.Figure, plt.Axes]:
    """Plot rolling beta over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Beta
    ax1.plot(rolling_results.index, rolling_results['beta'], linewidth=1.5, color='darkblue')
    ax1.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='β = 1')
    ax1.fill_between(rolling_results.index, rolling_results['beta'], 1.0, 
                      alpha=0.3, color='lightblue')
    ax1.set_ylabel('Beta', fontsize=12)
    ax1.set_title(f'Rolling Beta: {ticker}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # R²
    ax2.plot(rolling_results.index, rolling_results['r_squared'], linewidth=1.5, color='darkgreen')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('Model Fit Quality', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, ax1


def plot_event_study(event_results: pd.DataFrame, ticker: str) -> Tuple[plt.Figure, plt.Axes]:
    """Plot cumulative abnormal returns around events."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each event
    for event_id in event_results['event_id'].unique():
        event_data = event_results[event_results['event_id'] == event_id]
        ax.plot(event_data['rel_day'], event_data['CAR'] * 100, 
                alpha=0.4, linewidth=1, color='gray')
    
    # Plot average CAR
    avg_car = event_results.groupby('rel_day')['CAR'].mean()
    ax.plot(avg_car.index, avg_car.values * 100, linewidth=2.5, 
            color='darkred', label='Average CAR')
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Days Relative to Event', fontsize=12)
    ax.set_ylabel('Cumulative Abnormal Return (%)', fontsize=12)
    ax.set_title(f'Event Study: {ticker}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def export_to_excel(output_path: Path,
                    capm_summary: pd.DataFrame,
                    rolling_beta_data: Dict[str, pd.DataFrame],
                    event_data: Optional[pd.DataFrame],
                    images: List[Tuple[str, Path]]) -> None:
    """Export comprehensive results to Excel."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Formats
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
        title_fmt = workbook.add_format({'bold': True, 'font_size': 14})
        number_fmt = workbook.add_format({'num_format': '0.0000'})
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        
        # CAPM Summary
        capm_summary.to_excel(writer, sheet_name='CAPM Summary', index=False)
        ws = writer.sheets['CAPM Summary']
        for col_num, col_name in enumerate(capm_summary.columns):
            ws.write(0, col_num, col_name, header_fmt)
        ws.set_column(0, 0, 15)
        ws.set_column(1, len(capm_summary.columns)-1, 12)
        
        # Rolling Beta
        if rolling_beta_data:
            for ticker, data in rolling_beta_data.items():
                sheet_name = f'{ticker}_Rolling'[:31]  # Excel limit
                data.to_excel(writer, sheet_name=sheet_name)
                ws = writer.sheets[sheet_name]
                ws.set_column(0, 0, 18)
                ws.set_column(1, data.shape[1], 12, number_fmt)
        
        # Event Study
        if event_data is not None and not event_data.empty:
            event_data.to_excel(writer, sheet_name='Event Study', index=False)
            ws = writer.sheets['Event Study']
            ws.set_column(0, 0, 12)
            ws.set_column(1, event_data.shape[1]-1, 15)
        
        # Charts
        charts_ws = workbook.add_worksheet('Charts')
        row = 1
        col = 1
        for label, img_path in images:
            try:
                charts_ws.write(row, col, label, title_fmt)
                charts_ws.insert_image(row + 1, col, str(img_path), 
                                      {'x_scale': 0.85, 'y_scale': 0.85})
                row += 32
            except Exception as e:
                print(f"Warning: Could not insert {label}: {e}")
                row += 2
    
    print(f"✓ Excel report saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Event & Factor Sensitivity Analysis")
    parser.add_argument("--base", type=str, default="last_fetch",
                       help="Data bundle base name")
    parser.add_argument("--benchmark", type=str, default=DEFAULT_BENCHMARK,
                       help="Benchmark ticker (default: ^GSPC)")
    parser.add_argument("--events", type=str, default=None,
                       help="Comma-separated event dates (YYYY-MM-DD)")
    parser.add_argument("--save-dir", type=str, default="results/sensitivity",
                       help="Directory for figures")
    parser.add_argument("--export-xlsx", type=str, default="reports/sensitivity_analysis.xlsx",
                       help="Excel output path")
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW,
                       help="Rolling beta window (days)")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("EVENT & FACTOR SENSITIVITY ANALYSIS")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading asset data...")
    try:
        prices, returns, tickers = load_asset_data(base_name=args.base)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Make sure you've run data_fetching.py first.")
        return
    
    if not tickers:
        print("✗ No tickers found")
        return
    
    print(f"✓ Loaded {len(tickers)} ticker(s): {tickers}")
    
    # Load benchmark
    start_date = returns.index.min().strftime('%Y-%m-%d')
    end_date = returns.index.max().strftime('%Y-%m-%d')
    
    print(f"Loading benchmark {args.benchmark}...")
    bench_prices, bench_returns = fetch_benchmark_data(args.benchmark, start_date, end_date)
    
    if bench_returns.empty:
        print("✗ Failed to load benchmark data")
        return
    
    # Create risk-free series
    rf = create_rf_series(returns.index)
    
    # Setup output
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    images: List[Tuple[str, Path]] = []
    capm_rows = []
    rolling_beta_data = {}
    all_event_results = []
    
    # Parse events
    event_dates = None
    if args.events:
        try:
            event_dates = [pd.to_datetime(d).tz_localize(None) for d in args.events.split(',')]
            print(f"✓ Parsed {len(event_dates)} event date(s)")
        except Exception as e:
            print(f"Warning: Could not parse event dates: {e}")
    
    # Analyze each ticker
    for ticker in tickers:
        print(f"\n--- Analyzing {ticker} ---")
        asset_ret = returns[ticker].dropna()
        
        # CAPM
        capm_result = run_capm(asset_ret, bench_returns, rf)
        
        capm_rows.append({
            'Ticker': ticker,
            'Alpha (daily)': capm_result.alpha,
            'Alpha (annual)': capm_result.alpha * 252,
            'Alpha t-stat': capm_result.alpha_tstat,
            'Alpha p-value': capm_result.alpha_pvalue,
            'Beta': capm_result.betas['Mkt-RF'],
            'Beta t-stat': capm_result.beta_tstats['Mkt-RF'],
            'Beta p-value': capm_result.beta_pvalues['Mkt-RF'],
            'R²': capm_result.r_squared,
            'Adj R²': capm_result.adj_r_squared,
            'N': capm_result.n_obs
        })
        
        print(f"  Beta: {capm_result.betas['Mkt-RF']:.3f}")
        print(f"  Alpha (annual): {capm_result.alpha * 252 * 100:.2f}%")
        print(f"  R²: {capm_result.r_squared:.3f}")
        
        # CAPM scatter plot
        fig, _ = plot_capm_scatter(asset_ret, bench_returns, rf, ticker, capm_result)
        img_path = outdir / f"{ticker}_capm_scatter.png"
        fig.savefig(img_path, dpi=120, bbox_inches='tight')
        images.append((f"{ticker} CAPM Scatter", img_path))
        plt.close(fig)
        
        # Rolling beta
        print(f"  Computing rolling beta (window={args.rolling_window})...")
        rolling_results = run_rolling_beta(asset_ret, bench_returns, rf, window=args.rolling_window)
        rolling_beta_data[ticker] = rolling_results
        
        fig, _ = plot_rolling_beta(rolling_results, ticker)
        img_path = outdir / f"{ticker}_rolling_beta.png"
        fig.savefig(img_path, dpi=120, bbox_inches='tight')
        images.append((f"{ticker} Rolling Beta", img_path))
        plt.close(fig)
        
        # Event study
        if event_dates:
            print(f"  Running event study...")
            event_results = event_study(asset_ret, bench_returns, rf, event_dates)
            if not event_results.empty:
                event_results['ticker'] = ticker
                all_event_results.append(event_results)
                
                fig, _ = plot_event_study(event_results, ticker)
                img_path = outdir / f"{ticker}_event_study.png"
                fig.savefig(img_path, dpi=120, bbox_inches='tight')
                images.append((f"{ticker} Event Study", img_path))
                plt.close(fig)
    
    # Create summary DataFrames
    capm_summary = pd.DataFrame(capm_rows)
    event_df = pd.concat(all_event_results) if all_event_results else None
    
    # Export to Excel
    export_path = Path(args.export_xlsx)
    export_to_excel(export_path, capm_summary, rolling_beta_data, event_df, images)
    
    # Console summary
    print(f"\n{'='*60}")
    print("CAPM REGRESSION RESULTS")
    print(f"{'='*60}")
    print(capm_summary.to_string(index=False))
    print(f"\n{'='*60}")
    print(f"✓ Analysis complete!")
    print(f"  Results: {export_path}")
    print(f"  Figures: {outdir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()