"""
Comprehensive data processing and feature engineering for financial time series.

This module computes and caches a full suite of descriptive statistics, risk metrics,
and derived features for stock analysis. All computed metrics are stored for reuse
in forecasting, simulation, and portfolio analysis.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from typing import Dict, Optional, Tuple
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

# ---- USER CONFIG -----------------------------------------------------------
USER_TICKERS: list[str] = ["AAPL", "MSFT"]
# ---------------------------------------------------------------------------


def forward_fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing values in the provided DataFrame."""
    return df.ffill()


def resample_to_business_days(df: pd.DataFrame, how: str = 'last') -> pd.DataFrame:
    """Resample the DataFrame to a business day frequency."""
    if how == 'last':
        return df.resample('B').last().ffill()
    elif how == 'first':
        return df.resample('B').first().ffill()
    elif how == 'mean':
        return df.resample('B').mean().ffill()
    else:
        raise ValueError(f"Unsupported aggregation method: {how}")


def get_current_session_tickers(base_name: str = "last_fetch") -> list[str] | None:
    """Return the list of tickers fetched in the current session from the meta JSON."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    meta_path = data_dir / f"{base_name}_meta.json"
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        mts = meta.get("tickers")
        if isinstance(mts, list) and len(mts) > 0:
            return mts
    except Exception:
        pass
    return None


def load_current_session_data(base_name: str = "last_fetch") -> Tuple[pd.DataFrame, Dict]:
    """
    Load the complete data bundle from the current session.
    
    Returns
    -------
    tuple
        (prices_df, bundle_dict) where bundle_dict contains prices, dividends, 
        splits, actions, info, and meta
    """
    data_dir = Path(__file__).resolve().parents[1] / "data"
    
    # Try to load the full bundle
    try:
        # Load prices
        pq_path = data_dir / f"{base_name}.parquet"
        csv_path = data_dir / f"{base_name}.csv"
        
        if pq_path.exists():
            prices = pd.read_parquet(pq_path)
        elif csv_path.exists():
            try:
                prices = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
            except Exception:
                prices = pd.read_csv(csv_path, header=0, index_col=0, parse_dates=True)
        else:
            raise FileNotFoundError(f"No data file found at {pq_path} or {csv_path}")
        
        # Load metadata
        meta_path = data_dir / f"{base_name}_meta.json"
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Load info
        info_path = data_dir / f"{base_name}_info.json"
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        # Load corporate actions for each ticker
        tickers = meta.get('tickers', [])
        dividends, splits, actions = {}, {}, {}
        
        for t in tickers:
            try:
                div_path = data_dir / f"{base_name}_dividends_{t}.csv"
                if div_path.exists():
                    dividends[t] = pd.read_csv(div_path, index_col=0, parse_dates=True).iloc[:, 0]
                
                split_path = data_dir / f"{base_name}_splits_{t}.csv"
                if split_path.exists():
                    splits[t] = pd.read_csv(split_path, index_col=0, parse_dates=True).iloc[:, 0]
                
                action_path = data_dir / f"{base_name}_actions_{t}.csv"
                if action_path.exists():
                    actions[t] = pd.read_csv(action_path, index_col=0, parse_dates=True)
            except Exception:
                pass
        
        bundle = {
            'prices': prices,
            'dividends': dividends,
            'splits': splits,
            'actions': actions,
            'info': info,
            'meta': meta
        }
        
        # IMPORTANT: For single-ticker data, the prices DataFrame won't have ticker in columns
        # We need to track this so the processor knows how to handle it
        bundle['is_single_ticker'] = not isinstance(prices.columns, pd.MultiIndex)
        
        return prices, bundle
        
    except Exception as e:
        raise RuntimeError(f"Failed to load data bundle: {e}")


# ============================================================================
# COLUMN NORMALIZATION HELPER
# ============================================================================
def normalize_prices_columns_ticker_first(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure price DataFrame uses MultiIndex columns in (ticker, field) order.
    Some persisted downloads store (field, ticker). If detected, swap levels.
    """
    if isinstance(prices_df.columns, pd.MultiIndex):
        known_fields = {'Adj Close', 'Close', 'Open', 'High', 'Low', 'Volume'}
        level0 = {c[0] for c in prices_df.columns}
        level1 = {c[1] for c in prices_df.columns}
        # If level 0 looks like fields and level 1 looks like tickers, swap them.
        if level0.issubset(known_fields) and not level1.issubset(known_fields):
            prices_df = prices_df.swaplevel(0, 1, axis=1).sort_index(axis=1)
    return prices_df


# ============================================================================
# CORE COMPUTATIONAL FUNCTIONS
# ============================================================================

def compute_returns(prices: pd.Series) -> Dict[str, pd.Series]:
    """Compute both simple and log returns."""
    simple_returns = prices.pct_change()
    log_returns = np.log(prices / prices.shift(1))
    
    return {
        'simple_returns': simple_returns.dropna(),
        'log_returns': log_returns.dropna(),
        'cumulative_returns': (1 + simple_returns).cumprod() - 1
    }


def compute_descriptive_stats(returns: pd.Series, prices: pd.Series, freq: int = 252) -> Dict[str, float]:
    """Compute comprehensive descriptive statistics on returns."""
    clean_returns = returns.dropna()
    
    # Basic moments
    mean_daily = clean_returns.mean()
    median_daily = clean_returns.median()
    std_daily = clean_returns.std(ddof=1)
    variance_daily = clean_returns.var(ddof=1)
    
    # Annualized metrics
    mean_annual = mean_daily * freq
    std_annual = std_daily * np.sqrt(freq)
    
    # Higher moments
    skewness = clean_returns.skew()
    kurtosis = clean_returns.kurtosis()  # excess kurtosis
    
    # Risk-adjusted returns
    sharpe_ratio = (mean_daily / std_daily) * np.sqrt(freq) if std_daily > 0 else np.nan
    
    # Sortino ratio (downside deviation)
    downside_returns = clean_returns[clean_returns < 0]
    downside_std = downside_returns.std(ddof=1) if len(downside_returns) > 1 else np.nan
    sortino_ratio = (mean_daily / downside_std) * np.sqrt(freq) if downside_std > 0 else np.nan
    
    return {
        'mean_daily_return': mean_daily,
        'median_daily_return': median_daily,
        'std_daily_return': std_daily,
        'variance_daily_return': variance_daily,
        'mean_annual_return': mean_annual,
        'std_annual_return': std_annual,
        'skewness': skewness,
        'excess_kurtosis': kurtosis,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'min_return': clean_returns.min(),
        'max_return': clean_returns.max(),
        'positive_days_pct': (clean_returns > 0).sum() / len(clean_returns) * 100,
        'negative_days_pct': (clean_returns < 0).sum() / len(clean_returns) * 100,
    }


def compute_risk_metrics(returns: pd.Series, confidence_levels: list = [0.95, 0.99]) -> Dict[str, float]:
    """Compute VaR, CVaR, and drawdown metrics."""
    clean_returns = returns.dropna()
    
    risk_metrics = {}
    
    # Value at Risk (Historical method)
    for conf in confidence_levels:
        var = np.quantile(clean_returns, 1 - conf)
        risk_metrics[f'var_{int(conf*100)}'] = var
        
        # Conditional VaR (Expected Shortfall)
        cvar = clean_returns[clean_returns <= var].mean()
        risk_metrics[f'cvar_{int(conf*100)}'] = cvar
    
    # Maximum Drawdown
    cum_returns = (1 + clean_returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Average Drawdown
    avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
    
    risk_metrics.update({
        'max_drawdown': max_drawdown,
        'avg_drawdown': avg_drawdown,
        'current_drawdown': drawdown.iloc[-1] if len(drawdown) > 0 else 0
    })
    
    return risk_metrics


def compute_rolling_statistics(returns: pd.Series, windows: list = [21, 63, 252]) -> Dict[str, pd.Series]:
    """Compute rolling statistics for various windows."""
    rolling_stats = {}
    
    for window in windows:
        rolling_stats[f'rolling_mean_{window}d'] = returns.rolling(window).mean()
        rolling_stats[f'rolling_std_{window}d'] = returns.rolling(window).std()
        rolling_stats[f'rolling_vol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        rolling_stats[f'rolling_sharpe_{window}d'] = (
            returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        )
    
    return rolling_stats


def compute_autocorrelation(returns: pd.Series, max_lags: int = 20) -> Dict:
    """Compute autocorrelation and test for patterns."""
    clean_returns = returns.dropna()
    
    # Autocorrelation coefficients
    acf_values = [clean_returns.autocorr(lag=i) for i in range(1, max_lags + 1)]
    
    # Ljung-Box test on returns (momentum test)
    try:
        lb_test_returns = acorr_ljungbox(clean_returns, lags=min(10, len(clean_returns)//5), return_df=True)
        lb_pvalue_returns = lb_test_returns['lb_pvalue'].iloc[-1]
    except Exception:
        lb_pvalue_returns = np.nan
    
    # Ljung-Box test on squared returns (volatility clustering)
    try:
        lb_test_squared = acorr_ljungbox(clean_returns**2, lags=min(10, len(clean_returns)//5), return_df=True)
        lb_pvalue_squared = lb_test_squared['lb_pvalue'].iloc[-1]
    except Exception:
        lb_pvalue_squared = np.nan
    
    return {
        'autocorr_lag1': acf_values[0] if len(acf_values) > 0 else np.nan,
        'autocorr_lag5': acf_values[4] if len(acf_values) >= 5 else np.nan,
        'autocorr_lag10': acf_values[9] if len(acf_values) >= 10 else np.nan,
        'ljungbox_returns_pvalue': lb_pvalue_returns,
        'ljungbox_squared_pvalue': lb_pvalue_squared,
        'has_momentum': lb_pvalue_returns < 0.05 if not np.isnan(lb_pvalue_returns) else False,
        'has_volatility_clustering': lb_pvalue_squared < 0.05 if not np.isnan(lb_pvalue_squared) else False,
    }


def compute_price_ratios(df: pd.DataFrame) -> pd.Series:
    """Compute intraday volatility proxies using high/low/close."""
    try:
        # Parkinson volatility estimator
        hl_ratio = np.log(df['High'] / df['Low'])**2 / (4 * np.log(2))
        return hl_ratio
    except KeyError:
        return pd.Series(dtype=float)


def compute_market_sensitivity(asset_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
    """Compute beta and alpha relative to a benchmark."""
    # Align dates
    combined = pd.concat([asset_returns, benchmark_returns], axis=1, join='inner').dropna()
    
    if len(combined) < 2:
        return {'beta': np.nan, 'alpha': np.nan, 'r_squared': np.nan}
    
    asset_r = combined.iloc[:, 0]
    bench_r = combined.iloc[:, 1]
    
    # Beta: covariance(asset, market) / variance(market)
    covariance = np.cov(asset_r, bench_r, ddof=1)[0, 1]
    variance_bench = np.var(bench_r, ddof=1)
    beta = covariance / variance_bench if variance_bench > 0 else np.nan
    
    # Alpha: asset_return - (risk_free + beta * (market_return - risk_free))
    # Simplified: alpha = mean(asset) - beta * mean(market)
    alpha = asset_r.mean() - beta * bench_r.mean() if not np.isnan(beta) else np.nan
    
    # R-squared
    correlation = asset_r.corr(bench_r)
    r_squared = correlation**2 if not np.isnan(correlation) else np.nan
    
    return {
        'beta': beta,
        'alpha_daily': alpha,
        'alpha_annual': alpha * 252,
        'r_squared': r_squared,
        'correlation': correlation
    }


def compute_liquidity_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute volume-based liquidity metrics."""
    try:
        volume = df['Volume']
        avg_volume = volume.mean()
        median_volume = volume.median()
        volume_std = volume.std()
        
        # Dollar volume
        close = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        dollar_volume = (volume * close).mean()
        
        return {
            'avg_daily_volume': avg_volume,
            'median_daily_volume': median_volume,
            'volume_std': volume_std,
            'avg_dollar_volume': dollar_volume,
        }
    except KeyError:
        return {}


def normality_tests(returns: pd.Series) -> Dict[str, float]:
    """Test if returns follow a normal distribution."""
    clean_returns = returns.dropna()
    
    # Jarque-Bera test
    jb_stat, jb_pvalue = stats.jarque_bera(clean_returns)
    
    # Shapiro-Wilk test (for smaller samples)
    if len(clean_returns) <= 5000:
        sw_stat, sw_pvalue = stats.shapiro(clean_returns)
    else:
        sw_stat, sw_pvalue = np.nan, np.nan
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.kstest(clean_returns, 'norm', 
                                       args=(clean_returns.mean(), clean_returns.std()))
    
    return {
        'jarque_bera_stat': jb_stat,
        'jarque_bera_pvalue': jb_pvalue,
        'is_normal_jb': jb_pvalue > 0.05,
        'shapiro_wilk_stat': sw_stat,
        'shapiro_wilk_pvalue': sw_pvalue,
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
    }


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

class StockDataProcessor:
    """Comprehensive processor for stock data analysis."""
    
    def __init__(self, ticker: str, prices_df: pd.DataFrame, bundle: Dict = None, benchmark_returns: pd.Series = None):
        """
        Initialize processor with price data and full bundle.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        prices_df : pd.DataFrame
            Price data (OHLCV)
        bundle : dict, optional
            Full data bundle from data_fetching (dividends, splits, actions, info)
        benchmark_returns : pd.Series, optional
            Benchmark returns for beta/alpha calculation
        """
        self.ticker = ticker
        self.prices_df = prices_df
        self.bundle = bundle or {}
        self.benchmark_returns = benchmark_returns

        # Ensure column orientation is (ticker, field) when MultiIndex
        if isinstance(self.prices_df.columns, pd.MultiIndex):
            self.prices_df = normalize_prices_columns_ticker_first(self.prices_df)
        
        # Determine if this is single-ticker or multi-ticker data
        is_single_ticker = self.bundle.get('is_single_ticker', False)
        
        # Extract adjusted close for main analysis
        if not is_single_ticker and isinstance(self.prices_df.columns, pd.MultiIndex):
            # Multi-ticker DataFrame: columns are (ticker, field)
            try:
                if ('Adj Close' in self.prices_df[ticker].columns):
                    self.close_prices = self.prices_df[ticker]['Adj Close']
                else:
                    self.close_prices = self.prices_df[ticker]['Close']
                self.full_data = self.prices_df[ticker]
            except KeyError:
                available = list(dict.fromkeys([c[0] for c in self.prices_df.columns]))
                raise ValueError(f"Ticker {ticker} not found in price data. Available: {available}")
        else:
            # Single-ticker DataFrame: columns are just field names
            self.close_prices = self.prices_df['Adj Close'] if 'Adj Close' in self.prices_df.columns else self.prices_df['Close']
            self.full_data = self.prices_df
        
        # Extract ticker-specific info from bundle
        self.dividends = self.bundle.get('dividends', {}).get(ticker, pd.Series(dtype=float))
        self.splits = self.bundle.get('splits', {}).get(ticker, pd.Series(dtype=float))
        self.actions = self.bundle.get('actions', {}).get(ticker, pd.DataFrame())
        self.info = self.bundle.get('info', {}).get(ticker, {})
        
        self.results = {}
        
    def process_all(self) -> Dict:
        """Run all computations and return comprehensive results."""
        print(f"Processing {self.ticker}...")
        
        # 1. Returns
        returns_data = compute_returns(self.close_prices)
        self.results['returns'] = returns_data
        
        # Use simple returns for most calculations
        returns = returns_data['simple_returns']
        
        # 2. Descriptive statistics
        desc_stats = compute_descriptive_stats(returns, self.close_prices)
        self.results['descriptive_stats'] = desc_stats
        
        # 3. Risk metrics
        risk_metrics = compute_risk_metrics(returns)
        self.results['risk_metrics'] = risk_metrics
        
        # 4. Rolling statistics
        rolling_stats = compute_rolling_statistics(returns)
        self.results['rolling_stats'] = rolling_stats
        
        # 5. Autocorrelation
        autocorr = compute_autocorrelation(returns)
        self.results['autocorrelation'] = autocorr
        
        # 6. Normality tests
        norm_tests = normality_tests(returns)
        self.results['normality_tests'] = norm_tests
        
        # 7. Price ratios (volatility proxy)
        price_ratios = compute_price_ratios(self.full_data)
        self.results['intraday_volatility'] = price_ratios
        
        # 8. Market sensitivity (if benchmark provided)
        if self.benchmark_returns is not None:
            market_sens = compute_market_sensitivity(returns, self.benchmark_returns)
            self.results['market_sensitivity'] = market_sens
        
        # 9. Liquidity metrics
        liquidity = compute_liquidity_metrics(self.full_data)
        self.results['liquidity'] = liquidity
        
        # 10. Corporate actions summary (from bundle)
        self.results['corporate_actions'] = {
            'num_dividends': len(self.dividends[self.dividends > 0]) if len(self.dividends) > 0 else 0,
            'total_dividends': self.dividends.sum() if len(self.dividends) > 0 else 0,
            'avg_dividend': self.dividends[self.dividends > 0].mean() if len(self.dividends[self.dividends > 0]) > 0 else 0,
            'num_splits': len(self.splits[self.splits != 0]) if len(self.splits) > 0 else 0,
        }
        
        # 11. Company info (from bundle)
        if self.info:
            self.results['company_info'] = {
                'sector': self.info.get('sector', 'N/A'),
                'industry': self.info.get('industry', 'N/A'),
                'market_cap': self.info.get('marketCap', None),
                'company_name': self.info.get('longName', self.ticker),
            }
        
        print(f"✓ Processing complete for {self.ticker}")
        return self.results
    
    def save_results(self, output_dir: str = "data/processed"):
        """Save all computed metrics to disk."""
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Save summary statistics as JSON
        summary = {
            'ticker': self.ticker,
            'descriptive_stats': self.results['descriptive_stats'],
            'risk_metrics': self.results['risk_metrics'],
            'autocorrelation': self.results['autocorrelation'],
            'normality_tests': self.results['normality_tests'],
            'corporate_actions': self.results.get('corporate_actions', {}),
        }
        
        if 'market_sensitivity' in self.results:
            summary['market_sensitivity'] = self.results['market_sensitivity']
        
        if 'liquidity' in self.results:
            summary['liquidity'] = self.results['liquidity']
        
        if 'company_info' in self.results:
            summary['company_info'] = self.results['company_info']
        
        with open(outdir / f"{self.ticker}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        
        # Save time series data as parquet
        returns_df = pd.DataFrame({
            'simple_returns': self.results['returns']['simple_returns'],
            'log_returns': self.results['returns']['log_returns'],
            'cumulative_returns': self.results['returns']['cumulative_returns'],
        })
        returns_df.to_parquet(outdir / f"{self.ticker}_returns.parquet")
        
        # Save rolling statistics
        rolling_df = pd.DataFrame(self.results['rolling_stats'])
        rolling_df.to_parquet(outdir / f"{self.ticker}_rolling_stats.parquet")
        
        # Save intraday volatility if available
        if len(self.results.get('intraday_volatility', [])) > 0:
            intraday_df = pd.DataFrame({'parkinson_volatility': self.results['intraday_volatility']})
            intraday_df.to_parquet(outdir / f"{self.ticker}_intraday_vol.parquet")
        
        # Save dividends and splits
        if len(self.dividends) > 0:
            self.dividends.to_csv(outdir / f"{self.ticker}_dividends.csv")
        if len(self.splits) > 0:
            self.splits.to_csv(outdir / f"{self.ticker}_splits.csv")
        
        # Save full results as pickle for complete reuse
        with open(outdir / f"{self.ticker}_full_results.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"✓ Results saved to {outdir}")
    
    def print_summary(self):
        """Print a formatted summary of key metrics."""
        print(f"\n{'='*60}")
        print(f"SUMMARY STATISTICS: {self.ticker}")
        if 'company_info' in self.results:
            info = self.results['company_info']
            print(f"{info.get('company_name', self.ticker)}")
            print(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")
        print(f"{'='*60}\n")
        
        desc = self.results['descriptive_stats']
        print("Performance Metrics:")
        print(f"  Mean Daily Return:      {desc['mean_daily_return']*100:>8.3f}%")
        print(f"  Mean Annual Return:     {desc['mean_annual_return']*100:>8.2f}%")
        print(f"  Annual Volatility:      {desc['std_annual_return']*100:>8.2f}%")
        print(f"  Sharpe Ratio:           {desc['sharpe_ratio']:>8.3f}")
        print(f"  Sortino Ratio:          {desc['sortino_ratio']:>8.3f}")
        
        risk = self.results['risk_metrics']
        print(f"\nRisk Metrics:")
        print(f"  VaR (95%):              {risk['var_95']*100:>8.3f}%")
        print(f"  CVaR (95%):             {risk['cvar_95']*100:>8.3f}%")
        print(f"  Max Drawdown:           {risk['max_drawdown']*100:>8.2f}%")
        
        print(f"\nDistribution Characteristics:")
        print(f"  Skewness:               {desc['skewness']:>8.3f}")
        print(f"  Excess Kurtosis:        {desc['excess_kurtosis']:>8.3f}")
        print(f"  Positive Days:          {desc['positive_days_pct']:>8.2f}%")
        
        if 'market_sensitivity' in self.results:
            mkt = self.results['market_sensitivity']
            print(f"\nMarket Sensitivity:")
            print(f"  Beta:                   {mkt['beta']:>8.3f}")
            print(f"  Alpha (annual):         {mkt['alpha_annual']*100:>8.2f}%")
            print(f"  R-squared:              {mkt['r_squared']:>8.3f}")
        
        corp = self.results.get('corporate_actions', {})
        if corp.get('num_dividends', 0) > 0:
            print(f"\nCorporate Actions:")
            print(f"  Dividends Paid:         {corp['num_dividends']:>8.0f}")
            print(f"  Avg Dividend:           ${corp['avg_dividend']:>8.2f}")
        
        print(f"\n{'='*60}\n")


def reset_processing_context(base_name: str = "last_fetch") -> Tuple[pd.DataFrame, Dict, list[str]]:
    """
    Reset processing to use the tickers from the current fetch session.
    
    Returns
    -------
    tuple
        (prices_df, bundle_dict, tickers_list)
    """
    prices, bundle = load_current_session_data(base_name=base_name)
    # Normalize column orientation to (ticker, field) if needed
    prices = normalize_prices_columns_ticker_first(prices)
    tickers = bundle['meta'].get('tickers', [])
    print(f"Data processing context reset. Using tickers: {tickers}")
    print(f"Date range: {bundle['meta'].get('start')} to {bundle['meta'].get('end')}")
    try:
        print(f"Normalized columns (first 6): {list(prices.columns)[:6]}")
    except Exception:
        pass
    return prices, bundle, tickers


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive stock data processing and analysis.")
    parser.add_argument("--base", type=str, default="last_fetch", help="Base name of data bundle")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--benchmark", type=str, default=None, 
                        help="Benchmark ticker for beta/alpha calculation (e.g., SPY, ^GSPC)")
    args = parser.parse_args()
    
    # Load current session data with full bundle
    try:
        prices, bundle, tickers = reset_processing_context(base_name=args.base)
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"\nMake sure you've run data_fetching.py first to fetch and save data.")
        print(f"Example: python data_fetching.py --tickers AAPL,MSFT")
        exit(1)
    
    if not tickers:
        print("No tickers found in the current session. Please run data_fetching.py first.")
        exit(1)
    
    # Load benchmark if provided
    benchmark_returns = None
    if args.benchmark:
        try:
            import yfinance as yf
            # Get date range from metadata
            start_date = bundle['meta'].get('start')
            end_date = bundle['meta'].get('end')
            bench_data = yf.download(args.benchmark, start=start_date, end=end_date, progress=False)
            benchmark_returns = bench_data['Adj Close'].pct_change().dropna()
            print(f"✓ Loaded benchmark: {args.benchmark}")
        except Exception as e:
            print(f"Warning: Could not load benchmark {args.benchmark}: {e}")
    
    # Process each ticker
    print(f"\nProcessing {len(tickers)} ticker(s)...\n")
    
    # Check if data is single-ticker format
    is_single_ticker = bundle.get('is_single_ticker', False)
    
    for ticker in tickers:
        try:
            # For multi-ticker data, verify ticker exists in columns
            if not is_single_ticker and isinstance(prices.columns, pd.MultiIndex):
                available_tickers = list(dict.fromkeys([c[0] for c in prices.columns]))
                if ticker not in available_tickers:
                    print(f"✗ Error: Ticker {ticker} not found in prices data. Available: {available_tickers}")
                    continue
            
            print(f"Processing {ticker}...")
            processor = StockDataProcessor(ticker, prices, bundle, benchmark_returns)
            processor.process_all()
            processor.print_summary()
            processor.save_results(output_dir=args.output_dir)
        except Exception as e:
            import traceback
            print(f"✗ Error processing {ticker}: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ All processing complete!")
    print(f"  Processed: {len(tickers)} ticker(s)")
    print(f"  Results saved to: {args.output_dir}/")
    print(f"{'='*60}\n")