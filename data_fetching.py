"""
data_fetching.py
=================

Comprehensive data fetching module for stock analysis. Retrieves and persists:
- Historical OHLCV price data
- Dividends and stock splits
- Corporate actions
- Company metadata and fundamentals

All data is automatically saved to disk for reuse across analysis modules.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
import numpy as np
from typing import List, Union, Dict, Tuple
import datetime
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ---- USER CONFIG -----------------------------------------------------------
# Edit this line to choose the tickers you want to work with across the module.
# Example: ["AAPL", "MSFT"] or ["TSLA"] or ["GOOGL", "AMZN", "META"]
USER_TICKERS: list[str] = ["AAPL"]
# ---------------------------------------------------------------------------

# Directory to persist datasets for reuse
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Module-level variables that other scripts can import directly
LAST_FETCH_TICKERS: list[str] | None = None
LAST_FETCH_START: str | None = None
LAST_FETCH_END: str | None = None
LAST_FETCH_INTERVAL: str | None = None

LAST_FETCH_PRICES: pd.DataFrame | None = None
LAST_FETCH_DIVIDENDS: dict[str, pd.Series] | None = None
LAST_FETCH_SPLITS: dict[str, pd.Series] | None = None
LAST_FETCH_ACTIONS: dict[str, pd.DataFrame] | None = None
LAST_FETCH_INFO: dict[str, dict] | None = None


def download_data(tickers: Union[str, List[str]] | None = None,
                  start: str | None = None,
                  end: str | None = None,
                  interval: str = "1d") -> pd.DataFrame:
    """Download historical price data for one or more tickers from Yahoo Finance.

    Parameters
    ----------
    tickers : str or list of str
        The ticker symbol(s) to download. For multiple tickers, pass a list.
    start : str, default None
        The start date (YYYY-MM-DD). If None, defaults to ~1 year ago.
    end : str, default None
        The end date (YYYY-MM-DD). If None, defaults to today.
    interval : str, default "1d"
        Data interval: "1d", "1wk", "1mo", etc.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with OHLCV data.
    """
    today = datetime.date.today()
    if end is None:
        end = today.strftime("%Y-%m-%d")
    if start is None:
        one_year_ago = today - datetime.timedelta(days=365)
        start = one_year_ago.strftime("%Y-%m-%d")

    # Normalize tickers
    if tickers is None:
        tickers_list = list(USER_TICKERS)
    elif isinstance(tickers, str):
        tickers_list = [tickers]
    else:
        tickers_list = list(tickers)

    print(f"Fetching data for {tickers_list}...")
    print(f"Period: {start} to {end} | Interval: {interval}")

    # Download data via yfinance
    data = yf.download(tickers=tickers_list, start=start, end=end,
                       interval=interval, auto_adjust=False, progress=False)

    if data.empty:
        print(f"Warning: No data retrieved for {tickers_list}")
        return data

    # Handle single vs multiple tickers
    if len(tickers_list) == 1:
        single = tickers_list[0]
        if not data.empty:
            try:
                latest_price = float(data['Adj Close'].iloc[-1])
                print(f"Latest {single} price: {latest_price:.2f}")
            except Exception:
                print(f"Latest {single} price: {data['Adj Close'].iloc[-1]}")
        print("Data fetching completed successfully.")
        return data

    # For multiple tickers, swap levels so columns are (ticker, field)
    data = data.swaplevel(axis=1)
    data = data.sort_index(axis=1)
    
    latest_prices = data.xs('Adj Close', level=1, axis=1).iloc[-1]
    for t, p in latest_prices.items():
        try:
            print(f"Latest {t} price: {float(p):.2f}")
        except Exception:
            print(f"Latest {t} price: {p}")
    print("Data fetching completed successfully.")
    return data


def fetch_dividends_and_splits(ticker: str) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Fetch dividend and split data for a single ticker.
    
    Returns
    -------
    tuple
        (dividends_series, splits_series, actions_dataframe)
    """
    try:
        tk = yf.Ticker(ticker)
        
        # Get dividends
        dividends = tk.dividends if tk.dividends is not None else pd.Series(dtype=float)
        if not isinstance(dividends, pd.Series):
            dividends = pd.Series(dtype=float)
        
        # Get splits
        splits = tk.splits if tk.splits is not None else pd.Series(dtype=float)
        if not isinstance(splits, pd.Series):
            splits = pd.Series(dtype=float)
        
        # Get combined actions
        actions = tk.actions if tk.actions is not None else pd.DataFrame()
        if not isinstance(actions, pd.DataFrame):
            actions = pd.DataFrame()
        
        return dividends, splits, actions
    except Exception as e:
        print(f"Warning: Could not fetch dividends/splits for {ticker}: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame()


def fetch_company_info(ticker: str) -> Dict:
    """
    Fetch comprehensive company information and fundamentals.
    
    Returns
    -------
    dict
        Company metadata including sector, industry, market cap, etc.
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info if tk.info else {}
        
        # Extract key fields with fallbacks
        clean_info = {
            'symbol': info.get('symbol', ticker),
            'longName': info.get('longName', ticker),
            'shortName': info.get('shortName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'country': info.get('country', 'N/A'),
            'marketCap': info.get('marketCap', None),
            'enterpriseValue': info.get('enterpriseValue', None),
            'trailingPE': info.get('trailingPE', None),
            'forwardPE': info.get('forwardPE', None),
            'priceToBook': info.get('priceToBook', None),
            'dividendYield': info.get('dividendYield', None),
            'beta': info.get('beta', None),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', None),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', None),
            'averageVolume': info.get('averageVolume', None),
            'currency': info.get('currency', 'USD'),
            'exchange': info.get('exchange', 'N/A'),
            'quoteType': info.get('quoteType', 'EQUITY'),
            'website': info.get('website', None),
            'description': info.get('longBusinessSummary', None),
        }
        
        return clean_info
    except Exception as e:
        print(f"Warning: Could not fetch info for {ticker}: {e}")
        return {'symbol': ticker, 'error': str(e)}


def fetch_full_bundle(tickers: Union[str, List[str]] | None = None,
                      start: str | None = None,
                      end: str | None = None,
                      interval: str = "1d") -> Dict:
    """
    Fetch comprehensive data bundle including prices, corporate actions, and metadata.

    Returns
    -------
    dict
        Bundle containing:
        - 'prices': pd.DataFrame (OHLCV data)
        - 'dividends': dict[ticker -> pd.Series]
        - 'splits': dict[ticker -> pd.Series]
        - 'actions': dict[ticker -> pd.DataFrame]
        - 'info': dict[ticker -> dict] (company metadata)
        - 'meta': dict (fetch parameters and summary)
    """
    # Normalize tickers
    if tickers is None:
        tickers_list = list(USER_TICKERS)
    elif isinstance(tickers, str):
        tickers_list = [tickers]
    else:
        tickers_list = list(tickers)

    # Fetch price data
    prices = download_data(tickers=tickers_list, start=start, end=end, interval=interval)

    # Initialize containers
    dividends: dict[str, pd.Series] = {}
    splits: dict[str, pd.Series] = {}
    actions: dict[str, pd.DataFrame] = {}
    info: dict[str, dict] = {}

    print(f"\nFetching corporate actions and metadata for {len(tickers_list)} ticker(s)...")

    # Fetch additional data for each ticker
    for ticker in tickers_list:
        print(f"  - {ticker}...", end=" ")
        
        # Dividends and splits
        divs, spls, acts = fetch_dividends_and_splits(ticker)
        dividends[ticker] = divs
        splits[ticker] = spls
        actions[ticker] = acts
        
        # Company info
        info[ticker] = fetch_company_info(ticker)
        
        # Summary
        div_count = len(divs[divs > 0]) if len(divs) > 0 else 0
        split_count = len(spls[spls != 0]) if len(spls) > 0 else 0
        print(f"{div_count} dividends, {split_count} splits")

    # Create metadata
    meta = {
        'tickers': tickers_list,
        'start': start,
        'end': end,
        'interval': interval,
        'fetch_date': datetime.datetime.now().isoformat(),
        'num_tickers': len(tickers_list),
        'date_range_actual': {
            'start': str(prices.index.min().date()) if not prices.empty else None,
            'end': str(prices.index.max().date()) if not prices.empty else None,
            'num_periods': len(prices) if not prices.empty else 0,
        }
    }

    bundle = {
        'prices': prices,
        'dividends': dividends,
        'splits': splits,
        'actions': actions,
        'info': info,
        'meta': meta,
    }
    
    print(f"\n✓ Complete bundle fetched successfully!")
    return bundle


def save_bundle(bundle: Dict, base_name: str = "last_fetch") -> None:
    """
    Persist complete data bundle to DATA_DIR.
    
    Saves:
    - Prices as parquet (or CSV fallback)
    - Dividends, splits, actions as CSV per ticker
    - Info and metadata as JSON
    """
    global DATA_DIR
    
    print(f"\nSaving data bundle to {DATA_DIR}...")
    
    prices: pd.DataFrame = bundle['prices']
    meta = bundle.get('meta', {})
    
    # Save prices (prefer Parquet for efficiency)
    pq_path = DATA_DIR / f"{base_name}.parquet"
    csv_path = DATA_DIR / f"{base_name}.csv"
    try:
        prices.to_parquet(pq_path)
        print(f"  ✓ Prices saved: {pq_path.name}")
    except Exception as e:
        prices.to_csv(csv_path)
        print(f"  ✓ Prices saved: {csv_path.name} (Parquet failed: {e})")

    # Save corporate actions per ticker
    saved_files = 0
    for key in ('dividends', 'splits', 'actions'):
        obj = bundle.get(key, {}) or {}
        for ticker, data in obj.items():
            if data is None or (isinstance(data, (pd.Series, pd.DataFrame)) and len(data) == 0):
                continue
            
            out_path = DATA_DIR / f"{base_name}_{key}_{ticker}.csv"
            try:
                if isinstance(data, pd.Series):
                    data.to_frame().to_csv(out_path)
                else:
                    data.to_csv(out_path)
                saved_files += 1
            except Exception as e:
                print(f"  Warning: Could not save {key} for {ticker}: {e}")
    
    if saved_files > 0:
        print(f"  ✓ Corporate actions saved: {saved_files} file(s)")

    # Save company info as JSON
    info_path = DATA_DIR / f"{base_name}_info.json"
    try:
        with open(info_path, 'w') as f:
            json.dump(bundle.get('info', {}), f, indent=2, default=str)
        print(f"  ✓ Company info saved: {info_path.name}")
    except Exception as e:
        print(f"  Warning: Could not save info: {e}")

    # Save metadata as JSON
    meta_path = DATA_DIR / f"{base_name}_meta.json"
    try:
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  ✓ Metadata saved: {meta_path.name}")
    except Exception as e:
        print(f"  Warning: Could not save metadata: {e}")
    
    print(f"✓ Bundle saved successfully to {DATA_DIR}/")


def load_last_fetch_bundle(base_name: str = "last_fetch") -> Dict | None:
    """
    Load previously saved bundle from DATA_DIR.
    
    Returns
    -------
    dict or None
        Complete bundle if found, None otherwise
    """
    pq_path = DATA_DIR / f"{base_name}.parquet"
    csv_path = DATA_DIR / f"{base_name}.csv"
    
    # Load prices
    if pq_path.exists():
        prices = pd.read_parquet(pq_path)
    elif csv_path.exists():
        try:
            prices = pd.read_csv(csv_path, header=[0,1], index_col=0, parse_dates=True)
        except Exception:
            prices = pd.read_csv(csv_path, header=0, index_col=0, parse_dates=True)
    else:
        return None

    # Load metadata
    try:
        with open(DATA_DIR / f"{base_name}_meta.json", 'r') as f:
            meta = json.load(f)
    except Exception:
        meta = {}
    
    # Load info
    try:
        with open(DATA_DIR / f"{base_name}_info.json", 'r') as f:
            info = json.load(f)
    except Exception:
        info = {}

    # Load corporate actions per ticker
    tickers = meta.get('tickers', [])
    dividends, splits, actions = {}, {}, {}
    
    for ticker in tickers:
        try:
            div_path = DATA_DIR / f"{base_name}_dividends_{ticker}.csv"
            if div_path.exists():
                df = pd.read_csv(div_path, index_col=0, parse_dates=True)
                dividends[ticker] = df.iloc[:, 0] if len(df.columns) > 0 else pd.Series(dtype=float)
            
            split_path = DATA_DIR / f"{base_name}_splits_{ticker}.csv"
            if split_path.exists():
                df = pd.read_csv(split_path, index_col=0, parse_dates=True)
                splits[ticker] = df.iloc[:, 0] if len(df.columns) > 0 else pd.Series(dtype=float)
            
            action_path = DATA_DIR / f"{base_name}_actions_{ticker}.csv"
            if action_path.exists():
                actions[ticker] = pd.read_csv(action_path, index_col=0, parse_dates=True)
        except Exception:
            pass

    return {
        'prices': prices,
        'dividends': dividends,
        'splits': splits,
        'actions': actions,
        'info': info,
        'meta': meta,
    }


def clear_saved_data(base_name: str = "last_fetch", remove_all: bool = False) -> None:
    """
    Delete persisted datasets in DATA_DIR and reset module-level variables.

    Parameters
    ----------
    base_name : str
        Base name of the persisted bundle (default: "last_fetch").
    remove_all : bool
        If True, remove all files in DATA_DIR. Otherwise only matching base_name.
    """
    # Reset module-level state
    global LAST_FETCH_TICKERS, LAST_FETCH_START, LAST_FETCH_END, LAST_FETCH_INTERVAL
    global LAST_FETCH_PRICES, LAST_FETCH_DIVIDENDS, LAST_FETCH_SPLITS, LAST_FETCH_ACTIONS, LAST_FETCH_INFO
    
    LAST_FETCH_TICKERS = None
    LAST_FETCH_START = None
    LAST_FETCH_END = None
    LAST_FETCH_INTERVAL = None
    LAST_FETCH_PRICES = None
    LAST_FETCH_DIVIDENDS = None
    LAST_FETCH_SPLITS = None
    LAST_FETCH_ACTIONS = None
    LAST_FETCH_INFO = None

    try:
        if remove_all:
            removed = 0
            for p in DATA_DIR.glob("*"):
                try:
                    if p.is_file():
                        p.unlink()
                        removed += 1
                except Exception:
                    pass
            print(f"Cleared {removed} file(s) from {DATA_DIR}")
            return

        # Remove only files related to the specified base_name
        patterns = [
            f"{base_name}.parquet",
            f"{base_name}.csv",
            f"{base_name}_meta.json",
            f"{base_name}_info.json",
            f"{base_name}_dividends_*.csv",
            f"{base_name}_splits_*.csv",
            f"{base_name}_actions_*.csv",
        ]
        removed = 0
        for pat in patterns:
            for p in DATA_DIR.glob(pat):
                try:
                    p.unlink()
                    removed += 1
                except Exception:
                    pass
        
        msg = f"Cleared {removed} file(s) for '{base_name}'" if removed > 0 else f"No files found for '{base_name}'"
        print(msg)
    except Exception as e:
        print(f"Warning: Failed to clear saved data: {e}")


def set_last_fetch_globals(bundle: Dict) -> None:
    """Populate module-level variables with the provided bundle."""
    global LAST_FETCH_TICKERS, LAST_FETCH_START, LAST_FETCH_END, LAST_FETCH_INTERVAL
    global LAST_FETCH_PRICES, LAST_FETCH_DIVIDENDS, LAST_FETCH_SPLITS, LAST_FETCH_ACTIONS, LAST_FETCH_INFO
    
    meta = bundle.get('meta', {})
    LAST_FETCH_TICKERS = meta.get('tickers')
    LAST_FETCH_START = meta.get('start')
    LAST_FETCH_END = meta.get('end')
    LAST_FETCH_INTERVAL = meta.get('interval')
    LAST_FETCH_PRICES = bundle.get('prices')
    LAST_FETCH_DIVIDENDS = bundle.get('dividends')
    LAST_FETCH_SPLITS = bundle.get('splits')
    LAST_FETCH_ACTIONS = bundle.get('actions')
    LAST_FETCH_INFO = bundle.get('info')


def compute_log_returns(price_df: pd.DataFrame, field: str = "Adj Close") -> pd.DataFrame:
    """
    Compute daily log returns from price levels.

    Parameters
    ----------
    price_df : pd.DataFrame
        DataFrame of price data.
    field : str, default "Adj Close"
        The field/column name from which to derive returns.

    Returns
    -------
    pd.DataFrame
        DataFrame of log returns.
    """
    if isinstance(price_df.columns, pd.MultiIndex):
        price_series = price_df.xs(field, level=1, axis=1)
    else:
        price_series = price_df[[field]].rename(columns={field: 'price'})

    returns = price_series.apply(lambda x: pd.Series(x).pct_change()).apply(np.log1p)
    returns = returns.dropna()
    return returns


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch comprehensive stock data via yfinance.")
    parser.add_argument("--clear", action="store_true", 
                        help="Delete saved data for current base and exit.")
    parser.add_argument("--clear-all", action="store_true", 
                        help="Delete ALL files in data directory and exit.")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated tickers (e.g., AAPL,MSFT). Defaults to USER_TICKERS if omitted.")
    parser.add_argument("--start", type=str, default=None, 
                        help="Start date YYYY-MM-DD. Defaults to ~1 year ago.")
    parser.add_argument("--end", type=str, default=None, 
                        help="End date YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--interval", type=str, default="1d", 
                        help='Data interval: "1d", "1wk", "1mo".')
    args = parser.parse_args()

    # Handle clear options
    if args.clear or args.clear_all:
        clear_saved_data(base_name="last_fetch", remove_all=bool(args.clear_all))
        raise SystemExit(0)

    # Parse tickers
    if args.tickers:
        tickers_arg = [t.strip() for t in args.tickers.split(",")] if "," in args.tickers else args.tickers.strip()
    else:
        tickers_arg = list(USER_TICKERS)

    # Fetch comprehensive bundle
    bundle = fetch_full_bundle(tickers=tickers_arg, start=args.start, end=args.end, interval=args.interval)
    
    # Save to disk
    save_bundle(bundle, base_name="last_fetch")
    
    # Set module-level variables
    set_last_fetch_globals(bundle)

    # Print summary
    prices = bundle['prices']
    meta = bundle['meta']
    
    print(f"\n{'='*60}")
    print(f"DATA FETCH SUMMARY")
    print(f"{'='*60}")
    print(f"Tickers:        {', '.join(meta['tickers'])}")
    print(f"Date Range:     {meta['date_range_actual']['start']} to {meta['date_range_actual']['end']}")
    print(f"Periods:        {meta['date_range_actual']['num_periods']}")
    print(f"Interval:       {meta['interval']}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"{'='*60}\n")