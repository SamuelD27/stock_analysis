import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple
import numpy as np
from pathlib import Path
import tempfile
import os

def _clear_saved_data(base_name: str = "last_fetch", remove_all: bool = False) -> None:
    """Delete persisted datasets in repo data/ and print a status line."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    try:
        if remove_all:
            removed = 0
            for p in data_dir.glob("*"):
                try:
                    p.unlink()
                    removed += 1
                except IsADirectoryError:
                    pass
            print(f"Cleared {removed} files in {data_dir}")
            return
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
            for p in data_dir.glob(pat):
                try:
                    p.unlink()
                    removed += 1
                except Exception:
                    pass
        msg = "No saved data to clear" if removed == 0 else f"Cleared {removed} file(s) in {data_dir}"
        print(msg)
    except Exception as e:
        print(f"Warning: failed to clear saved data: {e}")

def _clear_figures(save_dir: str | None) -> None:
    """Delete figure files from the given directory."""
    if not save_dir:
        print("No save directory provided; nothing to clear.")
        return
    d = Path(save_dir)
    if not d.exists():
        print(f"Directory not found: {d}")
        return
    exts = ("*.png", "*.jpg", "*.jpeg", "*.pdf", "*.svg")
    removed = 0
    for ext in exts:
        for p in d.glob(ext):
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    msg = "No figures to clear" if removed == 0 else f"Cleared {removed} figure(s) in {d}"
    print(msg)

def save_or_show(fig: plt.Figure, outpath: str | None = None, dpi: int = 120, show: bool = True) -> None:
    """Save the figure to outpath if provided; otherwise show it interactively."""
    if outpath:
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    elif show:
        plt.show()

def plot_price_series(df: pd.DataFrame, ticker: Optional[str] = None, field: str = "Adj Close") -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(12, 6))
    if isinstance(df.columns, pd.MultiIndex):
        if ticker is None:
            raise ValueError("ticker must be specified when DataFrame has multiple tickers")
        series = df[ticker][field]
        title = f"{ticker} {field}"
    else:
        series = df[field]
        title = f"{field}"
    ax.plot(series.index, series.values, linewidth=2)
    ax.set_title(f"{title} Price Series", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax

def plot_return_distribution(returns: pd.Series, ticker: Optional[str] = None, bins: int = 50) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))
    clean_returns = returns.dropna().values
    ax.hist(clean_returns, bins=bins, density=True, alpha=0.7, edgecolor='black')
    
    # Add normal distribution overlay
    mu, sigma = clean_returns.mean(), clean_returns.std()
    x = np.linspace(clean_returns.min(), clean_returns.max(), 100)
    ax.plot(x, (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2), 
            'r-', linewidth=2, label='Normal Distribution')
    
    ttl = "Return Distribution" if ticker is None else f"{ticker} Return Distribution"
    ax.set_title(ttl, fontsize=14, fontweight='bold')
    ax.set_xlabel("Returns", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax

def plot_rolling_volatility(returns: pd.Series, window: int = 21, title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    vol = returns.rolling(window).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(vol.index, vol.values, linewidth=2, color='darkred')
    ax.fill_between(vol.index, vol.values, alpha=0.3, color='red')
    ax.set_title(title or f"Rolling {window}-Day Annualized Volatility", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Volatility", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax

def plot_cumulative_returns(returns: pd.Series, ticker: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot cumulative returns over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    cum_returns = (1 + returns).cumprod() - 1
    ax.plot(cum_returns.index, cum_returns.values * 100, linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ttl = "Cumulative Returns" if ticker is None else f"{ticker} Cumulative Returns"
    ax.set_title(ttl, fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax

def plot_drawdown(returns: pd.Series, ticker: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot drawdown (peak-to-trough decline)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max * 100
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, color='red')
    ttl = "Drawdown" if ticker is None else f"{ticker} Drawdown"
    ax.set_title(ttl, fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax

def plot_monthly_returns_heatmap(returns: pd.Series, ticker: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Create a heatmap of monthly returns."""
    monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
    
    # Create pivot table for heatmap
    monthly_returns_df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    pivot = monthly_returns_df.pivot(index='Month', columns='Year', values='Return')
    
    # Get the actual months present in the data
    months_present = sorted(pivot.index.tolist())
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    actual_month_labels = [month_labels[m-1] for m in months_present]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
    
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(months_present)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(actual_month_labels)
    
    # Add text annotations
    for i in range(len(months_present)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.1f}', ha="center", va="center", 
                             color="black" if abs(val) < 5 else "white", fontsize=9)
    
    ttl = "Monthly Returns Heatmap (%)" if ticker is None else f"{ticker} Monthly Returns Heatmap (%)"
    ax.set_title(ttl, fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Return (%)')
    fig.tight_layout()
    return fig, ax

def calculate_summary_statistics(returns: pd.Series, prices: pd.Series) -> pd.DataFrame:
    """Calculate comprehensive summary statistics."""
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    
    stats = {
        'Total Return (%)': [(cum_returns.iloc[-1] - 1) * 100],
        'Annualized Return (%)': [returns.mean() * 252 * 100],
        'Annualized Volatility (%)': [returns.std() * np.sqrt(252) * 100],
        'Sharpe Ratio': [(returns.mean() / returns.std()) * np.sqrt(252)],
        'Max Drawdown (%)': [drawdown.min() * 100],
        'Skewness': [returns.skew()],
        'Kurtosis': [returns.kurtosis()],
        'Best Day (%)': [returns.max() * 100],
        'Worst Day (%)': [returns.min() * 100],
        'Positive Days (%)': [(returns > 0).sum() / len(returns) * 100],
        'Current Price': [prices.iloc[-1]],
        'Period High': [prices.max()],
        'Period Low': [prices.min()],
    }
    
    return pd.DataFrame(stats).T.rename(columns={0: 'Value'})

def _choose_price_field(df: pd.DataFrame) -> str:
    """Pick a sensible price field from the DataFrame."""
    if isinstance(df.columns, pd.MultiIndex):
        fields = list(dict.fromkeys([c[1] for c in df.columns]))
        for cand in ("Adj Close", "Close"):
            if cand in fields:
                return cand
        return fields[0]
    else:
        cols = list(df.columns)
        for cand in ("Adj Close", "Close"):
            if cand in cols:
                return cand
        return cols[0]

if __name__ == "__main__":
    import argparse
    import sys

    # Pre-parse export flags
    raw_argv = sys.argv[1:]
    _pre_export = None
    _filtered = []
    i = 0
    while i < len(raw_argv):
        tok = raw_argv[i]
        if tok in ("--export-xlsx", "--export_xlsx", "--export"):
            if i + 1 < len(raw_argv) and not raw_argv[i + 1].startswith("-"):
                _pre_export = raw_argv[i + 1]
                i += 2
                continue
            else:
                _pre_export = ""
                i += 1
                continue
        _filtered.append(tok)
        i += 1
    sys.argv = [sys.argv[0]] + _filtered

    parser = argparse.ArgumentParser(description="Generate enhanced financial charts and Excel reports.")
    parser.add_argument("--clear", action="store_true", help="Delete saved data and exit.")
    parser.add_argument("--clear-all", action="store_true", help="Delete ALL data files and exit.")
    parser.add_argument("--clear-figs", action="store_true", help="Delete all figures and exit.")
    parser.add_argument("--data-path", type=str, default="data/last_fetch.parquet",
                        help="Path to saved dataset.")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated tickers to visualize.")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save figures.")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not show charts interactively.")
    parser.add_argument("--do", type=str, default="price,returns,vol,cumulative,drawdown,heatmap",
                        help="Comma list of charts: price,returns,vol,cumulative,drawdown,heatmap")
    parser.add_argument("--export-xlsx", "--export_xlsx", "--export", dest="export_xlsx", type=str, default=None,
                        help="Excel file path for comprehensive report")
    args = parser.parse_args()

    if (not hasattr(args, "export_xlsx")) or (args.export_xlsx is None):
        args.export_xlsx = _pre_export

    temp_dir_for_excel = None

    # Handle clear operations
    if args.clear or args.clear_all or args.clear_figs:
        if args.clear_figs:
            _clear_figures(args.save_dir)
        if args.clear or args.clear_all:
            _clear_saved_data(base_name="last_fetch", remove_all=bool(args.clear_all))
        sys.exit(0)

    data_path = Path(args.data_path)
    if not data_path.exists():
        csv_fallback = data_path.with_suffix(".csv")
        if csv_fallback.exists():
            data_path = csv_fallback
        else:
            print(f"Data file not found: {data_path}")
            sys.exit(1)

    meta_path = data_path.with_suffix("")
    if meta_path.suffix:
        meta_path = meta_path.with_suffix("")
    meta_path = meta_path.with_name(meta_path.name + "_meta.json")

    # Load data
    try:
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            try:
                df = pd.read_csv(data_path, header=[0,1], index_col=0, parse_dates=True)
            except Exception:
                df = pd.read_csv(data_path, header=0, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    # Resolve tickers
    if args.tickers:
        active_tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        active_tickers = None
        try:
            import json
            with open(meta_path, "r") as f:
                meta = json.load(f)
            mts = meta.get("tickers")
            if isinstance(mts, list) and len(mts) > 0:
                active_tickers = mts
        except Exception:
            pass

    if active_tickers is None:
        if isinstance(df.columns, pd.MultiIndex):
            active_tickers = list(dict.fromkeys([c[0] for c in df.columns]))
        else:
            active_tickers = ["__single__"]

    # Filter data
    if isinstance(df.columns, pd.MultiIndex):
        present = list(dict.fromkeys([c[0] for c in df.columns]))
        active = [t for t in active_tickers if t in present]
        if not active:
            active = present
        df = df[active]

    # Extract prices and compute returns
    chosen_field = _choose_price_field(df)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            px = df.xs(chosen_field, level=1, axis=1)
        except KeyError:
            fld = list(dict.fromkeys([c[1] for c in df.columns]))[0]
            px = df.xs(fld, level=1, axis=1)
    else:
        if chosen_field in df.columns:
            px = df[chosen_field]
        else:
            px = df.iloc[:, 0]

    px_filled = px.ffill()
    rets = px_filled.pct_change()
    rets = rets.dropna(how="all").ffill().bfill(limit=2)

    # Setup output
    outdir = None
    if args.save_dir:
        outdir = Path(args.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
    elif args.export_xlsx:
        temp_dir_for_excel = tempfile.TemporaryDirectory()
        outdir = Path(temp_dir_for_excel.name)

    generated_images: list[tuple[str, Path]] = []
    todo = [x.strip().lower() for x in args.do.split(",") if x.strip()]

    # Generate charts
    if "price" in todo:
        if isinstance(px, pd.DataFrame):
            for t in px.columns:
                fig, _ = plot_price_series(df, ticker=t, field=chosen_field)
                img_path = (outdir / f"{t}_price.png") if outdir else None
                save_or_show(fig, str(img_path) if img_path else None, show=not args.no_show)
                if img_path:
                    generated_images.append((f"{t} Price", img_path))
                plt.close(fig)
        else:
            fig, _ = plot_price_series(df, field=chosen_field)
            img_path = (outdir / "price.png") if outdir else None
            save_or_show(fig, str(img_path) if img_path else None, show=not args.no_show)
            if img_path:
                generated_images.append(("Price", img_path))
            plt.close(fig)

    if "returns" in todo:
        if isinstance(rets, pd.DataFrame):
            for t in rets.columns:
                fig, _ = plot_return_distribution(rets[t], ticker=t)
                img_path = (outdir / f"{t}_return_dist.png") if outdir else None
                save_or_show(fig, str(img_path) if img_path else None, show=not args.no_show)
                if img_path:
                    generated_images.append((f"{t} Return Dist", img_path))
                plt.close(fig)
        else:
            fig, _ = plot_return_distribution(rets)
            img_path = (outdir / "return_dist.png") if outdir else None
            save_or_show(fig, str(img_path) if img_path else None, show=not args.no_show)
            if img_path:
                generated_images.append(("Return Dist", img_path))
            plt.close(fig)

    if "vol" in todo:
        if isinstance(rets, pd.DataFrame):
            for t in rets.columns:
                fig, _ = plot_rolling_volatility(rets[t], window=21, title=f"{t} 21D Rolling Vol")
                img_path = (outdir / f"{t}_rolling_vol.png") if outdir else None
                save_or_show(fig, str(img_path) if img_path else None, show=not args.no_show)
                if img_path:
                    generated_images.append((f"{t} Rolling Vol", img_path))
                plt.close(fig)
        else:
            fig, _ = plot_rolling_volatility(rets, window=21)
            img_path = (outdir / "rolling_vol.png") if outdir else None
            save_or_show(fig, str(img_path) if img_path else None, show=not args.no_show)
            if img_path:
                generated_images.append(("Rolling Vol", img_path))
            plt.close(fig)

    if "cumulative" in todo:
        if isinstance(rets, pd.DataFrame):
            for t in rets.columns:
                fig, _ = plot_cumulative_returns(rets[t], ticker=t)
                img_path = (outdir / f"{t}_cumulative.png") if outdir else None
                save_or_show(fig, str(img_path) if img_path else None, show=not args.no_show)
                if img_path:
                    generated_images.append((f"{t} Cumulative Returns", img_path))
                plt.close(fig)
        else:
            fig, _ = plot_cumulative_returns(rets)
            img_path = (outdir / "cumulative.png") if outdir else None
            save_or_show(fig, str(img_path) if img_path else None, show=not args.no_show)
            if img_path:
                generated_images.append(("Cumulative Returns", img_path))
            plt.close(fig)

    if "drawdown" in todo:
        if isinstance(rets, pd.DataFrame):
            for t in rets.columns:
                fig, _ = plot_drawdown(rets[t], ticker=t)
                img_path = (outdir / f"{t}_drawdown.png") if outdir else None
                save_or_show(fig, str(img_path) if img_path else None, show=not args.no_show)
                if img_path:
                    generated_images.append((f"{t} Drawdown", img_path))
                plt.close(fig)
        else:
            fig, _ = plot_drawdown(rets)
            img_path = (outdir / "drawdown.png") if outdir else None
            save_or_show(fig, str(img_path) if img_path else None, show=not args.no_show)
            if img_path:
                generated_images.append(("Drawdown", img_path))
            plt.close(fig)

    if "heatmap" in todo:
        if isinstance(rets, pd.DataFrame):
            for t in rets.columns:
                fig, _ = plot_monthly_returns_heatmap(rets[t], ticker=t)
                img_path = (outdir / f"{t}_heatmap.png") if outdir else None
                save_or_show(fig, str(img_path) if img_path else None, show=not args.no_show)
                if img_path:
                    generated_images.append((f"{t} Monthly Heatmap", img_path))
                plt.close(fig)
        else:
            fig, _ = plot_monthly_returns_heatmap(rets)
            img_path = (outdir / "heatmap.png") if outdir else None
            save_or_show(fig, str(img_path) if img_path else None, show=not args.no_show)
            if img_path:
                generated_images.append(("Monthly Heatmap", img_path))
            plt.close(fig)

    # Export to Excel with enhanced formatting
    if args.export_xlsx:
        export_path = Path(args.export_xlsx)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        px_df = px if isinstance(px, pd.DataFrame) else px.to_frame(name=chosen_field)
        px_df.index = pd.to_datetime(px_df.index)
        rets.index = pd.to_datetime(rets.index)
        vol21 = rets.rolling(21, min_periods=5).std(ddof=0) * np.sqrt(252)
        vol21 = vol21.ffill(limit=2)
        vol21.index = pd.to_datetime(vol21.index)
        
        with pd.ExcelWriter(export_path, engine="xlsxwriter") as writer:
            workbook = writer.book
            
            # Define formats
            date_fmt = workbook.add_format({"num_format": "yyyy-mm-dd"})
            price_fmt = workbook.add_format({"num_format": "#,##0.00"})
            pct_fmt = workbook.add_format({"num_format": "0.00%"})
            header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})
            title_fmt = workbook.add_format({"bold": True, "font_size": 14})
            
            # Write Prices sheet
            px_df.to_excel(writer, sheet_name="Prices", index=True, index_label="Date")
            ws_prices = writer.sheets["Prices"]
            ws_prices.set_column(0, 0, 18, date_fmt)
            ws_prices.set_column(1, px_df.shape[1], 15, price_fmt)
            
            # Write Returns sheet
            rets.to_excel(writer, sheet_name="Returns", index=True, index_label="Date")
            ws_returns = writer.sheets["Returns"]
            ws_returns.set_column(0, 0, 18, date_fmt)
            ws_returns.set_column(1, rets.shape[1], 12, pct_fmt)
            
            # Write Volatility sheet
            vol21.to_excel(writer, sheet_name="RollingVol_21D", index=True, index_label="Date")
            ws_vol = writer.sheets["RollingVol_21D"]
            ws_vol.set_column(0, 0, 18, date_fmt)
            ws_vol.set_column(1, vol21.shape[1], 12, pct_fmt)
            
            # Add Summary Statistics sheet
            summary_ws = workbook.add_worksheet("Summary Statistics")
            summary_ws.write(0, 0, "Performance Summary", title_fmt)
            
            row_offset = 2
            if isinstance(rets, pd.DataFrame):
                for col_idx, t in enumerate(rets.columns):
                    stats_df = calculate_summary_statistics(rets[t], px_df[t] if isinstance(px_df, pd.DataFrame) else px_df)
                    
                    # Write ticker name
                    summary_ws.write(row_offset, col_idx * 3, t, title_fmt)
                    
                    # Write statistics
                    for i, (idx, row) in enumerate(stats_df.iterrows()):
                        summary_ws.write(row_offset + i + 1, col_idx * 3, idx)
                        summary_ws.write(row_offset + i + 1, col_idx * 3 + 1, row['Value'])
                    
            else:
                stats_df = calculate_summary_statistics(rets, px_df if isinstance(px_df, pd.Series) else px_df.iloc[:, 0])
                summary_ws.write(row_offset, 0, "Metric", header_fmt)
                summary_ws.write(row_offset, 1, "Value", header_fmt)
                for i, (idx, row) in enumerate(stats_df.iterrows()):
                    summary_ws.write(row_offset + i + 1, 0, idx)
                    summary_ws.write(row_offset + i + 1, 1, row['Value'])
            
            summary_ws.set_column(0, 0, 30)
            summary_ws.set_column(1, 10, 15)
            
            # Add Charts sheet
            charts_ws = workbook.add_worksheet("Charts")
            row = 1
            col = 1
            for label, path in generated_images:
                try:
                    charts_ws.write(row, col, label, title_fmt)
                    charts_ws.insert_image(row + 1, col, str(path), {"x_scale": 0.85, "y_scale": 0.85})
                    row += 32
                except Exception as e:
                    print(f"Warning: Could not insert image {label}: {e}")
                    row += 2
        
        print(f"Excel report written to: {export_path}")

    print(f"Visualization completed. Tickers: {active_tickers}")

    if temp_dir_for_excel is not None:
        temp_dir_for_excel.cleanup()