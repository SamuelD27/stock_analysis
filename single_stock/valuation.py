"""
Comprehensive Equity Valuation Module
-------------------------------------
Advanced valuation analysis using market data, fundamentals, and technical indicators.
Includes DCF, relative valuation, and market-based metrics.

Fully integrated with data fetching and processing pipeline.
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
from scipy import stats

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
    print(f"Error: Could not import utilities: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Fundamentals:
    """Container for fundamental metrics."""
    # Market metrics
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    shares_outstanding: Optional[float] = None
    
    # Balance sheet
    total_debt: Optional[float] = None
    cash: Optional[float] = None
    book_value: Optional[float] = None
    total_assets: Optional[float] = None
    
    # Income statement (TTM)
    revenue: Optional[float] = None
    ebitda: Optional[float] = None
    ebit: Optional[float] = None
    net_income: Optional[float] = None
    eps: Optional[float] = None
    
    # Cash flow (TTM)
    operating_cf: Optional[float] = None
    free_cf: Optional[float] = None
    
    # Ratios & margins
    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    profit_margin: Optional[float] = None
    
    # Valuation multiples (from Yahoo)
    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    
    # Dividends
    dividend_rate: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    
    # Growth
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    
    # Other
    beta: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None


@dataclass
class ValuationResult:
    """Complete valuation analysis results."""
    ticker: str
    current_price: float
    fundamentals: Fundamentals
    
    # Calculated ratios
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    pcf_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    ev_revenue: Optional[float] = None
    
    # Quality scores
    profitability_score: Optional[float] = None
    efficiency_score: Optional[float] = None
    leverage_score: Optional[float] = None
    
    # Market position
    price_to_52w_high: Optional[float] = None
    price_to_52w_low: Optional[float] = None
    
    # DCF estimate
    dcf_value: Optional[float] = None
    dcf_upside: Optional[float] = None


# ---------------------------------------------------------------------------
# Data loading and extraction
# ---------------------------------------------------------------------------

def extract_fundamentals(info: Dict, ticker: str) -> Fundamentals:
    """Extract fundamental data from Yahoo Finance info dict."""
    def safe_get(key, default=None):
        val = info.get(key, default)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return val
    
    return Fundamentals(
        # Market metrics
        market_cap=safe_get('marketCap'),
        enterprise_value=safe_get('enterpriseValue'),
        shares_outstanding=safe_get('sharesOutstanding'),
        
        # Balance sheet
        total_debt=safe_get('totalDebt'),
        cash=safe_get('totalCash'),
        book_value=safe_get('bookValue'),
        total_assets=safe_get('totalAssets'),
        
        # Income statement
        revenue=safe_get('totalRevenue'),
        ebitda=safe_get('ebitda'),
        ebit=safe_get('ebit'),
        net_income=safe_get('netIncome'),
        eps=safe_get('trailingEps'),
        
        # Cash flow
        operating_cf=safe_get('operatingCashflow'),
        free_cf=safe_get('freeCashflow'),
        
        # Ratios
        roe=safe_get('returnOnEquity'),
        roa=safe_get('returnOnAssets'),
        roic=safe_get('returnOnCapital'),
        gross_margin=safe_get('grossMargins'),
        operating_margin=safe_get('operatingMargins'),
        profit_margin=safe_get('profitMargins'),
        
        # Valuation multiples
        trailing_pe=safe_get('trailingPE'),
        forward_pe=safe_get('forwardPE'),
        peg_ratio=safe_get('pegRatio'),
        price_to_book=safe_get('priceToBook'),
        
        # Dividends
        dividend_rate=safe_get('dividendRate'),
        dividend_yield=safe_get('dividendYield'),
        payout_ratio=safe_get('payoutRatio'),
        
        # Growth
        revenue_growth=safe_get('revenueGrowth'),
        earnings_growth=safe_get('earningsGrowth'),
        
        # Other
        beta=safe_get('beta'),
        fifty_two_week_high=safe_get('fiftyTwoWeekHigh'),
        fifty_two_week_low=safe_get('fiftyTwoWeekLow'),
    )


def load_stock_fundamentals(base_name: str = "last_fetch") -> Dict[str, Tuple[float, Fundamentals]]:
    """
    Load fundamentals for all tickers in current session.
    
    Returns
    -------
    dict
        ticker -> (current_price, Fundamentals)
    """
    prices_df, bundle = load_current_session_data(base_name=base_name)
    tickers = bundle['meta'].get('tickers', [])
    is_single = bundle.get('is_single_ticker', False)
    info = bundle.get('info', {})
    
    results = {}
    
    for ticker in tickers:
        # Get current price
        if is_single or not isinstance(prices_df.columns, pd.MultiIndex):
            price = prices_df['Adj Close'].iloc[-1] if 'Adj Close' in prices_df.columns else prices_df['Close'].iloc[-1]
        else:
            price = prices_df[ticker]['Adj Close'].iloc[-1]
        
        # Get fundamentals
        ticker_info = info.get(ticker, {})
        fundamentals = extract_fundamentals(ticker_info, ticker)
        
        results[ticker] = (float(price), fundamentals)
    
    return results


# ---------------------------------------------------------------------------
# Valuation calculations
# ---------------------------------------------------------------------------

def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Safe division handling None and zero."""
    if a is None or b is None or b == 0:
        return None
    return float(a) / float(b)


def calculate_valuation_ratios(price: float, fund: Fundamentals) -> Dict[str, Optional[float]]:
    """Calculate all valuation ratios."""
    ratios = {}
    
    # Price ratios
    ratios['PE'] = safe_div(price, fund.eps)
    ratios['PB'] = safe_div(price * fund.shares_outstanding, fund.book_value)
    ratios['PS'] = safe_div(price * fund.shares_outstanding, fund.revenue)
    ratios['PCF'] = safe_div(price * fund.shares_outstanding, fund.free_cf)
    
    # Enterprise value ratios
    ratios['EV/EBITDA'] = safe_div(fund.enterprise_value, fund.ebitda)
    ratios['EV/Revenue'] = safe_div(fund.enterprise_value, fund.revenue)
    ratios['EV/FCF'] = safe_div(fund.enterprise_value, fund.free_cf)
    
    # Per share metrics
    ratios['Book_Value_Per_Share'] = safe_div(fund.book_value, fund.shares_outstanding)
    ratios['FCF_Per_Share'] = safe_div(fund.free_cf, fund.shares_outstanding)
    ratios['Revenue_Per_Share'] = safe_div(fund.revenue, fund.shares_outstanding)
    
    # Yield metrics
    ratios['Earnings_Yield'] = safe_div(fund.eps, price) if fund.eps and price else None
    ratios['FCF_Yield'] = safe_div(fund.free_cf, fund.market_cap) if fund.free_cf and fund.market_cap else None
    
    return ratios


def calculate_quality_scores(fund: Fundamentals) -> Dict[str, Optional[float]]:
    """Calculate quality scores (0-100 scale)."""
    scores = {}
    
    # Profitability score (ROE, ROA, margins)
    prof_components = [
        fund.roe if fund.roe and fund.roe > 0 else 0,
        fund.roa if fund.roa and fund.roa > 0 else 0,
        fund.profit_margin if fund.profit_margin and fund.profit_margin > 0 else 0,
    ]
    scores['Profitability'] = np.mean([min(x * 100, 100) for x in prof_components if x is not None])
    
    # Efficiency score (margins)
    eff_components = [
        fund.gross_margin if fund.gross_margin else 0,
        fund.operating_margin if fund.operating_margin else 0,
        fund.profit_margin if fund.profit_margin else 0,
    ]
    scores['Efficiency'] = np.mean([x * 100 for x in eff_components if x is not None])
    
    # Leverage score (lower debt is better)
    if fund.total_debt and fund.market_cap:
        debt_to_equity = fund.total_debt / fund.market_cap
        scores['Leverage'] = max(0, 100 - debt_to_equity * 50)
    else:
        scores['Leverage'] = None
    
    return scores


def simple_dcf_valuation(fund: Fundamentals, wacc: float = 0.10, 
                         growth_rate: float = 0.03, terminal_growth: float = 0.02,
                         projection_years: int = 5) -> Optional[float]:
    """
    Simplified DCF valuation using free cash flow.
    
    Parameters
    ----------
    fund : Fundamentals
        Company fundamentals
    wacc : float
        Weighted average cost of capital
    growth_rate : float
        Expected FCF growth rate for projection period
    terminal_growth : float
        Terminal growth rate
    projection_years : int
        Number of years to project
    
    Returns
    -------
    float or None
        Estimated intrinsic value per share
    """
    if not fund.free_cf or not fund.shares_outstanding:
        return None
    
    if fund.free_cf <= 0:
        return None
    
    # Project free cash flows
    fcf = fund.free_cf
    pv_fcf = 0
    
    for year in range(1, projection_years + 1):
        fcf_projected = fcf * ((1 + growth_rate) ** year)
        discount_factor = (1 + wacc) ** year
        pv_fcf += fcf_projected / discount_factor
    
    # Terminal value
    fcf_terminal = fcf * ((1 + growth_rate) ** projection_years) * (1 + terminal_growth)
    terminal_value = fcf_terminal / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** projection_years)
    
    # Enterprise value
    enterprise_value = pv_fcf + pv_terminal
    
    # Equity value
    net_debt = (fund.total_debt or 0) - (fund.cash or 0)
    equity_value = enterprise_value - net_debt
    
    # Per share value
    value_per_share = equity_value / fund.shares_outstanding
    
    return float(value_per_share)


def calculate_graham_number(eps: Optional[float], bvps: Optional[float]) -> Optional[float]:
    """Calculate Benjamin Graham's intrinsic value formula."""
    if not eps or not bvps or eps <= 0 or bvps <= 0:
        return None
    return float(np.sqrt(22.5 * eps * bvps))


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def perform_valuation_analysis(ticker: str, price: float, fund: Fundamentals) -> ValuationResult:
    """Comprehensive valuation analysis for a single stock."""
    
    # Calculate ratios
    ratios = calculate_valuation_ratios(price, fund)
    
    # Quality scores
    quality = calculate_quality_scores(fund)
    
    # DCF valuation
    dcf_value = simple_dcf_valuation(fund)
    dcf_upside = safe_div(dcf_value - price, price) if dcf_value else None
    
    # Market position
    price_to_high = safe_div(price, fund.fifty_two_week_high) if fund.fifty_two_week_high else None
    price_to_low = safe_div(price, fund.fifty_two_week_low) if fund.fifty_two_week_low else None
    
    return ValuationResult(
        ticker=ticker,
        current_price=price,
        fundamentals=fund,
        pe_ratio=ratios.get('PE'),
        pb_ratio=ratios.get('PB'),
        ps_ratio=ratios.get('PS'),
        pcf_ratio=ratios.get('PCF'),
        ev_ebitda=ratios.get('EV/EBITDA'),
        ev_revenue=ratios.get('EV/Revenue'),
        profitability_score=quality.get('Profitability'),
        efficiency_score=quality.get('Efficiency'),
        leverage_score=quality.get('Leverage'),
        price_to_52w_high=price_to_high,
        price_to_52w_low=price_to_low,
        dcf_value=dcf_value,
        dcf_upside=dcf_upside,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_valuation_dashboard(result: ValuationResult) -> Tuple[plt.Figure, np.ndarray]:
    """Create comprehensive valuation dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Valuation Dashboard: {result.ticker}', fontsize=16, fontweight='bold')
    
    # 1. Valuation multiples
    ax = axes[0, 0]
    multiples = {
        'P/E': result.pe_ratio,
        'P/B': result.pb_ratio,
        'P/S': result.ps_ratio,
        'EV/EBITDA': result.ev_ebitda
    }
    multiples = {k: v for k, v in multiples.items() if v is not None and not np.isnan(v)}
    
    if multiples:
        ax.bar(multiples.keys(), multiples.values(), color='steelblue', alpha=0.7)
        ax.set_title('Valuation Multiples')
        ax.set_ylabel('Ratio')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # 2. Quality scores
    ax = axes[0, 1]
    scores = {
        'Profitability': result.profitability_score,
        'Efficiency': result.efficiency_score,
        'Leverage': result.leverage_score
    }
    scores = {k: v for k, v in scores.items() if v is not None}
    
    if scores:
        colors = ['green' if v > 60 else 'orange' if v > 40 else 'red' for v in scores.values()]
        ax.barh(list(scores.keys()), list(scores.values()), color=colors, alpha=0.7)
        ax.set_xlim(0, 100)
        ax.set_title('Quality Scores (0-100)')
        ax.set_xlabel('Score')
        ax.grid(True, alpha=0.3)
    
    # 3. Profitability metrics
    ax = axes[0, 2]
    fund = result.fundamentals
    prof_metrics = {
        'ROE': fund.roe * 100 if fund.roe else None,
        'ROA': fund.roa * 100 if fund.roa else None,
        'ROIC': fund.roic * 100 if fund.roic else None
    }
    prof_metrics = {k: v for k, v in prof_metrics.items() if v is not None}
    
    if prof_metrics:
        ax.bar(prof_metrics.keys(), prof_metrics.values(), color='darkgreen', alpha=0.7)
        ax.set_title('Return Metrics (%)')
        ax.set_ylabel('Return (%)')
        ax.axhline(15, color='red', linestyle='--', linewidth=1, alpha=0.5, label='15% threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Margins
    ax = axes[1, 0]
    margins = {
        'Gross': fund.gross_margin * 100 if fund.gross_margin else None,
        'Operating': fund.operating_margin * 100 if fund.operating_margin else None,
        'Net': fund.profit_margin * 100 if fund.profit_margin else None
    }
    margins = {k: v for k, v in margins.items() if v is not None}
    
    if margins:
        ax.bar(margins.keys(), margins.values(), color='purple', alpha=0.7)
        ax.set_title('Profit Margins (%)')
        ax.set_ylabel('Margin (%)')
        ax.grid(True, alpha=0.3)
    
    # 5. Price vs 52-week range
    ax = axes[1, 1]
    if fund.fifty_two_week_low and fund.fifty_two_week_high:
        positions = [fund.fifty_two_week_low, result.current_price, fund.fifty_two_week_high]
        labels = ['52W Low', 'Current', '52W High']
        colors = ['red', 'blue', 'green']
        ax.barh(labels, positions, color=colors, alpha=0.7)
        ax.set_title('Price Position')
        ax.set_xlabel('Price')
        ax.grid(True, alpha=0.3)
    
    # 6. DCF valuation
    ax = axes[1, 2]
    if result.dcf_value:
        values = [result.current_price, result.dcf_value]
        labels = ['Market Price', 'DCF Value']
        colors = ['blue', 'green' if result.dcf_value > result.current_price else 'red']
        ax.bar(labels, values, color=colors, alpha=0.7)
        ax.set_title('DCF Valuation')
        ax.set_ylabel('Price per Share')
        
        if result.dcf_upside:
            upside_text = f"Upside: {result.dcf_upside*100:.1f}%"
            ax.text(0.5, 0.95, upside_text, transform=ax.transAxes,
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, axes


def plot_peer_comparison(results_df: pd.DataFrame, metric: str) -> Tuple[plt.Figure, plt.Axes]:
    """Plot comparison of specific metric across tickers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = results_df[metric].dropna()
    if len(data) == 0:
        ax.text(0.5, 0.5, f'No data available for {metric}', 
               ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    colors = ['green' if x < data.median() else 'red' for x in data.values]
    ax.bar(data.index, data.values, color=colors, alpha=0.7)
    ax.axhline(data.median(), color='blue', linestyle='--', linewidth=2, 
              label=f'Median: {data.median():.2f}')
    
    ax.set_title(f'Peer Comparison: {metric}', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def export_to_excel(output_path: Path, 
                    summary_df: pd.DataFrame,
                    fundamentals_df: pd.DataFrame,
                    ratios_df: pd.DataFrame,
                    images: List[Tuple[str, Path]]) -> None:
    """Export comprehensive valuation results to Excel."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Formats
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        num_fmt = workbook.add_format({'num_format': '#,##0.00'})
        
        # Summary
        summary_df.to_excel(writer, sheet_name='Valuation Summary')
        
        # Fundamentals
        fundamentals_df.to_excel(writer, sheet_name='Fundamentals')
        
        # Ratios
        ratios_df.to_excel(writer, sheet_name='Valuation Ratios')
        
        # Charts
        charts_ws = workbook.add_worksheet('Charts')
        row = 1
        for label, img_path in images:
            try:
                charts_ws.write(row, 1, label, header_fmt)
                charts_ws.insert_image(row + 1, 1, str(img_path),
                                      {'x_scale': 0.75, 'y_scale': 0.75})
                row += 35
            except Exception:
                row += 2
    
    print(f"✓ Excel report saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Equity Valuation")
    parser.add_argument("--base", type=str, default="last_fetch")
    parser.add_argument("--save-dir", type=str, default="results/valuation")
    parser.add_argument("--export-xlsx", type=str, default="reports/valuation_analysis.xlsx")
    parser.add_argument("--wacc", type=float, default=0.10,
                       help="WACC for DCF (default: 10%%)")
    parser.add_argument("--growth", type=float, default=0.05,
                       help="Growth rate for DCF (default: 5%%)")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE EQUITY VALUATION")
    print(f"{'='*60}\n")
    
    # Load data
    try:
        stock_data = load_stock_fundamentals(args.base)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    if not stock_data:
        print("✗ No stock data found")
        return
    
    # Perform valuation for each ticker
    results = {}
    summary_rows = []
    fundamentals_rows = []
    ratios_rows = []
    
    for ticker, (price, fund) in stock_data.items():
        print(f"Analyzing {ticker}...")
        result = perform_valuation_analysis(ticker, price, fund)
        results[ticker] = result
        
        # Summary row
        summary_rows.append({
            'Ticker': ticker,
            'Price': result.current_price,
            'P/E': result.pe_ratio,
            'P/B': result.pb_ratio,
            'EV/EBITDA': result.ev_ebitda,
            'DCF Value': result.dcf_value,
            'DCF Upside (%)': result.dcf_upside * 100 if result.dcf_upside else None,
            'Profitability Score': result.profitability_score,
            'Beta': fund.beta
        })
        
        # Fundamentals row
        fundamentals_rows.append({
            'Ticker': ticker,
            'Market Cap': fund.market_cap,
            'Revenue': fund.revenue,
            'EBITDA': fund.ebitda,
            'Net Income': fund.net_income,
            'Free Cash Flow': fund.free_cf,
            'ROE': fund.roe,
            'ROA': fund.roa,
            'Debt': fund.total_debt,
            'Cash': fund.cash
        })
        
        # Ratios row
        ratios = calculate_valuation_ratios(price, fund)
        ratios_rows.append({'Ticker': ticker, **ratios})
    
    summary_df = pd.DataFrame(summary_rows).set_index('Ticker')
    fundamentals_df = pd.DataFrame(fundamentals_rows).set_index('Ticker')
    ratios_df = pd.DataFrame(ratios_rows).set_index('Ticker')
    
    # Generate visualizations
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    images = []
    
    for ticker, result in results.items():
        # Dashboard
        fig, _ = plot_valuation_dashboard(result)
        img_path = outdir / f"{ticker}_valuation_dashboard.png"
        fig.savefig(img_path, dpi=120, bbox_inches='tight')
        images.append((f"{ticker} Dashboard", img_path))
        plt.close(fig)
    
    # Peer comparisons (if multiple tickers)
    if len(results) > 1:
        for metric in ['P/E', 'P/B', 'EV/EBITDA']:
            if metric in summary_df.columns:
                fig, _ = plot_peer_comparison(summary_df, metric)
                img_path = outdir / f"peer_comparison_{metric.replace('/', '_')}.png"
                fig.savefig(img_path, dpi=120, bbox_inches='tight')
                images.append((f"Peer {metric}", img_path))
                plt.close(fig)
    
    # Export to Excel
    export_path = Path(args.export_xlsx)
    export_to_excel(export_path, summary_df, fundamentals_df, ratios_df, images)
    
    # Console summary
    print(f"\n{'='*60}")
    print("VALUATION SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string())
    print(f"\n{'='*60}")
    print(f"✓ Analysis complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()