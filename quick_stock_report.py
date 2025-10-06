"""
Comprehensive Quick Stock Report
---------------------------------
Runs all single-stock analyses and produces a hedge-fund-ready Excel report.

Output:
- Summary page: Key metrics, valuation, risk assessment, recommendation
- Charts appendix: Price history, returns distribution, volatility, forecasts
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import warnings
import tempfile
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    # Core modules
    from data_fetching import fetch_full_bundle, save_bundle, set_last_fetch_globals
    from utils.data_processing import (
        StockDataProcessor,
        load_current_session_data,
        get_current_session_tickers
    )
    
    # Single stock modules
    from single_stock.forecasting import fit_arima_model, fit_exponential_smoothing
    from single_stock.risk_metrics import compute_var, compute_sharpe_ratio, compute_beta
    
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure all modules are in place and you're running from project root.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Analysis orchestrator
# ---------------------------------------------------------------------------

def run_complete_analysis(ticker: str, start_date: str = None, 
                         benchmark: str = "^GSPC") -> dict:
    """
    Run complete single-stock analysis pipeline.
    
    Returns comprehensive results dictionary.
    """
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE ANALYSIS: {ticker}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Step 1: Fetch data
    print("1/5 Fetching market data...")
    bundle = fetch_full_bundle(tickers=ticker, start=start_date)
    save_bundle(bundle, base_name="last_fetch")
    set_last_fetch_globals(bundle)
    
    # Step 2: Load and process
    print("2/5 Processing data...")
    prices_df, bundle = load_current_session_data(base_name="last_fetch")
    
    # Get benchmark
    benchmark_returns = None
    try:
        import yfinance as yf
        start = bundle['meta']['date_range_actual']['start']
        end = bundle['meta']['date_range_actual']['end']
        bench_data = yf.download(benchmark, start=start, end=end, progress=False)
        if not bench_data.empty:
            benchmark_returns = bench_data['Adj Close'].pct_change().dropna()
            print(f"   Loaded benchmark: {benchmark}")
    except Exception:
        print(f"   Warning: Could not load benchmark {benchmark}")
    
    # Step 3: Run processing
    print("3/5 Computing statistics and risk metrics...")
    processor = StockDataProcessor(ticker, prices_df, bundle, benchmark_returns)
    results = processor.process_all()
    
    # Step 4: Forecasting
    print("4/5 Generating forecasts...")
    returns = results['returns']['simple_returns']
    prices = processor.close_prices
    
    # ARIMA forecast
    try:
        train_size = int(len(prices) * 0.9)
        train = prices.iloc[:train_size].values
        test = prices.iloc[train_size:].values
        
        arima_result = fit_arima_model(train, test[:min(30, len(test))], order=(1,1,1))
        forecast_data = {
            'model': 'ARIMA',
            'predictions': arima_result.predictions,
            'horizon': arima_result.forecast_horizon,
            'rmse': arima_result.rmse,
            'directional_accuracy': arima_result.directional_accuracy
        }
    except Exception as e:
        print(f"   Forecast failed: {e}")
        forecast_data = None
    
    # Step 5: Generate visualizations
    print("5/5 Creating visualizations...")
    charts = generate_charts(processor, results, forecast_data, ticker)
    
    elapsed = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed:.1f}s")
    
    return {
        'ticker': ticker,
        'processor': processor,
        'results': results,
        'forecast': forecast_data,
        'charts': charts,
        'benchmark': benchmark,
        'analysis_time': elapsed
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_charts(processor, results, forecast_data, ticker: str) -> dict:
    """Generate all charts for the report."""
    charts = {}
    temp_dir = tempfile.mkdtemp()
    
    prices = processor.close_prices
    returns = results['returns']['simple_returns']
    
    # Chart 1: Price History with Moving Averages
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(prices.index, prices.values, linewidth=2, label='Price', color='#2E86AB')
    
    # Moving averages
    sma20 = prices.rolling(20).mean()
    sma50 = prices.rolling(50).mean()
    ax.plot(sma20.index, sma20.values, '--', alpha=0.7, label='SMA 20', color='orange')
    ax.plot(sma50.index, sma50.values, '--', alpha=0.7, label='SMA 50', color='red')
    
    ax.set_title(f'{ticker} Price History', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price ($)', fontsize=11)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    path = Path(temp_dir) / 'price_history.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    charts['price_history'] = path
    plt.close(fig)
    
    # Chart 2: Returns Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    clean_returns = returns.dropna() * 100
    
    ax.hist(clean_returns, bins=50, density=True, alpha=0.7, color='#A23B72', edgecolor='black')
    
    # Normal overlay
    mu, sigma = clean_returns.mean(), clean_returns.std()
    x = np.linspace(clean_returns.min(), clean_returns.max(), 100)
    ax.plot(x, (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2),
            'r-', linewidth=2, label='Normal Distribution')
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_title(f'{ticker} Returns Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Daily Return (%)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add stats box
    stats_text = f'Mean: {mu:.3f}%\nStd: {sigma:.3f}%\nSkew: {clean_returns.skew():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.tight_layout()
    path = Path(temp_dir) / 'returns_dist.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    charts['returns_dist'] = path
    plt.close(fig)
    
    # Chart 3: Rolling Volatility
    fig, ax = plt.subplots(figsize=(12, 5))
    vol = returns.rolling(21).std() * np.sqrt(252) * 100
    
    ax.plot(vol.index, vol.values, linewidth=2, color='#F18F01')
    ax.fill_between(vol.index, vol.values, alpha=0.3, color='#F18F01')
    ax.set_title(f'{ticker} Rolling Volatility (21-day)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Annualized Volatility (%)', fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    path = Path(temp_dir) / 'volatility.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    charts['volatility'] = path
    plt.close(fig)
    
    # Chart 4: Forecast (if available)
    if forecast_data:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Historical
        last_n = 60
        hist_prices = prices.iloc[-last_n:]
        ax.plot(range(len(hist_prices)), hist_prices.values, 
                'k-', linewidth=2, label='Historical', alpha=0.8)
        
        # Forecast
        forecast_start = len(hist_prices)
        horizon = range(forecast_start, forecast_start + len(forecast_data['predictions']))
        ax.plot(horizon, forecast_data['predictions'], 
                'b--', linewidth=2, label='Forecast', alpha=0.8)
        
        ax.axvline(forecast_start - 1, color='red', linestyle='--', 
                  linewidth=1, alpha=0.5, label='Forecast Start')
        
        ax.set_title(f'{ticker} Price Forecast (ARIMA)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Days', fontsize=11)
        ax.set_ylabel('Price ($)', fontsize=11)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add accuracy box
        acc_text = f'RMSE: ${forecast_data["rmse"]:.2f}\nDir Acc: {forecast_data["directional_accuracy"]:.1f}%'
        ax.text(0.02, 0.98, acc_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        fig.tight_layout()
        path = Path(temp_dir) / 'forecast.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        charts['forecast'] = path
        plt.close(fig)
    
    return charts


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_recommendation(results: dict, processor) -> str:
    """Generate investment recommendation based on analysis."""
    desc = results['descriptive_stats']
    risk = results['risk_metrics']
    
    # Scoring system
    score = 0
    reasons = []
    
    # Positive factors
    if desc['sharpe_ratio'] > 1.0:
        score += 2
        reasons.append("Strong risk-adjusted returns")
    elif desc['sharpe_ratio'] > 0.5:
        score += 1
        reasons.append("Moderate risk-adjusted returns")
    
    if desc['mean_annual_return'] > 0.10:
        score += 2
        reasons.append("High annual returns (>10%)")
    elif desc['mean_annual_return'] > 0:
        score += 1
        reasons.append("Positive annual returns")
    
    if abs(risk['max_drawdown']) < 0.20:
        score += 1
        reasons.append("Limited drawdown risk")
    
    if results['market_sensitivity'].get('alpha_annual', 0) > 0:
        score += 1
        reasons.append("Positive alpha vs market")
    
    # Negative factors
    if desc['std_annual_return'] > 0.40:
        score -= 1
        reasons.append("High volatility (>40%)")
    
    if abs(risk['max_drawdown']) > 0.50:
        score -= 2
        reasons.append("Severe drawdown risk")
    
    # Recommendation
    if score >= 5:
        return "STRONG BUY", reasons, "green"
    elif score >= 3:
        return "BUY", reasons, "lightgreen"
    elif score >= 0:
        return "HOLD", reasons, "yellow"
    elif score >= -2:
        return "REDUCE", reasons, "orange"
    else:
        return "SELL", reasons, "red"


def create_summary_sheet(analysis: dict) -> pd.DataFrame:
    """Create comprehensive summary table."""
    ticker = analysis['ticker']
    processor = analysis['processor']
    results = analysis['results']
    
    desc = results['descriptive_stats']
    risk = results['risk_metrics']
    market = results['market_sensitivity']
    info = results.get('company_info', {})
    corp = results.get('corporate_actions', {})
    
    prices = processor.close_prices
    current_price = float(prices.iloc[-1])
    
    # Build summary sections
    sections = []
    
    # === COMPANY OVERVIEW ===
    sections.append({'Section': 'COMPANY OVERVIEW', 'Metric': '', 'Value': '', 'Comment': ''})
    sections.append({
        'Section': '', 'Metric': 'Company Name',
        'Value': info.get('company_name', ticker),
        'Comment': ''
    })
    sections.append({
        'Section': '', 'Metric': 'Sector',
        'Value': info.get('sector', 'N/A'),
        'Comment': ''
    })
    sections.append({
        'Section': '', 'Metric': 'Industry',
        'Value': info.get('industry', 'N/A'),
        'Comment': ''
    })
    
    # === PRICE & PERFORMANCE ===
    sections.append({'Section': 'PRICE & PERFORMANCE', 'Metric': '', 'Value': '', 'Comment': ''})
    sections.append({
        'Section': '', 'Metric': 'Current Price',
        'Value': f'${current_price:.2f}',
        'Comment': f'As of {prices.index[-1].date()}'
    })
    sections.append({
        'Section': '', 'Metric': 'Annual Return',
        'Value': f'{desc["mean_annual_return"]*100:.2f}%',
        'Comment': 'Annualized average'
    })
    sections.append({
        'Section': '', 'Metric': 'Total Return',
        'Value': f'{((prices.iloc[-1]/prices.iloc[0] - 1)*100):.2f}%',
        'Comment': 'Period total'
    })
    sections.append({
        'Section': '', 'Metric': 'Win Rate',
        'Value': f'{desc["positive_days_pct"]:.1f}%',
        'Comment': '% of positive days'
    })
    
    # === RISK METRICS ===
    sections.append({'Section': 'RISK METRICS', 'Metric': '', 'Value': '', 'Comment': ''})
    sections.append({
        'Section': '', 'Metric': 'Annual Volatility',
        'Value': f'{desc["std_annual_return"]*100:.2f}%',
        'Comment': 'Annualized std dev'
    })
    sections.append({
        'Section': '', 'Metric': 'Sharpe Ratio',
        'Value': f'{desc["sharpe_ratio"]:.3f}',
        'Comment': 'Risk-adjusted return'
    })
    sections.append({
        'Section': '', 'Metric': 'Sortino Ratio',
        'Value': f'{desc["sortino_ratio"]:.3f}',
        'Comment': 'Downside risk-adjusted'
    })
    sections.append({
        'Section': '', 'Metric': 'Max Drawdown',
        'Value': f'{risk["max_drawdown"]*100:.2f}%',
        'Comment': 'Largest peak-to-trough'
    })
    sections.append({
        'Section': '', 'Metric': 'VaR (95%)',
        'Value': f'{risk["var_95"]*100:.2f}%',
        'Comment': '1-day max loss (95% conf)'
    })
    
    # === MARKET SENSITIVITY ===
    if market:
        sections.append({'Section': 'MARKET SENSITIVITY', 'Metric': '', 'Value': '', 'Comment': ''})
        sections.append({
            'Section': '', 'Metric': 'Beta',
            'Value': f'{market.get("beta", 0):.3f}',
            'Comment': 'Systematic risk vs market'
        })
        sections.append({
            'Section': '', 'Metric': 'Alpha (Annual)',
            'Value': f'{market.get("alpha_annual", 0)*100:.2f}%',
            'Comment': 'Excess return vs market'
        })
        sections.append({
            'Section': '', 'Metric': 'R-Squared',
            'Value': f'{market.get("r_squared", 0):.3f}',
            'Comment': 'Explained by market'
        })
    
    # === VALUATION ===
    sections.append({'Section': 'VALUATION', 'Metric': '', 'Value': '', 'Comment': ''})
    if processor.info:
        sections.append({
            'Section': '', 'Metric': 'P/E Ratio',
            'Value': f'{processor.info.get("trailingPE", 0):.2f}' if processor.info.get('trailingPE') else 'N/A',
            'Comment': 'Price to earnings'
        })
        sections.append({
            'Section': '', 'Metric': 'P/B Ratio',
            'Value': f'{processor.info.get("priceToBook", 0):.2f}' if processor.info.get('priceToBook') else 'N/A',
            'Comment': 'Price to book'
        })
        sections.append({
            'Section': '', 'Metric': 'Dividend Yield',
            'Value': f'{processor.info.get("dividendYield", 0)*100:.2f}%' if processor.info.get('dividendYield') else 'N/A',
            'Comment': 'Annual yield'
        })
    
    # === DIVIDENDS ===
    if corp.get('num_dividends', 0) > 0:
        sections.append({'Section': 'DIVIDENDS', 'Metric': '', 'Value': '', 'Comment': ''})
        sections.append({
            'Section': '', 'Metric': 'Dividends Paid',
            'Value': f'{corp["num_dividends"]}',
            'Comment': 'Number of payments'
        })
        sections.append({
            'Section': '', 'Metric': 'Total Dividends',
            'Value': f'${corp["total_dividends"]:.2f}',
            'Comment': 'Cumulative'
        })
    
    # === FORECAST ===
    if analysis['forecast']:
        sections.append({'Section': 'FORECAST', 'Metric': '', 'Value': '', 'Comment': ''})
        last_price = prices.iloc[-1]
        forecast_price = analysis['forecast']['predictions'][-1]
        change = ((forecast_price / last_price) - 1) * 100
        
        sections.append({
            'Section': '', 'Metric': 'Forecast Price',
            'Value': f'${forecast_price:.2f}',
            'Comment': f'{len(analysis["forecast"]["predictions"])} days ahead'
        })
        sections.append({
            'Section': '', 'Metric': 'Expected Change',
            'Value': f'{change:+.2f}%',
            'Comment': 'vs current price'
        })
        sections.append({
            'Section': '', 'Metric': 'Forecast Accuracy',
            'Value': f'{analysis["forecast"]["directional_accuracy"]:.1f}%',
            'Comment': 'Directional accuracy'
        })
    
    return pd.DataFrame(sections)


def export_hedge_fund_report(analysis: dict, output_path: str):
    """Generate hedge-fund-ready Excel report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ticker = analysis['ticker']
    
    # Generate recommendation
    recommendation, reasons, color = generate_recommendation(
        analysis['results'], analysis['processor']
    )
    
    # Create summary
    summary_df = create_summary_sheet(analysis)
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # === FORMATS ===
        title_fmt = workbook.add_format({
            'bold': True, 'font_size': 18, 'font_color': '#1F4788',
            'align': 'center'
        })
        
        section_fmt = workbook.add_format({
            'bold': True, 'font_size': 12, 'bg_color': '#4472C4',
            'font_color': 'white', 'align': 'left'
        })
        
        header_fmt = workbook.add_format({
            'bold': True, 'bg_color': '#D9E1F2', 'border': 1,
            'align': 'center'
        })
        
        rec_buy_fmt = workbook.add_format({
            'bold': True, 'font_size': 16, 'bg_color': '#C6EFCE',
            'font_color': '#006100', 'align': 'center', 'border': 2
        })
        
        rec_hold_fmt = workbook.add_format({
            'bold': True, 'font_size': 16, 'bg_color': '#FFEB9C',
            'font_color': '#9C6500', 'align': 'center', 'border': 2
        })
        
        rec_sell_fmt = workbook.add_format({
            'bold': True, 'font_size': 16, 'bg_color': '#FFC7CE',
            'font_color': '#9C0006', 'align': 'center', 'border': 2
        })
        
        # === SHEET 1: EXECUTIVE SUMMARY ===
        ws = workbook.add_worksheet('Executive Summary')
        ws.set_column('A:A', 25)
        ws.set_column('B:B', 20)
        ws.set_column('C:C', 30)
        
        row = 0
        
        # Title
        ws.merge_range(row, 0, row, 2, f'STOCK ANALYSIS REPORT: {ticker}', title_fmt)
        row += 2
        
        # Recommendation
        ws.write(row, 0, 'RECOMMENDATION:', header_fmt)
        rec_fmt = rec_buy_fmt if 'BUY' in recommendation else (rec_hold_fmt if 'HOLD' in recommendation else rec_sell_fmt)
        ws.merge_range(row, 1, row, 2, recommendation, rec_fmt)
        row += 2
        
        # Key reasons
        ws.write(row, 0, 'Key Factors:', section_fmt)
        row += 1
        for reason in reasons[:5]:  # Top 5 reasons
            ws.write(row, 0, f'  • {reason}')
            row += 1
        row += 1
        
        # Summary table
        ws.write(row, 0, 'Metric', header_fmt)
        ws.write(row, 1, 'Value', header_fmt)
        ws.write(row, 2, 'Comment', header_fmt)
        row += 1
        
        current_section = None
        for _, data_row in summary_df.iterrows():
            if data_row['Section']:
                ws.merge_range(row, 0, row, 2, data_row['Section'], section_fmt)
                current_section = data_row['Section']
                row += 1
            else:
                ws.write(row, 0, data_row['Metric'])
                ws.write(row, 1, data_row['Value'])
                ws.write(row, 2, data_row['Comment'])
                row += 1
        
        # === SHEET 2: CHARTS APPENDIX ===
        charts_ws = workbook.add_worksheet('Charts Appendix')
        
        row = 0
        charts_ws.merge_range(row, 0, row, 3, 'VISUAL ANALYSIS', title_fmt)
        row += 2
        
        charts = analysis['charts']
        chart_configs = [
            ('price_history', 'Price History'),
            ('returns_dist', 'Returns Distribution'),
            ('volatility', 'Rolling Volatility'),
            ('forecast', 'Price Forecast')
        ]
        
        for chart_key, chart_title in chart_configs:
            if chart_key in charts:
                charts_ws.write(row, 0, chart_title, section_fmt)
                row += 1
                try:
                    charts_ws.insert_image(row, 0, str(charts[chart_key]),
                                          {'x_scale': 0.7, 'y_scale': 0.7})
                    row += 25
                except Exception:
                    row += 2
        
        # Footer
        row += 2
        footer_text = f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Analysis Time: {analysis["analysis_time"]:.1f}s'
        charts_ws.write(row, 0, footer_text)
    
    print(f"\n✓ Hedge fund report saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive hedge-fund-ready stock report",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--start', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--benchmark', type=str, default='^GSPC',
                       help='Benchmark (default: S&P 500)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"reports/{args.ticker}_hedge_fund_report.xlsx"
    
    try:
        # Run complete analysis
        analysis = run_complete_analysis(args.ticker, args.start, args.benchmark)
        
        # Generate report
        print("\nGenerating hedge fund report...")
        export_hedge_fund_report(analysis, args.output)
        
        # Summary
        processor = analysis['processor']
        results = analysis['results']
        desc = results['descriptive_stats']
        
        print(f"\n{'='*60}")
        print("EXECUTIVE SUMMARY")
        print(f"{'='*60}")
        print(f"Current Price: ${processor.close_prices.iloc[-1]:.2f}")
        print(f"Annual Return: {desc['mean_annual_return']*100:.2f}%")
        print(f"Sharpe Ratio: {desc['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {results['risk_metrics']['max_drawdown']*100:.2f}%")
        print(f"\nReport: {args.output}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()