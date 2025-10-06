# Stock Analysis — Command Guide & Workflow
This guide shows **what to run, when to run it, and what you get**. It’s organized by workflow stage, with copy‑pastable commands and brief explanations of inputs/outputs.
> Tip: Run commands from your project **root** (e.g., `~/Desktop/stock_analysis`).  
> Your current timezone is assumed to be Asia/Singapore; dates in saved metadata are in local time.

---
## 0) One‑time setup (recommended)
Create and activate your virtual environment, then install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Keep your venv active while working.

---
## 1) Data Fetching (`data_fetching.py`)
Pulls historical **daily** prices/metadata from Yahoo Finance, saves into `data/` as Parquet/CSV plus a session meta JSON (used downstream).

### Clean current session artifacts
```bash
python data_fetching.py --clear       # deletes current-session files (e.g., last_fetch.*)
python data_fetching.py --clear-all   # deletes ALL cached files in data/
```

### Fetch selected tickers
```bash
python data_fetching.py --tickers AAPL,MSFT
```
**What you get**
- `data/last_fetch.parquet` (or `.csv`) — prices for the selected tickers
- `data/last_fetch_meta.json` — session metadata (tickers, start/end dates, etc.)
- `data/last_fetch_info.json` — per‑ticker info (fundamentals/metadata from Yahoo)

> Keep using the same **current session** unless you intentionally change tickers. Downstream scripts use this session to stay consistent.

---
## 2) Data Processing (`utils/data_processing.py`)
Cleans/transforms the fetched dataset, computes returns/volatility series and saves “processed” artifacts for other modules.

```bash
python utils/data_processing.py --save-dir data/processed
```
**Outputs**
- Processed frames in `data/processed/` (e.g., returns, rolling vols, etc.) aligned to your **current session tickers**.

---
## 3) Visualization (`utils/visualization.py`)
Generates charts from the **current session** dataset (no refetch).

### Clean figures and/or cached session data created by visualization
```bash
python utils/visualization.py --clear         # clears current-session data files produced by viz
python utils/visualization.py --clear-all     # clears ALL files in data/ (dangerous)
```

### Save charts to a folder (don’t open windows)
```bash
python utils/visualization.py --save-dir figures --no-show
```

### Save charts **and** export an Excel summary
```bash
python utils/visualization.py --save-dir figures --export-xlsx reports/visual_summary.xlsx --no-show
```
**Outputs**
- PNG charts in `figures/`
- `reports/visual_summary.xlsx` with tables (Prices, Returns, RollingVol_21D) and charts sheet

> If you see a warning like “none of the requested tickers found”, it usually means visualization found a **different session** than expected. Run a **fresh fetch** first to resync: `python data_fetching.py --tickers ...`

---
## 4) Single‑Stock Analyses
All scripts below use the **current session tickers** unless you pass explicit `--tickers`/`--ticker`. Run them **after** data fetch (and optionally processing/visualization).

### 4.1 Event & Factor Sensitivity (`single_stock/E&F_sensitivity.py`)
Regress asset returns vs market & factor returns; optional event study (AR/CAR).
```bash
# Basic run (uses current-session ticker)
python single_stock/E&F_sensitivity.py

# Add event dates (comma-separated)
python single_stock/E&F_sensitivity.py --events 2023-01-15,2023-06-20

# Specify a different benchmark and factors CSV (Fama–French style)
python single_stock/E&F_sensitivity.py --benchmark ^GSPC --factors data/factors/ff_daily.csv

# Save charts & tables
python single_stock/E&F_sensitivity.py --save-dir figures/efs --export-xlsx reports/efs_summary.xlsx
```
**Outputs**
- Console summary of CAPM and (optional) multi‑factor betas/alpha, R²
- Optional PNGs: beta scatter, CAR plot
- Optional Excel: CAPM & MultiFactor sheets + Charts

> Note: factor CSV is optional. Without it, only CAPM vs benchmark is run.

### 4.2 Options & Derivatives (`single_stock/opt_deriv_metrics.py`)
Black–Scholes–Merton analytics: price & Greeks grids across strikes × maturities, optional charts/Excel. Uses the latest **spot** from the current session.
```bash
# Basic analysis (uses defaults; call options)
python single_stock/opt_deriv_metrics.py

# Override core parameters (S in price units; r/q as annual simple)
python single_stock/opt_deriv_metrics.py --S 100.0 --sigma 0.30 --r 0.03 --q 0.00

# Custom strikes and maturities (tenors in years; strikes as multiples of spot)
python single_stock/opt_deriv_metrics.py --strikes "0.7,0.85,1.0,1.15,1.3" --tenors "0.5,1.0,2.0"

# Put options
python single_stock/opt_deriv_metrics.py --option put

# Full analysis with custom output
python single_stock/opt_deriv_metrics.py --save-dir results/kakao_options --export-xlsx reports/kakao_options.xlsx
```
**Outputs**
- Tables: BSM price surface and Greeks (Δ, Γ, Vega, Θ, ρ)
- Charts: price surface heatmap, smile for a chosen tenor
- Optional Excel with tables + embedded charts

> Flags in this module are `--tenors` and `--option` (not `--maturities` / `--option-type`).

### 4.3 Valuation (`single_stock/valuation.py`)
Computes basic market/accounting ratios from the saved `info` + latest prices; peer relative z‑scores/percentiles; optional plots/Excel.
```bash
# Basic valuation (uses current-session tickers)
python single_stock/valuation.py

# Save figures & Excel
python single_stock/valuation.py --save-dir results/kakao_valuation --export-xlsx reports/kakao_valuation.xlsx

# Restrict to a subset of current-session tickers
python single_stock/valuation.py --tickers AAPL,MSFT
```
**Outputs**
- Console tables: base ratios (PE, P/FCF, PB, EV/EBITDA, EV/Sales, NetDebt, margins/ROE if available)
- Optional bar charts and Excel export

> The DCF parameters you listed (`--wacc`, `--growth`) aren’t part of the current skeleton; this module focuses on **ratios/relatives**. We can add DCF later.

### 4.4 Forecasting (`single_stock/forecasting.py`)
Compares ARIMA, Exponential Smoothing, (optional) GARCH and LSTM. Exports charts and an Excel summary.
```bash
# Basic (ARIMA + Exp + LSTM if TF is installed)
python single_stock/forecasting.py

# Specific models only
python single_stock/forecasting.py --models arima,exp

# Custom forecast horizon & lookback
python single_stock/forecasting.py --forecast-days 60 --lookback 120

# All models including GARCH
python single_stock/forecasting.py --models arima,exp,garch,lstm
```
**Outputs**
- Figures: forecast comparison (w/ confidence intervals), model performance
- Excel: summary metrics + per‑model predictions

> LSTM requires TensorFlow; GARCH requires `arch`. If unavailable, those models are skipped gracefully.

---
## 5) Typical End‑to‑End Sequences

### A) Fresh analysis of new tickers
```bash
# 1) Fetch
python data_fetching.py --clear
python data_fetching.py --tickers AAPL,MSFT

# 2) Process & visualize
python utils/data_processing.py --save-dir data/processed
python utils/visualization.py --save-dir figures --export-xlsx reports/visual_summary.xlsx --no-show

# 3) Single‑stock analytics
python single_stock/E&F_sensitivity.py --save-dir figures/efs --export-xlsx reports/efs_summary.xlsx
python single_stock/opt_deriv_metrics.py --save-dir figures/opt --export-xlsx reports/opt_metrics.xlsx
python single_stock/valuation.py --save-dir figures/val --export-xlsx reports/valuation_summary.xlsx

# 4) Forecasts
python single_stock/forecasting.py --forecast-days 30 --lookback 90 --models arima,exp,lstm
```

### B) Rerun visuals/forecasts on the same session
```bash
python utils/visualization.py --save-dir figures --export-xlsx reports/visual_summary.xlsx --no-show
python single_stock/forecasting.py --forecast-days 30 --lookback 60 --models arima,exp
```

### C) Clean slate
```bash
python data_fetching.py --clear-all
```

---
## 6) Troubleshooting
- **“none of the requested tickers found in dataset”**  
  Your visualization/processing picked up a different saved dataset than expected. Re‑fetch and try again:
  ```bash
  python data_fetching.py --tickers YOUR_TICKERS
  ```
- **Excel shows `####` for dates**  
  The visualization export formats the Date column and widens its width. If you still see `####`, widen the Excel column or switch to `yyyy-mm-dd hh:mm` format if you want timestamps.
- **Missing recent rolling vol values**  
  We forward‑fill small gaps and relax `min_periods` to avoid tail holes. If you prefer stricter windows (no fills), we can add a `--strict-vol` flag.
- **TensorFlow/Keras import errors**  
  The forecasting module uses **TensorFlow’s bundled Keras** (`from tensorflow import keras`). Make sure `tensorflow` is installed in your venv.

---
## 7) Where things are saved
- **Data**: `data/` (raw `last_fetch.*`, meta & info JSON)  
- **Processed**: `data/processed/`  
- **Figures**: `figures/` and module‑specific subfolders if you pass `--save-dir`  
- **Reports**: `reports/*.xlsx`

---
## 8) Next steps (optional)
- Add sector/industry awareness to valuation and forecasting (group by sector, sector‑relative betas and ratios).
- Add factor CSV auto‑download for `E&F_sensitivity.py`.
- Add DCF and WACC/Growth params to `valuation.py`.
- Add walk‑forward CV and ARIMAX/transformer variants to `forecasting.py`.
