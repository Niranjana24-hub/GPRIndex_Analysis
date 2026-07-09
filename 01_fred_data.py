"""
Script 1: Download all data for GPR-Gold paper
================================================
DATA SOURCES USED:

1. FRED API (Federal Reserve Economic Data)
   - Website:  https://fred.stlouisfed.org/
   - Requires: Free API key (already set below)
   - Package:  fredapi
   - Series downloaded:
       VIXCLS       = CBOE Volatility Index (VIX), daily -> monthly average
   - Note: Gold (GOLDAMGBD228NLBM) and S&P 500 (SP500) were permanently
     removed from FRED in January 2022. They are fetched from alternative
     sources below.
   - Note: CPI, real rates, and oil are NOT used in this analysis.
     Research question focuses on GPR predicting gold returns with only
     S&P 500 and VIX as controls.

2. GitHub Public Dataset (Gold Price)
   - Website:  https://github.com/datasets/gold-prices
   - Requires: No API key
   - Data:     Gold spot price (USD/troy oz), monthly

3. Yahoo Finance via yfinance (S&P 500)
   - Website:  https://finance.yahoo.com/
   - Ticker:   ^GSPC (S&P 500 index), monthly

SAMPLE PERIOD: January 1991 – December 2024 (N≈408 months)
   Extended from 1993 to 1991 to capture the Gulf War episode
   (January–March 1991), which is the clearest GPRA-dominant episode
   and one of the three case studies in the paper. Adding these months:
   - Increases war sub-sample from N=72 to N≈75
   - Rebalances horse race toward GPRA dominance (2 GPRA episodes vs 1 GPRT)
   - Completes the three-episode story in the regression data

VARIABLES USED IN REGRESSIONS:
   r_gold   = nominal gold log return (NO CPI deflation — consistent with GPR index)
   r_sp500  = nominal S&P 500 log return (equity control)
   vix      = CBOE VIX monthly average (general fear control)

OUTPUT:
   data/fred_monthly_data.csv   <- main dataset
   data/fred_series_plot.png    <- time series plot

INSTALL (one-time):
   pip install fredapi yfinance pandas numpy matplotlib
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fredapi import Fred
import yfinance as yf
import os

API_KEY  = "be74fefcaa1ac00fbd5af84b49561809"
START    = "1991-01-01"   # Extended from 1993 to capture Gulf War (Jan-Mar 1991)
END      = "2024-12-31"
OUT_DIR  = "data"
os.makedirs(OUT_DIR, exist_ok=True)

fred = Fred(api_key=API_KEY)

# Only VIX is needed from FRED
SERIES = {
    "vix": "VIXCLS",
}

print("Downloading FRED series...")
raw = {}
for name, series_id in SERIES.items():
    print(f"  Fetching {name} ({series_id})...")
    try:
        s = fred.get_series(series_id, observation_start=START, observation_end=END)
        raw[name] = s
        print(f"    OK: {len(s)} observations")
    except Exception as e:
        print(f"    ERROR: {e}")

print("\n  Fetching gold price from GitHub public dataset...")
try:
    url = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"
    gold_df = pd.read_csv(url)
    gold_df["Date"] = pd.to_datetime(gold_df["Date"])
    gold_df = gold_df.set_index("Date")
    gold_df = gold_df.loc[START:END]
    raw["gold_price_usd"] = gold_df["Price"]
    print(f"    OK: {len(raw['gold_price_usd'])} observations")
except Exception as e:
    print(f"    ERROR: {e}")

print("\n  Fetching S&P 500 from yfinance (^GSPC)...")
try:
    sp500_raw = yf.download(
        "^GSPC",
        start=START,
        end=END,
        interval="1mo",
        auto_adjust=True,
        progress=False
    )
    raw["sp500_index"] = sp500_raw["Close"].squeeze()
    print(f"    OK: {len(raw['sp500_index'])} observations")
except Exception as e:
    print(f"    ERROR: {e}")

print("\nDownloaded:", list(raw.keys()))

required = ["gold_price_usd", "sp500_index", "vix"]
missing = [k for k in required if k not in raw]
if missing:
    print(f"FAILED to download: {missing}")
    exit(1)

print("\nResampling to monthly frequency...")

def to_monthly_last(s, name=""):
    return s.resample("MS").last().rename(name)

def to_monthly_mean(s, name=""):
    return s.resample("MS").mean().rename(name)

monthly = pd.DataFrame({
    "gold_price": to_monthly_last(raw["gold_price_usd"], "gold_price"),
    "sp500":      to_monthly_last(raw["sp500_index"],    "sp500"),
    "vix":        to_monthly_mean(raw["vix"],            "vix"),
})
monthly.index.name = "date"

# ── NOMINAL LOG RETURNS (no CPI deflation) ──────────────────────────────────
# Rationale: GPR index is not inflation-adjusted, so keeping returns nominal
# is internally consistent and standard in the GPR-gold literature.
print("Computing nominal log returns (no CPI deflation)...")
monthly["r_gold"]  = np.log(monthly["gold_price"] / monthly["gold_price"].shift(1)) * 100
monthly["r_sp500"] = np.log(monthly["sp500"]       / monthly["sp500"].shift(1))      * 100

# ── EPISODE DUMMIES ──────────────────────────────────────────────────────────
print("Adding episode dummies...")
monthly["gulf_war"]  = ((monthly.index >= "1991-01-01") & (monthly.index <= "1991-03-31")).astype(int)
monthly["nine11"]    = ((monthly.index >= "2001-09-01") & (monthly.index <= "2001-11-30")).astype(int)
monthly["ukraine22"] = ((monthly.index >= "2022-02-01") & (monthly.index <= "2022-04-30")).astype(int)

# ── WAR / CONTROL FLAGS (set here for convenience; also set in Script 2) ─────
monthly["episode_active"] = (
    (monthly["gulf_war"]  == 1) |
    (monthly["nine11"]    == 1) |
    (monthly["ukraine22"] == 1)
).astype(int)

final_cols = [
    "gold_price", "r_gold",
    "sp500",      "r_sp500",
    "vix",
    "gulf_war", "nine11", "ukraine22", "episode_active"
]
monthly_final = monthly[final_cols].dropna(subset=["r_gold", "r_sp500", "vix"])

# Restrict to study period
monthly_final = monthly_final.loc[START:END]

out_path = os.path.join(OUT_DIR, "fred_monthly_data.csv")
monthly_final.to_csv(out_path)
print(f"\nSaved: {out_path}")
print(f"Shape: {monthly_final.shape}")
print(f"Date range: {monthly_final.index.min()} to {monthly_final.index.max()}")
print("\nFirst 5 rows:")
print(monthly_final[["r_gold", "r_sp500", "vix"]].head())

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
monthly_final["gold_price"].plot(ax=axes[0], title="Gold Price (USD/troy oz, nominal)", color="goldenrod")
monthly_final["r_gold"].plot(ax=axes[1], title="Gold Monthly Log Returns, % (nominal)", color="darkorange")
monthly_final["r_sp500"].plot(ax=axes[2], title="S&P 500 Monthly Log Returns, % (nominal)", color="steelblue")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fred_series_plot.png"), dpi=120)
print(f"\nPlot saved: {os.path.join(OUT_DIR, 'fred_series_plot.png')}")
print("\nScript 01 complete.")