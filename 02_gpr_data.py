"""
Script 2: Download the Caldara-Iacoviello GPR Index
=====================================================
DATA SOURCE:

Caldara-Iacoviello GPR Index (matteoiacoviello.com)
   - Website:  https://www.matteoiacoviello.com/gpr.htm
   - Requires: No API key — freely available public dataset
   - Package:  requests, pandas, xlrd
   - File:     data_gpr_export.xls (downloaded automatically)
   - Contains: GPR benchmark index, GPRA (acts), GPRT (threats)
   - Source:   Caldara, D. and Iacoviello, M. (2022), American Economic Review

SUB-SAMPLE DEFINITIONS (consistent with paper):
   War/high-risk  : GPR > 120 OR within episode window
   Control group  : GPR < 80  AND no episode active

   NOTE ON SAMPLE PERIOD (1991-2024):
   Extended from 1993 to 1991 to capture the Gulf War episode
   (January–March 1991). This is the clearest GPRA-dominant episode
   in the dataset (GPRA/GPRT ratio = 1.23) and one of the paper's
   three case studies. Active episodes within 1991-2024:
     - Gulf War         : Jan-Mar 1991  (3 months) — NOW ACTIVE
     - 9/11 + Afghanistan: Sep-Nov 2001 (3 months)
     - Russia-Ukraine   : Feb-Apr 2022  (3 months)
   Expected N ≈ 75 war months and ≈ 120 control months for 1991-2024.

OUTPUT:
   data/gpr_monthly_data.csv   <- GPR, GPRA, GPRT monthly dataset
   data/gpr_plot.png           <- GPR index plot with episode markers

INSTALL (one-time):
   pip install pandas requests xlrd matplotlib openpyxl
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import requests
import io
import os
import numpy as np
import matplotlib.pyplot as plt

START   = "1991-01-01"   # Extended from 1993 to capture Gulf War (Jan-Mar 1991)
END     = "2024-12-31"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"

print("Downloading GPR index from matteoiacoviello.com...")
print(f"URL: {GPR_URL}")

try:
    headers = {"User-Agent": "Mozilla/5.0 (research data download)"}
    response = requests.get(GPR_URL, headers=headers, timeout=30, verify=False)
    response.raise_for_status()
    print(f"Download successful: {len(response.content) / 1024:.1f} KB")
    gpr_raw = pd.read_excel(io.BytesIO(response.content))
    print(f"Raw shape: {gpr_raw.shape}")
    print("Columns found:", list(gpr_raw.columns))

except Exception as e:
    print(f"\nAutomatic download failed: {e}")
    print("Trying fallback local file...")
    gpr_raw = pd.read_excel("data_gpr_export.xls")

print("\nParsing GPR data...")

col_map = {}
for col in gpr_raw.columns:
    c = str(col).lower().strip()
    if "month" in c or "date" in c or "year" in c:
        col_map[col] = "date"
    elif c == "gpr" or c == "gprbenchmark" or "benchmark" in c:
        col_map[col] = "GPR"
    elif c == "gpra" or "act" in c:
        col_map[col] = "GPRA"
    elif c == "gprt" or "threat" in c:
        col_map[col] = "GPRT"

gpr_renamed = gpr_raw.rename(columns=col_map)

if "date" in gpr_renamed.columns:
    gpr_renamed["date"] = pd.to_datetime(gpr_renamed["date"])
    gpr_renamed = gpr_renamed.set_index("date")
else:
    if "year" in gpr_renamed.columns and "month" in gpr_renamed.columns:
        gpr_renamed["date"] = pd.to_datetime(
            dict(year=gpr_renamed["year"], month=gpr_renamed["month"], day=1)
        )
        gpr_renamed = gpr_renamed.set_index("date")

available_cols = [c for c in ["GPR", "GPRA", "GPRT"] if c in gpr_renamed.columns]
if not available_cols:
    print("WARNING: Standard column names not found. Available:", list(gpr_renamed.columns))
    numeric_cols = gpr_renamed.select_dtypes(include="number").columns[:3]
    gpr_renamed.columns = ["GPR", "GPRA", "GPRT"][:len(numeric_cols)]
    available_cols = list(gpr_renamed.columns)

gpr = gpr_renamed[available_cols].copy()
gpr.index = pd.to_datetime(gpr.index).to_period("M").to_timestamp()
gpr = gpr.loc[START:END]
print(f"\nFiltered to {START} - {END}: {len(gpr)} monthly observations")

# ── LOG-DIFFERENCED GPR VARIABLES ────────────────────────────────────────────
# dln_X = log(X_t) - log(X_{t-1}) = monthly change in geopolitical risk
# This is the regressor used in all OLS and horse race regressions.
print("Computing log-differenced GPR variables (dln_GPR, dln_GPRA, dln_GPRT)...")
for col in available_cols:
    gpr[f"dln_{col}"] = np.log(gpr[col].replace(0, np.nan)).diff()

# ── EPISODE DUMMIES ───────────────────────────────────────────────────────────
# D_GULF: Gulf War (Jan-Mar 1991) — NOW ACTIVE with 1991 sample start.
#         3 months: Jan 1991 (air campaign), Feb 1991 (ground war), Mar 1991 (ceasefire)
#         GPRA/GPRT ratio = 1.23 → clearest GPRA-dominant episode in dataset
# D_911:  9/11 + Afghanistan War (Sep-Nov 2001) — 3 active months.
#         GPRA/GPRT ratio = 1.29 → second GPRA-dominant episode
# D_UKR:  Russia-Ukraine invasion (Feb-Apr 2022) — 3 active months.
#         GPRT-dominant during threat phase (Nov 2021-Jan 2022) before invasion
gpr["D_GULF"] = ((gpr.index >= "1991-01-01") & (gpr.index <= "1991-03-31")).astype(int)
gpr["D_911"]  = ((gpr.index >= "2001-09-01") & (gpr.index <= "2001-11-30")).astype(int)
gpr["D_UKR"]  = ((gpr.index >= "2022-02-01") & (gpr.index <= "2022-04-30")).astype(int)

episode_mask = (gpr["D_GULF"] == 1) | (gpr["D_911"] == 1) | (gpr["D_UKR"] == 1)

# ── SUB-SAMPLE FLAGS ──────────────────────────────────────────────────────────
# War/high-risk: GPR > 120 OR within episode window
# With 1991 start: Gulf War (3 months) + 9/11 (3 months) + Ukraine (3 months)
# + all GPR>120 months across 1991-2024
# Expected N ≈ 75 months
gpr["WAR_HIGH"] = ((gpr["GPR"] > 120) | episode_mask).astype(int)

# Control group: GPR < 80 AND no episode active
# Genuinely tranquil months — GPR should be INSIGNIFICANT here.
# Expected N ≈ 120 months with 1991 start
gpr["CONTROL"] = ((gpr["GPR"] < 80) & ~episode_mask).astype(int)

print(f"\nSub-sample sizes (1991-2024 sample):")
print(f"  War/high-risk months (GPR>120 or episode): {gpr['WAR_HIGH'].sum()}")
print(f"  Control group months (GPR<80, no episode): {gpr['CONTROL'].sum()}")
print(f"  Gulf War months active (Jan-Mar 1991):     {gpr['D_GULF'].sum()}")
print(f"  9/11 months active (Sep-Nov 2001):         {gpr['D_911'].sum()}")
print(f"  Ukraine months active (Feb-Apr 2022):      {gpr['D_UKR'].sum()}")
print(f"  Overlap check: {((gpr['WAR_HIGH']==1) & (gpr['CONTROL']==1)).sum()} months (must be 0)")

print("\nGPR Descriptive Statistics:")
print(gpr[available_cols].describe().round(2))

out_path = os.path.join(OUT_DIR, "gpr_monthly_data.csv")
gpr.to_csv(out_path)
print(f"\nSaved: {out_path}")

fig, ax = plt.subplots(figsize=(14, 5))
gpr["GPR"].plot(ax=ax, color="firebrick", linewidth=1.2, label="GPR Benchmark")
if "GPRA" in gpr.columns:
    gpr["GPRA"].plot(ax=ax, color="darkorange", linewidth=0.8, alpha=0.7, label="GPRA (Acts)")
if "GPRT" in gpr.columns:
    gpr["GPRT"].plot(ax=ax, color="steelblue", linewidth=0.8, alpha=0.7, label="GPRT (Threats)")

for ep_start, ep_end, label in [
    ("1991-01-01", "1991-03-31", "Gulf War"),
    ("2001-09-01", "2001-11-30", "9/11"),
    ("2022-02-01", "2022-04-30", "Ukraine"),
]:
    ax.axvspan(pd.Timestamp(ep_start), pd.Timestamp(ep_end), alpha=0.15, color="red")
    mid = pd.Timestamp(ep_start) + (pd.Timestamp(ep_end) - pd.Timestamp(ep_start)) / 2
    ax.text(mid, ax.get_ylim()[1] * 0.9, label, ha="center", fontsize=8, color="darkred")

ax.axhline(80,  color="gray",  linestyle="--", linewidth=0.8, label="GPR=80  (control threshold)")
ax.axhline(120, color="black", linestyle="--", linewidth=0.8, label="GPR=120 (war threshold)")
ax.set_title("Caldara-Iacoviello Geopolitical Risk Index (1991-2024)")
ax.set_ylabel("GPR Index Level")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gpr_plot.png"), dpi=120)
print(f"Plot saved: {os.path.join(OUT_DIR, 'gpr_plot.png')}")
print("\nScript 02 complete.")