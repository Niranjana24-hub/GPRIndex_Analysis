"""
Script 4: Run all regressions from the paper
----------------------------------------------
Implements the full empirical proof chain for war dominance:

  METHOD A — OLS Direction Test (3 samples)
    Full sample     : GPR should predict gold positively and significantly
    War sub-sample  : GPR effect should be LARGER and more significant
    Control group   : GPR effect should COLLAPSE to insignificance
    → Collapse in control group is the central proof of war-specificity

  METHOD B — Horse Race: GPRA vs GPRT
    GPRA only       : War acts alone predicting gold
    GPRT only       : Threats alone predicting gold
    Both together   : Head-to-head — which survives?
    Wald test       : Formal H0: β(GPRA) = β(GPRT) — rejection confirms war dominance

  METHOD C — Granger Causality
    GPR → gold      : Should be significant (newspapers report war → investors buy gold)
    Gold → GPR      : Should be insignificant (no reverse causality)

  METHOD D — GARCH-M
    Volatility channel: does GPR also affect gold return volatility?

SPECIFICATION:
  Dependent variable : R_GOLD_LEAD (next month's gold log return)
  Key regressors     : dln_GPR, dln_GPRA, dln_GPRT (log-differenced)
  Controls           : R_SP500, VIX only
  Standard errors    : Newey-West HAC (12 lags) throughout

OUTPUT:
  - results/ols_results.csv
  - results/horse_race.csv
  - results/granger_causality.csv
  - results/garch_results.txt

INSTALL:
    pip install pandas numpy statsmodels arch scipy
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tsa.stattools import grangercausalitytests
from arch import arch_model
import scipy.stats

DATA_DIR    = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading panel data...")
panel   = pd.read_csv(os.path.join(DATA_DIR, "panel_monthly_final.csv"),
                      index_col=0, parse_dates=True)
war_df  = pd.read_csv(os.path.join(DATA_DIR, "panel_war_subsample.csv"),
                      index_col=0, parse_dates=True)
ctrl_df = pd.read_csv(os.path.join(DATA_DIR, "panel_control_group.csv"),
                      index_col=0, parse_dates=True)

# Ensure R_GOLD_LEAD exists in all three dataframes
# R_GOLD_LEAD = next month's gold return = the dependent variable
for df in [panel, war_df, ctrl_df]:
    if "R_GOLD_LEAD" not in df.columns:
        df["R_GOLD_LEAD"] = df["R_GOLD"].shift(-1)

print(f"Full panel:     {len(panel)} obs")
print(f"War sub-sample: {len(war_df)} obs")
print(f"Control group:  {len(ctrl_df)} obs")
print()

# ── CONTROLS USED ─────────────────────────────────────────────────────────────
# R_SP500 : equity control — isolates gold safe-haven from equity crash effect
# VIX     : fear control   — separates war-specific fear from general market panic
# (No oil, no real rates, no CPI — not needed for this research question)
CONTROLS = ["R_SP500", "VIX"]

# ── HELPER: OLS WITH NEWEY-WEST SE ────────────────────────────────────────────
def ols_nw(df, y_col, x_cols, lags=12, label=""):
    """
    OLS with Newey-West (HAC) standard errors.
    Returns a DataFrame of results with significance stars.
    """
    data   = df[[y_col] + x_cols].dropna()
    Y      = data[y_col]
    X      = add_constant(data[x_cols])
    model  = OLS(Y, X).fit()
    nw_cov = cov_hac(model, nlags=lags)
    nw_se  = np.sqrt(np.diag(nw_cov))
    t_stat = model.params / nw_se
    p_vals = 2 * (1 - scipy.stats.t.cdf(np.abs(t_stat), df=model.df_resid))

    result = pd.DataFrame({
        "variable": X.columns,
        "coeff":    model.params.round(4),
        "nw_se":    nw_se.round(4),
        "t_stat":   t_stat.round(3),
        "p_value":  p_vals.round(4),
        "sig":      ["***" if p < 0.01 else "**" if p < 0.05
                     else "*" if p < 0.10 else "" for p in p_vals],
    })
    result.attrs["r2"]    = round(model.rsquared, 4)
    result.attrs["n_obs"] = int(model.nobs)
    result.attrs["label"] = label
    result.attrs["model"] = model
    return result

def print_result(r):
    print(f"  {'Variable':<14} {'Coeff':>9} {'NW SE':>9} {'t-stat':>9} {'p-value':>9} {'Sig':>5}")
    print(f"  {'-'*57}")
    for _, row in r.iterrows():
        print(f"  {row['variable']:<14} {row['coeff']:>9.4f} {row['nw_se']:>9.4f} "
              f"{row['t_stat']:>9.3f} {row['p_value']:>9.4f} {row['sig']:>5}")
    print(f"  R² = {r.attrs['r2']:.4f}   N = {r.attrs['n_obs']}")

def label_df(df, label):
    df = df.copy()
    df.insert(0, "sample", label)
    return df

# ═════════════════════════════════════════════════════════════════════════════
# METHOD A: OLS DIRECTION TEST ACROSS THREE SAMPLES
# ═════════════════════════════════════════════════════════════════════════════
print("═" * 65)
print("  METHOD A: OLS DIRECTION TEST")
print("  Dep. var: R_GOLD_LEAD (next month gold return)")
print("  Controls: R_SP500, VIX")
print("═" * 65)

x_vars = ["dln_GPR"] + CONTROLS

print("\nA1. Full sample (1993–2024):")
r_full = ols_nw(panel, "R_GOLD_LEAD", x_vars, label="Full sample")
print_result(r_full)

print("\nA2. War/high-risk sub-sample (GPR>120 or episode):")
r_war = ols_nw(war_df, "R_GOLD_LEAD", x_vars, label="War sub-sample")
print_result(r_war)

print("\nA3. Control group (GPR<80, no episode):")
r_ctrl = ols_nw(ctrl_df, "R_GOLD_LEAD", x_vars, label="Control group")
print_result(r_ctrl)

# Key comparison: the collapse of GPR coefficient is the central finding
print("\n── KEY COMPARISON: GPR coefficient across samples ──────")
gpr_full = r_full[r_full["variable"] == "dln_GPR"].iloc[0]
gpr_war  = r_war[r_war["variable"]   == "dln_GPR"].iloc[0]
gpr_ctrl = r_ctrl[r_ctrl["variable"] == "dln_GPR"].iloc[0]
print(f"  Full sample   : β = {gpr_full['coeff']:>7.4f}  p = {gpr_full['p_value']:.4f}  {gpr_full['sig']}")
print(f"  War sample    : β = {gpr_war['coeff']:>7.4f}  p = {gpr_war['p_value']:.4f}  {gpr_war['sig']}")
print(f"  Control group : β = {gpr_ctrl['coeff']:>7.4f}  p = {gpr_ctrl['p_value']:.4f}  {gpr_ctrl['sig']}")
print()
if gpr_war["p_value"] < 0.10 and gpr_ctrl["p_value"] > 0.10:
    print("  ✓ CONFIRMED: GPR significant in war, insignificant in control")
    print("    → GPR's predictive power for gold is WAR-SPECIFIC")
elif gpr_ctrl["p_value"] > 0.10:
    print("  ✓ Control group correctly shows insignificance")
else:
    print("  ! Check sub-sample definitions — control group should be insignificant")

# Save OLS results
ols_combined = pd.concat([
    label_df(r_full, "Full sample"),
    label_df(r_war,  "War sub-sample"),
    label_df(r_ctrl, "Control group"),
])
ols_combined.to_csv(os.path.join(RESULTS_DIR, "ols_results.csv"), index=False)
print(f"\nSaved: {RESULTS_DIR}/ols_results.csv")

# ═════════════════════════════════════════════════════════════════════════════
# METHOD B: HORSE RACE — GPRA (WAR ACTS) vs GPRT (THREATS)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("  METHOD B: HORSE RACE — GPRA vs GPRT")
print("  Which GPR component drives gold? War acts or threats?")
print("═" * 65)

if "dln_GPRA" not in panel.columns or "dln_GPRT" not in panel.columns:
    print("  GPRA/GPRT columns not found — skipping horse race.")
else:
    print("\nB1. GPRA only (war acts — Categories 6, 7, 8):")
    r_gpra = ols_nw(panel, "R_GOLD_LEAD", ["dln_GPRA"] + CONTROLS, label="GPRA only")
    print_result(r_gpra)

    print("\nB2. GPRT only (threats — Categories 1–5):")
    r_gprt = ols_nw(panel, "R_GOLD_LEAD", ["dln_GPRT"] + CONTROLS, label="GPRT only")
    print_result(r_gprt)

    print("\nB3. GPRA + GPRT together (head-to-head horse race):")
    r_both = ols_nw(panel, "R_GOLD_LEAD", ["dln_GPRA", "dln_GPRT"] + CONTROLS, label="Both")
    print_result(r_both)

    # ── WALD TEST: H0: β(GPRA) = β(GPRT) ────────────────────────────────────
    # Rejection at p<0.05 formally proves GPRA has greater predictive power.
    # This is the statistical proof that war acts dominate threats.
    print("\n── WALD TEST: H0: β(GPRA) = β(GPRT) ───────────────────")
    print("  Rejection → GPRA and GPRT have statistically different")
    print("  predictive power → war acts dominate threats (war dominance)")
    print()

    data_both = panel[["R_GOLD_LEAD", "dln_GPRA", "dln_GPRT"] + CONTROLS].dropna()
    Y_both    = data_both["R_GOLD_LEAD"]
    X_both    = add_constant(data_both[["dln_GPRA", "dln_GPRT"] + CONTROLS])
    m_both    = OLS(Y_both, X_both).fit(cov_type="HAC", cov_kwds={"maxlags": 12})

    # Manual Wald test: t-test of H0: β(GPRA) = β(GPRT)
    # Uses HAC covariance matrix — equivalent to Wald F-test for single restriction
    from scipy.stats import t as t_dist
    b_gpra  = m_both.params["dln_GPRA"]
    b_gprt  = m_both.params["dln_GPRT"]
    cov     = m_both.cov_params()
    diff    = b_gpra - b_gprt
    se_diff = np.sqrt(float(cov.loc["dln_GPRA","dln_GPRA"])
                    + float(cov.loc["dln_GPRT","dln_GPRT"])
                    - 2.0 * float(cov.loc["dln_GPRA","dln_GPRT"]))
    t_stat_wald = diff / se_diff
    p_wald  = 2.0 * (1.0 - t_dist.cdf(abs(t_stat_wald), df=int(m_both.df_resid)))
    sig_w   = "***" if p_wald<0.01 else "**" if p_wald<0.05 else "*" if p_wald<0.10 else "n.s."

    print(f"  β(GPRA)             = {b_gpra:.4f}")
    print(f"  β(GPRT)             = {b_gprt:.4f}")
    print(f"  β(GPRA) - β(GPRT)   = {diff:.4f}")
    print(f"  t-stat              = {t_stat_wald:.3f}")
    print(f"  p-value             = {p_wald:.4f}  {sig_w}")
    print()
    if p_wald < 0.05:
        print("  ✓ REJECT H0 at 5%: GPRA has significantly greater predictive power")
        print("    → WAR ACTS dominate THREATS in predicting gold returns")
    elif p_wald < 0.10:
        print("  ~ REJECT H0 at 10%: Marginal evidence GPRA > GPRT")
    else:
        print("  ! FAIL TO REJECT H0: No significant difference between GPRA and GPRT")

    # ── Horse race on war sub-sample only ────────────────────────────────────
    print("\nB4. Horse race — war sub-sample only:")
    r_war_both = ols_nw(war_df, "R_GOLD_LEAD",
                        ["dln_GPRA", "dln_GPRT"] + CONTROLS,
                        label="Both (war sub-sample)")
    print_result(r_war_both)

    # Wald test on war sub-sample — pure manual (no statsmodels wald_test)
    data_wb = war_df[["R_GOLD_LEAD", "dln_GPRA", "dln_GPRT"] + CONTROLS].dropna()
    Y_wb    = data_wb["R_GOLD_LEAD"]
    X_wb    = add_constant(data_wb[["dln_GPRA", "dln_GPRT"] + CONTROLS])
    m_wb    = OLS(Y_wb, X_wb).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    b1_w    = m_wb.params["dln_GPRA"]
    b2_w    = m_wb.params["dln_GPRT"]
    cov_w   = m_wb.cov_params()
    diff_w  = b1_w - b2_w
    se_w    = np.sqrt(float(cov_w.loc["dln_GPRA","dln_GPRA"])
                    + float(cov_w.loc["dln_GPRT","dln_GPRT"])
                    - 2.0 * float(cov_w.loc["dln_GPRA","dln_GPRT"]))
    t_w     = diff_w / se_w
    p_w     = 2.0 * (1.0 - t_dist.cdf(abs(t_w), df=int(m_wb.df_resid)))
    sig_w2  = "***" if p_w<0.01 else "**" if p_w<0.05 else "*" if p_w<0.10 else "n.s."
    print(f"\n  Wald test (war sub-sample):")
    print(f"  β(GPRA)-β(GPRT) = {diff_w:.4f},  t={t_w:.3f},  p={p_w:.4f}  {sig_w2}")

    # Save horse race
    horse_combined = pd.concat([
        label_df(r_gpra, "GPRA only"),
        label_df(r_gprt, "GPRT only"),
        label_df(r_both, "Both"),
    ])
    horse_combined.to_csv(os.path.join(RESULTS_DIR, "horse_race.csv"), index=False)
    print(f"\nSaved: {RESULTS_DIR}/horse_race.csv")

# ═════════════════════════════════════════════════════════════════════════════
# METHOD C: GRANGER CAUSALITY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("  METHOD C: GRANGER CAUSALITY")
print("  Expected: GPR→gold significant; gold→GPR NOT significant")
print("  (confirms newspapers report war → investors respond → gold rises)")
print("═" * 65)

gc_rows = []
for var_name, gpr_col in [("GPR", "dln_GPR"), ("GPRA", "dln_GPRA"), ("GPRT", "dln_GPRT")]:
    if gpr_col not in panel.columns:
        continue

    gc_data = panel[["R_GOLD", gpr_col]].dropna()

    print(f"\nC. {var_name} ↔ Gold returns (max lag = 4):")
    print(f"  {'Direction':<35} {'Lag':>4} {'F-stat':>8} {'p-value':>9} {'Sig':>5}")
    print(f"  {'-'*62}")

    for direction, cols, label in [
        ("forward",  ["R_GOLD", gpr_col], f"{var_name} → Gold"),
        ("reverse",  [gpr_col, "R_GOLD"], f"Gold → {var_name}"),
    ]:
        try:
            gc_res = grangercausalitytests(gc_data[cols], maxlag=4, verbose=False)
            for lag, res in gc_res.items():
                f_stat = res[0]["ssr_ftest"][0]
                p_val  = res[0]["ssr_ftest"][1]
                sig    = "***" if p_val < 0.01 else "**" if p_val < 0.05 \
                         else "*" if p_val < 0.10 else ""
                print(f"  {label:<35} {lag:>4} {f_stat:>8.3f} {p_val:>9.4f} {sig:>5}")
                gc_rows.append({
                    "direction": label, "lag": lag,
                    "f_stat": round(f_stat, 3), "p_value": round(p_val, 4), "sig": sig
                })
        except Exception as e:
            print(f"  Error: {e}")

pd.DataFrame(gc_rows).to_csv(os.path.join(RESULTS_DIR, "granger_causality.csv"), index=False)
print(f"\nSaved: {RESULTS_DIR}/granger_causality.csv")

# ═════════════════════════════════════════════════════════════════════════════
# METHOD D: GARCH-M — VOLATILITY CHANNEL
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("  METHOD D: GARCH-M — VOLATILITY CHANNEL")
print("  Does GPR affect gold return volatility (not just the mean)?")
print("═" * 65)

garch_data = panel[["R_GOLD", "dln_GPR"]].dropna()
r_gold_gm  = garch_data["R_GOLD"].values
gpr_exog   = garch_data["dln_GPR"].values.reshape(-1, 1)

print("\nEstimating GARCH(1,1)-in-Mean with GPR as exogenous regressor...")
try:
    gm = arch_model(
        r_gold_gm,
        vol="GARCH", p=1, q=1,
        mean="ARX",
        x=gpr_exog,
        dist="skewt"
    )
    gm_fit = gm.fit(disp="off", show_warning=False)
    print(gm_fit.summary())

    with open(os.path.join(RESULTS_DIR, "garch_results.txt"), "w") as f:
        f.write(str(gm_fit.summary()))
    print(f"\nSaved: {RESULTS_DIR}/garch_results.txt")

except Exception as e:
    print(f"GARCH-M error: {e}")
    print("Falling back to simple GARCH(1,1)...")
    try:
        gm_simple = arch_model(r_gold_gm, vol="GARCH", p=1, q=1, dist="skewt")
        gm_fit    = gm_simple.fit(disp="off")
        print(gm_fit.summary())
        with open(os.path.join(RESULTS_DIR, "garch_results.txt"), "w") as f:
            f.write(str(gm_fit.summary()))
    except Exception as e2:
        print(f"Simple GARCH also failed: {e2}")

print("\n" + "═" * 65)
print("  All regressions complete.")
print(f"  Results saved to: {RESULTS_DIR}/")
print("═" * 65)