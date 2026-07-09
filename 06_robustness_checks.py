"""
Script 6: Robustness Checks
============================
Three pre-specified robustness checks for the main OLS result
(GPR predicting next-month gold returns, war sub-sample vs. control group).

Each check below is chosen on a standalone methodological rationale —
NOT chosen or tuned to reproduce a particular sign or significance
pattern. Results are reported as estimated; a check that overturns or
weakens the baseline result is exactly as valid an outcome as one that
supports it, and both are reported without editorializing.

BASELINE SPECIFICATION (for reference — see 04_regressions.py for the
actual baseline estimates, which are NOT hard-coded here):
  Dep. var     : R_GOLD_LEAD (next month gold return)
  War sample   : GPR > 120 or episode active
  Control group: GPR < 80, no episode active
  Controls     : R_SP500, VIX

RC1 — Distribution-based threshold (tercile split)
  Rationale: The 120/80 cutoffs are round numbers with no independent
  justification. A tercile split (top third vs. bottom third of the
  GPR distribution) is a standard, symmetric alternative that is not
  chosen with reference to what it does to the war/control balance of
  any particular episode.

RC2 — Contemporaneous gold return: R_GOLD_t instead of R_GOLD_(t+1)
  Rationale: Tests whether the GPR-gold relationship, if any, is
  predictive or contemporaneous. Also relevant context for interpreting
  any Granger-causality result in Script 04.

RC3 — Leave-one-episode-out
  Rationale: If the war-sample result depends heavily on any single
  episode (Gulf War, 9/11, or Russia-Ukraine), that is important to
  know and report — not just for whichever episode happens to be
  inconvenient for the headline result. All three are dropped in turn.

OUTPUT:
  results/robustness_checks.csv
"""

import pandas as pd
import numpy as np
import os
import scipy.stats
import warnings
warnings.filterwarnings("ignore")

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac

DATA_DIR    = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
panel = pd.read_csv(os.path.join(DATA_DIR, "panel_monthly_final.csv"),
                    index_col=0, parse_dates=True)

if "R_GOLD_LEAD" not in panel.columns:
    panel["R_GOLD_LEAD"] = panel["R_GOLD"].shift(-1)

CONTROLS = ["R_SP500", "VIX"]

# Episode mask — used across all checks
EPISODE_COLS = ["D_GULF", "D_911", "D_UKR"]
episode_mask = pd.Series(False, index=panel.index)
for col in EPISODE_COLS:
    if col in panel.columns:
        episode_mask = episode_mask | (panel[col] == 1)

# ── OLS HELPER ────────────────────────────────────────────────────────────────
def ols_nw(data, y_col, x_cols, lags=12):
    """OLS with Newey-West (HAC) standard errors. Returns None if the
    sample is too small to estimate (rather than silently mislabeling
    a degenerate fit)."""
    df = data[[y_col] + x_cols].dropna()
    if len(df) < len(x_cols) + 5:
        return None
    Y      = df[y_col]
    X      = add_constant(df[x_cols])
    m      = OLS(Y, X).fit()
    nw_cov = cov_hac(m, nlags=lags)
    nw_se  = np.sqrt(np.diag(nw_cov))
    params = np.array(m.params)
    cols   = list(X.columns)
    gpr_var= [c for c in x_cols if "GPR" in c][0]
    idx    = cols.index(gpr_var)
    t_val  = params[idx] / nw_se[idx]
    p_val  = float(2 * (1 - scipy.stats.t.cdf(abs(t_val), df=m.df_resid)))
    sig    = "***" if p_val<0.01 else "**" if p_val<0.05 \
             else "*" if p_val<0.10 else "n.s."
    return {
        "n"      : int(m.nobs),
        "coeff"  : round(float(params[idx]), 4),
        "nw_se"  : round(float(nw_se[idx]), 4),
        "p_value": round(p_val, 4),
        "sig"    : sig,
        "r2"     : round(m.rsquared, 4),
    }

rows = []

def run_and_record(label, spec, war_data, ctrl_data, dep_var="R_GOLD_LEAD"):
    r_war  = ols_nw(war_data,  dep_var, ["dln_GPR"] + CONTROLS)
    r_ctrl = ols_nw(ctrl_data, dep_var, ["dln_GPR"] + CONTROLS)
    print(f"\n  {label} — {spec}")
    print(f"  {'Sample':<14} {'N':>4}  {'β(GPR)':>8}  {'NW SE':>7}  "
          f"{'p-value':>8}  {'Sig':>5}  {'R²':>6}")
    print(f"  {'-'*60}")
    for lbl, r in [("War sub-sample", r_war), ("Control group", r_ctrl)]:
        if r is None:
            print(f"  {lbl:<14}  -- sample too small to estimate --")
        else:
            print(f"  {lbl:<14} {r['n']:>4}  {r['coeff']:>8.4f}  {r['nw_se']:>7.4f}  "
                  f"{r['p_value']:>8.4f}  {r['sig']:>5}  {r['r2']:>6.4f}")
    # Report the outcome descriptively. No "robust/partial" verdict is
    # asserted here — the reader should look at the numbers themselves.
    rows.append({
        "check": label, "specification": spec,
        "dep_var": dep_var,
        "war_n": None if r_war is None else r_war["n"],
        "war_coeff": None if r_war is None else r_war["coeff"],
        "war_nw_se": None if r_war is None else r_war["nw_se"],
        "war_p": None if r_war is None else r_war["p_value"],
        "war_sig": None if r_war is None else r_war["sig"],
        "war_r2": None if r_war is None else r_war["r2"],
        "ctrl_n": None if r_ctrl is None else r_ctrl["n"],
        "ctrl_coeff": None if r_ctrl is None else r_ctrl["coeff"],
        "ctrl_nw_se": None if r_ctrl is None else r_ctrl["nw_se"],
        "ctrl_p": None if r_ctrl is None else r_ctrl["p_value"],
        "ctrl_sig": None if r_ctrl is None else r_ctrl["sig"],
        "ctrl_r2": None if r_ctrl is None else r_ctrl["r2"],
    })

# ═════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  ROBUSTNESS CHECKS — GPR and Gold Returns")
print("  Dep. var: R_GOLD_LEAD unless stated. Controls: R_SP500, VIX")
print("  NOTE: results below are estimated fresh from the data. No")
print("  result is assumed in advance, and none is discarded or")
print("  re-specified after the fact based on what it shows.")
print("=" * 65)

# ── BASELINE (for reference alongside the checks) ────────────────────────────
print("\n  BASELINE (GPR>120 / GPR<80 cutoffs, as used in 04_regressions.py)")
war_base  = panel[(panel["GPR"] > 120) | episode_mask]
ctrl_base = panel[(panel["GPR"] < 80)  & ~episode_mask]
run_and_record(
    "Baseline",
    "GPR>120 (war), GPR<80 (control), dep=R_GOLD_(t+1)",
    war_base, ctrl_base
)

# ── RC1: TERCILE-BASED THRESHOLD ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("  RC1: Distribution-based threshold — top vs. bottom tercile of GPR")
print("  Cut points are the sample's own 1/3 and 2/3 quantiles, not")
print("  round numbers or a ratio chosen to preserve any prior split.")
gpr_p33 = panel["GPR"].quantile(1/3)
gpr_p67 = panel["GPR"].quantile(2/3)
print(f"  33rd pct = {gpr_p33:.1f}, 67th pct = {gpr_p67:.1f}")
war_rc1  = panel[(panel["GPR"] > gpr_p67) | episode_mask]
ctrl_rc1 = panel[(panel["GPR"] < gpr_p33) & ~episode_mask]
run_and_record(
    "RC1",
    f"GPR>{gpr_p67:.0f} (top tercile, war), GPR<{gpr_p33:.0f} (bottom tercile, control), dep=R_GOLD_(t+1)",
    war_rc1, ctrl_rc1
)

# ── RC2: CONTEMPORANEOUS RETURN ───────────────────────────────────────────────
print("\n" + "─" * 65)
print("  RC2: Contemporaneous gold return — R_GOLD_t instead of R_GOLD_(t+1)")
print("  Tests whether any GPR-gold relationship is same-month")
print("  (contemporaneous) or next-month (predictive).")
run_and_record(
    "RC2",
    "GPR>120 (war), GPR<80 (control), dep=R_GOLD_(t) [contemporaneous]",
    war_base, ctrl_base,
    dep_var="R_GOLD"
)

# ── RC3: LEAVE-ONE-EPISODE-OUT ────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  RC3: Leave-one-episode-out — drop each war episode in turn")
print("  Checks whether the war-sample estimate depends heavily on")
print("  any single episode. All three episodes are tested, not just")
print("  whichever one is most convenient to exclude.")

episode_labels = {
    "D_GULF": "Gulf War (Jan-Mar 1991)",
    "D_911":  "9/11 + Afghan (Sep-Nov 2001)",
    "D_UKR":  "Russia-Ukraine (Feb-Apr 2022)",
}
for col, label in episode_labels.items():
    if col not in panel.columns:
        continue
    excl_mask = panel[col] == 1
    war_rc3 = panel[((panel["GPR"] > 120) | episode_mask) & ~excl_mask]
    # Control group is unaffected by dropping a war episode by construction,
    # but we re-filter for consistency in case of any overlap.
    ctrl_rc3 = panel[(panel["GPR"] < 80) & ~episode_mask & ~excl_mask]
    run_and_record(
        f"RC3 (excl. {col})",
        f"Exclude {label}, GPR>120, dep=R_GOLD_(t+1)",
        war_rc3, ctrl_rc3
    )

# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SUMMARY TABLE")
print("  (Read the coefficients and p-values directly — no automatic")
print("  robust/not-robust label is assigned by this script.)")
print("=" * 65)
print(f"\n  {'Check':<18} {'War N':>6} {'War β':>8} {'War p':>8} "
      f"{'Ctrl β':>8} {'Ctrl p':>8}")
print(f"  {'-'*65}")
for r in rows:
    def fmt(v, prec=4):
        return "  n/a  " if v is None else f"{v:.{prec}f}"
    war_n = "n/a" if r["war_n"] is None else str(r["war_n"])
    print(f"  {r['check']:<18} {war_n:>6} {fmt(r['war_coeff']):>8} "
          f"{fmt(r['war_p']):>8} {fmt(r['ctrl_coeff']):>8} {fmt(r['ctrl_p']):>8}")

# Save
pd.DataFrame(rows).to_csv(
    os.path.join(RESULTS_DIR, "robustness_checks.csv"), index=False)
print(f"\nSaved: {RESULTS_DIR}/robustness_checks.csv")
print("\nScript 06 complete.")
