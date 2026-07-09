═══════════════════════════════════════════════════════════════════════════════
PYTHON DATA COLLECTION & ANALYSIS SCRIPTS
Paper: "Geopolitical Risk and the Gold Market as a Safe Haven"
Course: Media Economics | 2024-2025
═══════════════════════════════════════════════════════════════════════════════

SCRIPTS (run in this exact order):

  01_fred_data.py       Download gold price, S&P 500, VIX from FRED/Yahoo/GitHub
  02_gpr_data.py        Download GPR / GPRA / GPRT from matteoiacoviello.com
  03_merge_and_clean.py Merge datasets, run unit root tests, descriptive stats
  04_regressions.py     OLS direction test, horse race, Granger causality, GARCH-M
  05_dcc_garch.py       DCC-GARCH model, hedge ratios, portfolio weights
  06_robustness_checks.py  3 pre-specified robustness checks (see below)

NO MANUAL DATA FILES NEEDED — all data is downloaded automatically by the scripts.
NO COLUMN RENAMING NEEDED   — column names are handled internally by each script.

═══════════════════════════════════════════════════════════════════════════════
SAMPLE PERIOD
═══════════════════════════════════════════════════════════════════════════════

  January 1991 – December 2024 (N ≈ 408 monthly observations)

  WHY 1991 AND NOT 1993:
  The sample was extended from January 1993 to January 1991 to include the
  Gulf War episode (January–March 1991), since the paper discusses Gulf War,
  9/11, and Russia-Ukraine as case studies and a 1993 start would leave one
  of the three outside the regression data.

  IMPORTANT CAVEAT — this choice is not sample-composition-neutral:
  Including the Gulf War changes the mix of episodes in the war sub-sample
  from one GPRA-dominant episode (9/11) plus one GPRT-dominant episode
  (Russia-Ukraine) to two GPRA-dominant episodes (Gulf War + 9/11, GPRA/GPRT
  ratios of 1.23 and 1.29) plus one GPRT-dominant episode. Because the
  GPRA-vs-GPRT "horse race" in Script 04 compares the two sub-indices
  against each other, this mechanically shifts that comparison toward GPRA
  before any regression is run.

  This does not mean the 1991 start is wrong — there is a genuine
  case-coverage argument for it — but the horse-race result should NOT be
  cited as independent evidence for GPRA dominance without acknowledging
  that the sample window was itself chosen partly on the basis of which
  episodes it would include. Anyone using this pipeline should ideally also
  report the 1993-2024 results as a sensitivity check, and disclose both.

═══════════════════════════════════════════════════════════════════════════════
INSTALLATION (one-time setup, run in terminal before anything else)
═══════════════════════════════════════════════════════════════════════════════

  pip install fredapi yfinance pandas numpy statsmodels arch scipy matplotlib openpyxl requests xlrd

═══════════════════════════════════════════════════════════════════════════════
FRED API KEY (required for Script 01 only)
═══════════════════════════════════════════════════════════════════════════════

  1. Go to:       https://fred.stlouisfed.org/
  2. Create a free account
  3. Click:       My Account → API Keys → Request API Key
  4. Open 01_fred_data.py and replace the line:
         API_KEY = "be74fefcaa1ac00fbd5af84b49561809"
     with your own key.

  The existing key in the scripts is the project key and may stop working.
  Getting your own key takes under 2 minutes and is free.

═══════════════════════════════════════════════════════════════════════════════
GPR DATA (Script 02 — no key needed)
═══════════════════════════════════════════════════════════════════════════════

  Downloaded automatically from:
    https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls

  If the automatic download fails (e.g. network issue):
    1. Visit https://www.matteoiacoviello.com/gpr.htm
    2. Download data_gpr_export.xls manually
    3. Place it in the same folder as the scripts
    4. Script 02 will detect the local file automatically

═══════════════════════════════════════════════════════════════════════════════
VARIABLES USED IN THIS ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

  DEPENDENT VARIABLE:
    R_GOLD_LEAD   = next month's gold log return (t+1)
                    This makes the analysis a PREDICTION exercise:
                    "Does GPR this month predict gold returns next month?"
                    Using t+1 avoids any look-ahead bias.

  KEY REGRESSORS:
    dln_GPR       = log change in GPR benchmark index
                    Monthly change in newspaper coverage of geopolitical risk.
                    Positive = more geopolitical risk articles than last month.

    dln_GPRA      = log change in GPRA (Geopolitical Acts sub-index)
                    Covers Categories 6, 7, 8: war onset, war escalation,
                    terrorist acts. Captures REALISED events — "it happened."

    dln_GPRT      = log change in GPRT (Geopolitical Threats sub-index)
                    Covers Categories 1–5: war threats, military buildups,
                    nuclear threats, political instability, terror threats.
                    Captures ANTICIPATED risks — "it might happen."

  CONTROLS (two only — deliberately minimal):
    R_SP500       = nominal S&P 500 log return
                    Controls for equity market crash effect.
                    Isolates gold safe-haven demand from general equity sell-off.
                    Without this control, GPR coefficient partly captures equity
                    flight-to-safety rather than gold-specific demand.

    VIX           = CBOE Volatility Index monthly average
                    Controls for general market fear.
                    Separates war-specific demand from general market panic.
                    A rising VIX during low-GPR periods reflects earnings
                    uncertainty, not geopolitical risk.

  NOT INCLUDED (and why):
    CPI / real returns   — GPR index is not inflation-adjusted; nominal returns
                           are internally consistent and standard in this literature
    Real interest rates  — not required for the core research question
    Oil returns          — not required for the core research question

═══════════════════════════════════════════════════════════════════════════════
SUB-SAMPLE DEFINITIONS
═══════════════════════════════════════════════════════════════════════════════

  WAR / HIGH-RISK SUB-SAMPLE (N ≈ 76 months):
    GPR > 120  OR  within an episode window
    Episode windows (all three now active in the 1991–2024 sample):
      Gulf War       : Jan 1991 – Mar 1991  (GPRA/GPRT ratio = 1.23 — acts dominate)
      9/11 + Afghan  : Sep 2001 – Nov 2001  (GPRA/GPRT ratio = 1.29 — acts dominate)
      Russia–Ukraine : Feb 2022 – Apr 2022  (GPRT-dominant pre-invasion phase)

  CONTROL GROUP (N ≈ 116 months):
    GPR < 80  AND  no episode window active
    These are genuinely tranquil, low-geopolitical-risk months.
    The GPR coefficient should be INSIGNIFICANT here.

  MIDDLE GROUND (N ≈ 216 months — excluded from both groups):
    GPR between 80 and 120, no episode active.
    Deliberately excluded to ensure a clean contrast between
    genuinely high-risk and genuinely tranquil months.

  WHAT THIS SPLIT IS DESIGNED TO TEST:
    The hypothesis is that the GPR coefficient will be larger/more significant
    in the war sub-sample than in the control group. Whether that hypothesis
    holds is an empirical result of Script 04, not something to assume going
    in — the split should be reported and interpreted based on what the
    regression actually returns, including if the pattern is weak, mixed, or
    reversed.

  ON THE 2 GPRA / 1 GPRT EPISODE COMPOSITION:
    As noted above, this composition (2 GPRA-dominant episodes vs. 1 GPRT-
    dominant episode) is partly a consequence of the 1991 sample-start
    decision, not an independent finding. Any interpretation of the GPRA-vs-
    GPRT horse race should account for that.

  KNOWN DISCREPANCY — SUB-SAMPLE SIZE:
    This README and Script 02 report expected war/control sizes of
    N≈76 / N≈116 for the 1991-2024 sample. Script 03's own comments state
    N≈72 / N≈112 for a 1993-2024 sample, while also noting the paper itself
    reports N≈91 / N≈156. These three numbers do not agree. Before citing
    any sub-sample N in the paper, re-run Script 03 on the actual final
    panel and use the number it prints — do not carry forward N figures
    from this README, from code comments, or from an earlier draft of the
    paper without re-verifying them against current output.

═══════════════════════════════════════════════════════════════════════════════
OUTPUT STRUCTURE (created automatically when scripts run)
═══════════════════════════════════════════════════════════════════════════════

  data/
    fred_monthly_data.csv           ← gold price, S&P 500, VIX (from Script 01)
    gpr_monthly_data.csv            ← GPR, GPRA, GPRT + sub-sample flags (Script 02)
    panel_monthly_final.csv         ← MAIN DATASET — all variables merged
    panel_war_subsample.csv         ← War/high-risk months only (GPR>120 or episode)
    panel_control_group.csv         ← Non-war control months (GPR<80, no episode)
    descriptive_stats.csv           ← Summary statistics table
    unit_root_tests.csv             ← ADF, PP, KPSS stationarity tests
    fred_series_plot.png            ← Gold price and S&P 500 time series (1991–2024)
    gpr_plot.png                    ← GPR index with all 3 episode markers and thresholds

  results/
    ols_results.csv                 ← OLS direction test (full, war, control samples)
    horse_race.csv                  ← Horse race: GPRA vs GPRT
    granger_causality.csv           ← Granger causality: GPR → gold direction
    garch_results.txt               ← GARCH-M volatility channel output
    dcc_garch_summary.txt           ← DCC-GARCH summary (correlations, hedge ratios)
    dcc_correlations.csv            ← Time-varying conditional correlation (rho_t)
    hedge_ratios_portfolio_weights.csv  ← Hedge ratios and optimal gold portfolio weights
    dcc_correlation_plot.png        ← DCC correlation plot with war episode markers
    robustness_checks.csv           ← RC1-RC3 results (Script 06, see below)

═══════════════════════════════════════════════════════════════════════════════
ROBUSTNESS CHECKS (Script 06)
═══════════════════════════════════════════════════════════════════════════════

  Three pre-specified checks on the baseline war-vs-control OLS result.
  Each is chosen on a standalone rationale, independent of what it does to
  the baseline's sign or significance — none is chosen after the fact to
  preserve a particular result, and results are reported as estimated:

  RC1 — Distribution-based threshold: top vs. bottom tercile of the GPR
        distribution, instead of the round-number 120/80 cutoffs.

  RC2 — Contemporaneous dependent variable: R_GOLD_t instead of
        R_GOLD_(t+1), to check whether any relationship is predictive
        or same-month.

  RC3 — Leave-one-episode-out: Gulf War, 9/11, and Russia-Ukraine are
        each excluded in turn (all three, not just one) to check whether
        the war-sample result depends on any single episode.

  The script does not label its own output "robust" or "not robust" —
  read the coefficients, standard errors, and p-values in
  results/robustness_checks.csv directly.

═══════════════════════════════════════════════════════════════════════════════
DATA SOURCES
═══════════════════════════════════════════════════════════════════════════════

  Gold price    : GitHub public dataset (datasets/gold-prices)
                  Original source: World Gold Council / ICE Benchmark Administration
                  (FRED removed gold price data permanently in January 2022)
                  Data available from 1833 — covers full 1991–2024 sample.

  S&P 500       : Yahoo Finance via yfinance (ticker: ^GSPC)
                  (FRED's SP500 series only goes back to 2016)
                  Yahoo Finance covers full 1991–2024 sample.

  VIX           : FRED series VIXCLS (CBOE Volatility Index)
                  Available from 1990 — covers full 1991–2024 sample.

  GPR index     : Caldara & Iacoviello (2022), matteoiacoviello.com
                  Available from 1985 — covers full 1991–2024 sample.
  GPRA sub-index: Caldara & Iacoviello (2022) — geopolitical acts (Categories 6–8)
  GPRT sub-index: Caldara & Iacoviello (2022) — geopolitical threats (Categories 1–5)

═══════════════════════════════════════════════════════════════════════════════
KEY PAPER REFERENCES
═══════════════════════════════════════════════════════════════════════════════

  Caldara, D., & Iacoviello, M. (2022). Measuring geopolitical risk.
    American Economic Review, 112(4), 1194–1225.

  Triki, M.B., & Ben Maatoug, A. (2021). The GOLD market as a safe haven
    against the stock market uncertainty: Evidence from geopolitical risk.
    Resources Policy, 70, 101872.

═══════════════════════════════════════════════════════════════════════════════
PYTHON PACKAGES
═══════════════════════════════════════════════════════════════════════════════

  fredapi      — FRED API access
  yfinance     — Yahoo Finance (S&P 500 history)
  pandas       — data manipulation
  numpy        — numerical computation
  statsmodels  — OLS, VAR, unit root tests, Newey-West standard errors
  arch         — GARCH / DCC-GARCH models (Kevin Sheppard, Oxford)
  scipy        — optimisation, statistics
  matplotlib   — plots
  requests     — GPR file download
  xlrd         — reading .xls files
  openpyxl     — reading .xlsx files

═══════════════════════════════════════════════════════════════════════════════
