"""
config.py  --  Trinidad and Tobago GDP nowcasting pipeline: every knob of 3_nwcst_tto.ipynb.

Edit this file, not the notebook. Each section says which notebook cell reads it (the cells are tagged
# [tto:<name>]). Two runs share the file and differ only in section 7:
  TTO_RUN_TAG=ident   identification run: test window from 2019-Q4 to 2022-Q4, COVID 2020-Q2 as a test quarter,
                      no live readings; used once to identify the watchlist clusters
  TTO_RUN_TAG=prod    production run: test window from 2023-Q2 to the last GDP print, live quarters after it
Environment variables (TTO_RUN_TAG, TTO_LAUNCH_DATE_STR, TTO_SA_METHOD) override the defaults so the orchestrator
can launch several seeds without editing the file.
"""
import os as _os
from datetime import date as _date

# ============================================================================
#  1. DATA, TARGET AND SAMPLE          read by [tto:data] and [tto:filters]
# ============================================================================
COUNTRY         = "Trinidad and Tobago"
DATA_CSV        = "data.csv"              # data/data.csv: monthly panel, 638 series, Jan 2012 to Oct 2026, rebuilt by code/2_data_build
METADATA_CSV    = "meta.csv"              # data/meta.csv: tickets, publication delays (months_lag), SA flags
TARGET          = "RGDP0000"              # real GDP, production approach, quarterly; not SA at source (adjusted in the pipeline)

# Nominal series are deflated by the CPI before the growth rates are taken; the CPI itself is then dropped from the
# candidate panel (filters cell, since 2026-11-09). Runs before that date kept its growth rate as a candidate.
CPI_TICKER      = "RCPI0000"
CPI_BASE_YEAR   = 2018                    # the CPI is rebased to its mean of this year

# Sample windows (first day of the quarter's last month)
TRAIN_START               = "2013-03-01"  # first quarter with a growth rate
TEST_START                = "2023-06-01"  # production run: first test quarter
TEST_START_IDENTIFICATION = "2019-12-01"  # identification run: first test quarter (COVID 2020-Q2 is the second)
TEST_END_IDENTIFICATION   = "2022-12-01"  # identification run: last test quarter (the production run stops at the last print)
DESIRED_DATE              = "2026-09-01"  # production run: live readings through this quarter (2026-Q3)

# Vintages: months relative to the end of the target quarter at which the nowcast is read
LAGS            = [-2, -1, 0, 1, 2, 3, 4, 5, 6]   # V-2 to V+6; the reported number is the V+6 reading

# Panel filters
MAX_PUB_LAG       = 8                     # months; a series published later than this is dropped (energy 6-8, BOP 7, labour 7 kept)
MIN_OBS             = 24                  # minimum observations for a series to enter (whole sample)
MAX_MISSING_TRAIN   = 0.75                # maximum share of missing months in the training window (monthly series)
MIN_TRAIN_HISTORY_Q = 8                   # real observations required inside the run's own training window: 8 quarters for a
                                          # quarterly series, 24 months otherwise; a series born later is out of that run
                                          # (each run uses the series with two years of history before its test start)
LAST_OBS_MIN      = "2024-01-01"          # a series whose last observation is before this year is dropped (stale series)

# ============================================================================
#  2. REDUNDANCY                       read by [tto:filters] and [tto:preselection]
#  Rules computed at run time:
#    degenerate: constant over the last DEGENERATE_WINDOW observations, or fewer than DEGENERATE_MIN_DISTINCT distinct values
#    duplicates: identical values on the overlap (the first ticket is kept)
#    correlation prune: within each ranking, a variable is kept only if |corr| < CORR_PRUNE with every variable kept before it
#  Typed lists below: mechanical derivations, additive identities, signed balances, exclusions by decision.
# ============================================================================
DEGENERATE_WINDOW       = 36
DEGENERATE_MIN_DISTINCT = 10
CORR_PRUNE  = 0.8                         # on growth rates (rule 1b)
MAX_ABS_Z   = 20                          # a series with a standardised growth beyond this against the training window is dropped
SMALL_SHARE = 0.2                         # rule 1c: a component below this share of its aggregate is dropped

# Mechanical derivations of other series in the panel (ticket: what it is); dropped, the source stays
MECHANICAL = {
    'UNTL3000': '30-day window of UNTL1000',
    'UNTL3002': '30-day window of UNTL0002',
    'UNTL3003': '30-day window of UNTL0003',
    'UNTL3004': '30-day window of UNTL0004',
    'UNTL3005': '30-day window of UNTL0005',
    'UNTL3006': '30-day window of UNTL0006',
    'MCRD5001': 'year-on-year growth of MCRD5000',
    'MCRD1001': 'year-on-year growth of MCRD1000',
    'MCRD3001': 'year-on-year growth of MCRD3000',
    'MCRD7001': 'year-on-year growth of MCRD7000',
    'FDEB2250': 'ratio to GDP (contains the target)',
    'FDEB2260': 'ratio to GDP (contains the target)',
    'FDEB2270': 'ratio to GDP (contains the target)',
    'FDEB2280': 'ratio to GDP (contains the target)',
    'MRES1000': 'import cover ratio = reserves / imports',
}

# Additive identities, verified parts / aggregate = 1 on the October 2026 levels (aggregate: components);
# a component below SMALL_SHARE of its aggregate is dropped
IDENTITIES = {
    'FTFM2100': ['FTFM2101', 'FTFM2102', 'FTFM2103'],
    'FTFM2200': ['FTFM2202', 'FTFM2203'],
    'MDEP1100': ['MDEP1110', 'MDEP1120'],
    'MLOA1700': ['MLOA1701', 'MLOA1703', 'MLOA1705'],
    'MLOA1100': ['MLOA1101', 'MLOA1102', 'MLOA1103', 'MLOA1104', 'MLOA1105'],
    'MLOA1300': ['MLOA1301', 'MLOA1302', 'MLOA1303', 'MLOA1304', 'MLOA1305'],
    'MLOA1400': ['MLOA1401', 'MLOA1402', 'MLOA1403', 'MLOA1404', 'MLOA1405'],
    'MLOA1500': ['MLOA1502', 'MLOA1503', 'MLOA1504', 'MLOA1505', 'MLOA1501'],
    'GCGO2200': ['GCGO2220', 'GCGO2210'],
    'GCGR1200': ['GCGR1210', 'GCGR1220'],
    'GCGR1300': ['GCGR1310', 'GCGR1320'],
    'GCGR1500': ['GCGR1510', 'GCGR1520'],
    'XBOP1111': ['XBOP1112', 'XBOP1113'],
    'XBOP1114': ['XBOP1115', 'XBOP1116'],
    'XBOP1310': ['XBOP1311', 'XBOP1312'],
    'XBOP1320': ['XBOP1321', 'XBOP1322'],
}

# Signed net balances: differences of gross flows already in the panel that change sign in sample; dropped.
# The goods and services balances (XBOP1110, XBOP1120) stay on one side of zero and are kept.
NET_BALANCES = ['XBOP1000', 'XBOP1100', 'XBOP1200', 'XBOP1300', 'XBOP2000', 'XBOP3000', 'XBOP3100', 'XBOP3200', 'XBOP3300', 'XBOP3400', 'XBOP4000', 'XBOP5000']

# Exclusions by decision (ticket: reason), logged in the run parameters
_RIEA = 'index of economic activity (CBTT), excluded as a proxy of the target'
EXCLUDED_BY_DECISION = {
    'UBML0000': 'Black Marble composite ends February 2026 (lag 0 in metadata): eight months stale at the data cut',
    # The CBTT index of economic activity is a near-identity of the target and the race scores groups at contemporaneous
    # values, so it dominated every winning screen (7 of 16 variables in the best group). Whole block excluded as the
    # target's own proxy.
    **{t: _RIEA for t in ['RIEA1000', 'RIEA1100', 'RIEA1110', 'RIEA1120', 'RIEA1130', 'RIEA1140', 'RIEA1150', 'RIEA1200',
                          'RIEA1210', 'RIEA1220', 'RIEA1230', 'RIEA1231', 'RIEA1240', 'RIEA1245', 'RIEA1246', 'RIEA1250',
                          'RIEA1260', 'RIEA1270', 'RIEA1280', 'RIEA1290', 'RIEA1300', 'RIEA1310', 'RIEA1320', 'RIEA1330',
                          'RIEA1340', 'RIEA1350']},
}

# ============================================================================
#  3. PRE-SELECTION RACE               read by [tto:preselection]
#  Six screens (LARS, SIS, t-stat, SHAP, OCMT, iBMA) rank the candidates; groups of every size are scored by
#  expanding-window cross-validation against the PCA-on-all benchmark; the screens with at least one
#  benchmark-beating group send their best groups to the estimation loop.
# ============================================================================
FACTORS            = 1       # PCA factors in the race evaluator (Bai-Ng IC1 selects 1 on these sample sizes)
FIXED_SIZE_MAX     = 99      # largest group size scored; 99 = no cap below the number of candidates
MAX_ROLLING_VARS   = 99      # largest rolling group carried into the loop; 99 = no cap
GDP_LAGS_SCREEN    = 2       # GDP lags in the t-stat and OCMT screening regressions
HAC_LAGS           = 4       # Newey-West window of the t-stat screen
RANK_ON_TRAIN_ONLY = {"prod": True, "ident": False}   # prod: static ranking on the training window (real time); ident: full panel
                                                      # (identification exercise; 28 quarters are too few to rank 347 candidates on).
                                                      # Rolling rankings are always computed before the test quarter in both runs.
CV_ON_TRAIN_ONLY   = {"prod": True, "ident": False}   # prod: no CV fold validates inside the test period (11 folds instead of 17);
                                                      # ident: full panel (bounding would drop 6 of 17 folds, the COVID ones included)
CV_MIN_TRAIN, CV_STEP, CV_EVAL_SIZE = 16, 2, 4   # expanding-window folds: first fold 16 quarters, step 2, 4 evaluation quarters
CV_MIN_REAL_OBS = 8      # a candidate enters a fold's design only with >= 8 real (not back-filled) quarters inside the training window
CV_CLIP_Z       = 5      # standardised designs in the race CV are clipped at +/- 5 sd; None = off
CHOSEN_METHODS       = None  # None = every screen with at least one benchmark-beating group; or a list of screen names
MAX_SPECS_PER_METHOD = 5     # a screen with more beating groups is thinned to this many by the spread pick (was 10)
KEEP_COMMON          = True  # the common group (variables in every winning screen's best group) is a fixed row
K_REF                = 60    # diagnostic only: group size at which the overlap matrix between screens is reported

# Shock indicator SHOK0000, built here and used as a feature by every design
SHOCK_SHARE        = 0.05    # fires when at least this share of the monthly indicators (floor 4) break -3 sigma in the quarter

# ============================================================================
#  4. GROUPS OUTSIDE THE RACE          read by [tto:preselection] and [tto:specs]
#  Economist groups: indicator-based hypotheses about transmission, estimated in every run beside the race
#  Frozen groups: the groups behind the watchlist clusters, frozen as identified on the identification run
#  (rankings on data before 2020-Q2) so the
#  clusters do not depend on the sizes the race happens to pick; the race's own groups are re-ranked every test
#  quarter. The prefix "f" keeps them apart from the race's groups. Tickets missing from a later panel are
#  dropped with a message.
# ============================================================================
_ALL_ESTIMATORS = "ols olsr enet lasso dt rf gbt lstm"
HUMAN_GROUPS = [
    {"spec": "human1", "label": "energy",
     "selected_vars": "RENE5005 RENE5004 RENE5002 RENE5015 RENE5008 RENE5007 XBRT0001 UNTL0003 GCGR1410 XBOP1112",
     #                 LNG      gas      oil      methanol ammonia  rig days Brent    NTL oil  energy tax energy exports
     "PCA": 3, "gdp_lags": 2, "data_flat": 3, "estimators": _ALL_ESTIMATORS},
    {"spec": "human2", "label": "activity",
     "selected_vars": "RCEM1100 RVEH2200 RPAY3010 RPAY5006 MCRD5000 MCRD3000 MDEP1000 GCGR1510 XUST0002 XEUT0002 UNTL1000 UNTL0005 UGTR0000 UGTR1026",
     #                 cement   vehicles ccard$   ATM$     priv cr  cons cr  deposits nonen tax US imp  EU imp   NTL sites NTL malls Trends   PriceSmart
     "PCA": 3, "gdp_lags": 2, "data_flat": 3, "estimators": _ALL_ESTIMATORS},
]

# "augment": False (default) = the race design, base tickets only, so the frozen cells reproduce the identified race cells;
#            True = the static design with the N/R companions and SHOK0000 (the design the early cluster was measured on)
FROZEN_GROUPS = [
    {"spec": "fibma6",  "method": "fibma", "K": 6,  "note": "iBMA top 6 at 2020-Q2 (zero cluster, V0)",
     "selected_vars": "UGTR1001 XBOP1115 MCRD7000 MRES0002 XBOP1220 RPAY5006"},
    {"spec": "fshap6",  "method": "fshap", "K": 6,  "note": "SHAP top 6 at 2020-Q2 (support V+1, confirmation V+2, late V+4)",
     "selected_vars": "RPAY5002 UGTR1001 MLOA1303 MDEP1800 MCRD5000 XFRED0007"},
    {"spec": "fshap20", "method": "fshap", "K": 20, "augment": True,
     "note": "SHAP top 20 at 2020-Q2 of the classical identification (early cluster, raw designs), augmented static design",
     "selected_vars": "MCRD5000 XBOP3420 RENE5003 MRES0002 UGTR1001 UGTR1008 RENE5020 UNTL3001 GCGR1110 FTFM1100 MNPL1300 RLAB2000 UGTR1028 XFRED0001 XBOP1112 XFRED0007 MRAT4001 XFRED0002 GCGR1115 RPAY6000"},
]

# ============================================================================
#  5. ESTIMATORS AND TRAINING LOOP     read by [tto:data] and [tto:loop]
# ============================================================================
RUN  = {"ols": 0, "olsr": 1, "enet": 1, "lasso": 1, "dt": 0, "dt4": 1, "rf": 1, "gbt": 1, "lstm": 1}
#       1 = estimated, 0 = off. OLS is off by design (unpenalised, unstable on these samples); DT4 (depth-4 tree) replaced DT
#       The keys stay so the loop's estimator map is complete.
ITER                  = 10          # bag size of DT4, RF and GBT
ITER_LSTM, EPOCH_LSTM = 20, 100     # networks per LSTM cell and epochs per network 
SAMPLE_WEIGHT_DECAY   = 0.95        # a quarter a quarters old weighs 0.95**a in every fit
NWCST_FILL            = "ffill"     # ragged-edge fill before estimation
SCALE_LINEAR          = 1           # StandardScaler around ridge (elastic net and LASSO always scale)
RIDGE_ALPHAS          = [0.1, 1, 10, 50, 100]   # ridge (OLSR) penalty grid, chosen by cross-validation
SHOCK_WINSOR_Z        = 3.0         # shock features winsorised at +/- 3 sigma for the linear models
LOO_SHOCK_DATES       = ["2020-06-01"]   # leave-one-out refit on these quarters seeds the shock record (COVID)

# ============================================================================
#  6. POOL, REGIME AND WATCHLIST       read by [tto:post] and [tto:live]
# ============================================================================
READING_GUARD      = 20.0   # pp; a cell reading beyond +/- this is dropped from the pool for that quarter and vintage
                            # (no Trinidad quarter has exceeded 12.3 in absolute value; linear cells on raw designs gave
                            # readings up to 859 before the guard)
BURN_IN_MIN_PRINTS = 3      # cells get equal weights until three published prints are on record
REGIME_OVERRIDE_ADD    = {} # analyst overrides of the automatic regime, {"YYYY-MM-01": "reason"}; every use is logged
REGIME_OVERRIDE_REMOVE = {}

# Regime switch. A watchlist cluster whose median is ALERT_PP below the pooled median raises an alert; the alert switches
# the regime when at least max(3, threshold // 3) indicators are broken (the breadth count corroborates). A divergence of
# ALERT_PP on either side is flagged on the watchlist (column "flag": -1 below, +1 above); only the ALERT_SIDE side can
# alert, because every upward alert on record was a rebound or a false one (identification 2020-Q3/Q4, production 2023-Q4).
# Upward flags are information only.
ALERT_PP        = 4.0
ALERT_SIDE      = "below"   # "below" (crash alarm, the rule in force) or "both" (the two-sided rule used before)
BREADTH_TRIGGER = False     # False = the breadth count never switches the regime on its own (it never reached its threshold at
                            # COVID, max 10 of 295, while 2022-Q4 reached 13); True = the "either" rule (breadth or alert)

# Watchlist clusters: sets of cells read at a fixed vintage, identified on the identification run
# (COVID 2020-Q2 actual -12.3, pooled median -0.2 to -0.6 at every vintage). Criterion: cluster median <= -4 at the shock and
# never more than 4 pp from the pooled median in the 12 ordinary test quarters. A cluster = frozen group (methods, K) x design
# variants x estimators at one vintage. "alerts": False draws the cluster on the panel without ever raising an alert.
_LIN = ["OLSR", "ENET", "LASSO"]
_FAC = ["p3", "p5", "p5l2"]
WATCHLIST = {
    "early_vm1":   {"vintage": -1, "methods": ["fshap"], "variants": ["base", "flats"], "estimators": _LIN, "K": [20, 20],
                    "note": "6 cells; COVID -4.5 at V-1, divergence -3.9 (a warning line, under the 4 pp alert); max ordinary divergence 2.8 (2020-Q4)"},
    "zero_v0":     {"vintage": 0,  "methods": ["fibma"], "variants": ["base", "flats"], "estimators": _LIN, "K": [6, 6],
                    "note": "6 cells; COVID -15.5 at V0, divergence -15.1; max ordinary divergence 3.3 (2021-Q1)"},
    "support_vp1": {"vintage": 1,  "methods": ["fshap"], "variants": _FAC, "estimators": ["OLSR"], "K": [6, 6],
                    "note": "3 cells; COVID -6.7 at V+1, divergence -6.4; max ordinary divergence 3.1 (2020-Q3)"},
    "confirm_vp2": {"vintage": 2,  "methods": ["fshap"], "variants": _FAC, "estimators": _LIN, "K": [6, 6],
                    "note": "9 cells; COVID -6.6 at V+2, divergence -6.3; max ordinary divergence 3.0 (2020-Q3)"},
    "late_vp4":    {"vintage": 4,  "methods": ["fshap"], "variants": _FAC, "estimators": ["ENET", "LASSO"], "K": [6, 6],
                    "note": "6 cells; COVID -6.6 at V+4, divergence -6.2; max ordinary divergence 3.5 (2022-Q2)"},
    "reference":   {"vintage": None, "methods": ["human"], "variants": ["base"], "estimators": None, "K": [2, 2], "alerts": False,
                    "note": "economist activity group human2 (7 cells); COVID -0.4 to -0.8; the reference line, never alerts"},
}

# ============================================================================
#  7. RUN SETTINGS                     read by [tto:setup], [tto:data], [tto:filters], run_tto.py and run_tto_seed.py
# ============================================================================
RUN_TAG  = _os.environ.get("TTO_RUN_TAG", "prod")                                        # "prod" or "ident" (see the header)
DATE_STR = _os.environ.get("TTO_LAUNCH_DATE_STR") or _date.today().strftime("%Y%m%d")    # output folder date; the orchestrator locks it
OUTPUT_SUBFOLDER = f"{DATE_STR}_tto_{RUN_TAG}"                                            # output/<date>_tto_prod or _ident
SA_METHOD = _os.environ.get("TTO_SA_METHOD", "x13")   # "x13": X-13ARIMA-SEATS on every non-SA series, target included (
                                                      #                                                      RMSE 0.88 of the target's swing vs 0.91, better 95 % band coverage);
                                                      # "classical": statsmodels additive decomposition, also the automatic fallback
                                                      # for a series X-13 rejects or when the binary is missing

# The two runs differ only here
TEST_START_ACTIVE   = TEST_START_IDENTIFICATION if RUN_TAG == "ident" else TEST_START
TEST_END_ACTIVE     = TEST_END_IDENTIFICATION   if RUN_TAG == "ident" else None     # None = last GDP print
DESIRED_DATE_ACTIVE = None if RUN_TAG == "ident" else DESIRED_DATE                  # no live readings in the identification run
RANK_ON_TRAIN_ONLY_ACTIVE = RANK_ON_TRAIN_ONLY[RUN_TAG]                             # section 3: static ranking window of this run
CV_ON_TRAIN_ONLY_ACTIVE   = CV_ON_TRAIN_ONLY[RUN_TAG]                               # section 3: CV fold window of this run

NWCST_SEED = 42                # seed of a single-seed run of the notebook
SEEDS      = [42, 43, 44]      # seeds the orchestrator runs in parallel and merges (run_tto.py)
MAX_WORKERS, SAVE_BATCH_SIZE = 4, 25   # parallel workers per seed, cells per checkpoint write

PROJECT_ROOT = ".."            # tto/ (raw/, data/, output/), relative to pipeline/
def derive_paths(d, make=True):
    """Project folders given d = the folder holding this file."""
    root = _os.path.abspath(_os.path.join(_os.path.abspath(d), PROJECT_ROOT))
    P = {"code": _os.path.abspath(d), "raw": _os.path.join(root, "raw"), "data": _os.path.join(root, "data"),
         "output": _os.path.join(root, "output")}
    if make:
        for k, p in P.items():
            if k != "code": _os.makedirs(p, exist_ok=True)
    return P
