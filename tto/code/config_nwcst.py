"""
config.py  --  Trinidad and Tobago production pipeline (build of 2026-10-22).
Every knob of 3_nwcst_tto.ipynb lives here. Sections 1-2 were fixed in steps 1-2 of the build;
sections 3+ (specifications, estimators, regime, watchlist, run) are filled in step 4.
"""

# ============================================================================
#  1. DATA
# ============================================================================
COUNTRY, CODE   = "Trinidad and Tobago", "TT"
DATA_CSV        = "data.csv"              # data/data.csv: the October 2026 panel, 638 series, rebuilt by 2_data_build (see README_pipeline.md)
METADATA_CSV    = "meta.csv"              # data/meta.csv: publication lags, nominal/sa flags, aggregation
TARGET          = "RGDP0000"              # real GDP, production approach, not SA at source: classical SA in the pipeline
CPI_TICKER      = "RCPI0000"
CPI_BASE_YEAR   = 2018
TRAIN_START     = "2013-03-01"            # first quarter with a growth rate
TEST_START      = "2023-06-01"            # production run; the identification run uses "2019-12-01" (COVID 2020-Q2 as a test quarter)
TEST_START_IDENTIFICATION = "2019-12-01"
DESIRED_DATE    = "2026-09-01"            # readings through the live quarter 2026-Q3
LAGS            = [-2, -1, 0, 1, 2, 3, 4, 5, 6]   # vintages: the print is scheduled at V+6 (user decision 2026-10-25; was -2..2)
MAX_PUB_LAG     = 8                       # months; Jamaica uses 5. User decision 2026-10-22: energy (6-8), BOP (7), labour (7) stay in
MIN_OBS         = 24
MAX_MISSING_TRAIN = 0.75
LAST_OBS_MIN    = "2024-01-01"

# ============================================================================
#  2. REDUNDANCY (applied by ticket at the pre-selection cell, before the rankings)
#     A0 degenerate and A duplicates are computed at run time (rules, no lists):
#       A0: constant over the last DEGENERATE_WINDOW observations or fewer than DEGENERATE_MIN_DISTINCT distinct values
#       A : identical values on the overlap (first ticket kept)
# ============================================================================
DEGENERATE_WINDOW       = 36
DEGENERATE_MIN_DISTINCT = 10
CORR_PRUNE  = 0.8      # within-ranking correlation prune (Jamaica rule 1b, on growth rates)
MAX_ABS_Z   = 20       # data-quality guard on standardised growth (Jamaica)
SMALL_SHARE = 0.2     # rule 1c: a component below this share of its aggregate is dropped

# B. mechanical derivations of other series in the panel (ticket: what it is)
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

# C. additive identities verified parts / aggregate = 1 on the October 2026 levels (aggregate: components)
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

# D. signed net balances: differences of gross flows already in the panel, change sign in sample -> dropped.
#    The goods and services balances (XBOP1110, XBOP1120) stay on one side of zero and are kept.
NET_BALANCES = ['XBOP1000', 'XBOP1100', 'XBOP1200', 'XBOP1300', 'XBOP2000', 'XBOP3000', 'XBOP3100', 'XBOP3200', 'XBOP3300', 'XBOP3400', 'XBOP4000', 'XBOP5000']

# E. by decision (ticket: reason), logged in the run parameters
EXCLUDED_BY_DECISION = {
    'UBML0000': 'Black Marble composite ends February 2026 (lag 0 in metadata): stale by 8 months at the data cut; refresh before the production run',
    # Index of economic activity, whole block (user decision 2026-10-24 after the first dry run): the CBTT index is a near-identity
    # of the target and the race scores groups at contemporaneous values, so it dominated every winning screen (7 of 16 variables
    # in the best group). Excluded as the target's own proxy; see README_pipeline.md.
    'RIEA1000': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1100': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1110': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1120': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1130': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1140': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1150': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1200': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1210': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1220': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1230': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1231': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1240': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1245': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1246': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1250': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1260': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1270': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1280': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1290': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1300': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1310': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1320': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1330': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1340': 'index of economic activity (CBTT), excluded as a proxy of the target',
    'RIEA1350': 'index of economic activity (CBTT), excluded as a proxy of the target',
}

# ============================================================================
#  3. ECONOMIST GROUPS (step 3, approved 2026-10-22). Indicator-based hypotheses about transmission;
#     the index of economic activity is left to the race. Design as in Jamaica: PCA 3, gdp_lags 2,
#     data_flat 3, all estimators.
# ============================================================================
HUMAN_GROUPS = [
    {"spec": "human1", "label": "energy",
     "selected_vars": "RENE5005 RENE5004 RENE5002 RENE5015 RENE5008 RENE5007 XBRT0001 UNTL0003 GCGR1410 XBOP1112",
     #                 LNG      gas      oil      methanol ammonia  rig days Brent    NTL oil  energy tax energy exports
     "PCA": 3, "gdp_lags": 2, "data_flat": 3, "estimators": "ols olsr enet lasso dt rf gbt lstm"},
    {"spec": "human2", "label": "activity",
     "selected_vars": "RCEM1100 RVEH2200 RPAY3010 RPAY5006 MCRD5000 MCRD3000 MDEP1000 GCGR1510 XUST0002 XEUT0002 UNTL1000 UNTL0005 UGTR0000 UGTR1026",
     #                 cement   vehicles ccard$   ATM$     priv cr  cons cr  deposits nonen tax US imp  EU imp   NTL sites NTL malls Trends   PriceSmart
     "PCA": 3, "gdp_lags": 2, "data_flat": 3, "estimators": "ols olsr enet lasso dt rf gbt lstm"},
]

# Frozen groups (user decision 2026-10-25): the groups behind the four watchlist clusters, taken from the identification
# run of 24 October 2026 (output/20261024_tto_ident/rolling_selections.pkl) at the ranking in force at COVID, i.e. the
# SIS and SHAP rankings on data before 2020-Q2 (key 2020-04-01). The race groups are rolling (re-ranked every test
# quarter); these are static and estimated in every later run beside the race, in the five designs, so the clusters do
# not depend on the sizes the race happens to pick. Prefix "f" keeps them apart from the race's own groups. Tickets
# missing from a later panel are dropped with a message. Alternative not used: rolling rows forced to these sizes.
FROZEN_GROUPS = [
    # "augment": False (default) = the race design, base tickets only, so the frozen cells reproduce the identified
    # race cells; True = the static design with N/R companions and SHOK0000 (the 2026-10-25 convention).
    # 2026-10-26: X-13 target adopted; groups re-frozen from the X-13 identification run (output/20261026_tto_ident,
    # rolling_selections.pkl, key 2020-04-01 = data before 2020-Q2). fsis17 / fsis24 (classical) retired: blind at COVID
    # on the X-13 pool (-1.0..-4.0) and the confirmation cluster raised a false alert at 2020-Q3 V+2.
    # 2026-10-27 (user decision): fsis40 (SIS top 40, candidate early reader) removed: -5.1 at COVID but -6.6 in 2020-Q4.
    {"spec": "fibma6",  "method": "fibma", "K": 6,  "note": "iBMA top 6 at 2020-Q2 (zero cluster, V0)",
     "selected_vars": "UGTR1001 XBOP1115 MCRD7000 MRES0002 XBOP1220 RPAY5006"},
    {"spec": "fshap6",  "method": "fshap", "K": 6,  "note": "SHAP top 6 at 2020-Q2 (support V+1, confirmation V+2, late V+4)",
     "selected_vars": "RPAY5002 UGTR1001 MLOA1303 MDEP1800 MCRD5000 XFRED0007"},
    {"spec": "fshap20", "method": "fshap", "K": 20, "augment": True, "note": "SHAP top 20 at 2020-Q2 of the CLASSICAL identification (early cluster, raw designs); augmented static design (N/R companions + SHOK0000), as measured on the X-13 identification pool",
     "selected_vars": "MCRD5000 XBOP3420 RENE5003 MRES0002 UGTR1001 UGTR1008 RENE5020 UNTL3001 GCGR1110 FTFM1100 MNPL1300 RLAB2000 UGTR1028 XFRED0001 XBOP1112 XFRED0007 MRAT4001 XFRED0002 GCGR1115 RPAY6000"},
]

# ============================================================================
#  5. PRE-SELECTION (Jamaica beta16 values; the typed redundancy lists are in section 2)
# ============================================================================
FACTORS            = 1       # PCA factors in the race evaluator (Bai-Ng IC1 selects 1 on these sample sizes)
SHOCK_SHARE        = 0.05    # SHOK0000 fires when >= this share of the monthly qmin variables (floor 4) break -3 sigma
FIXED_SIZE_MAX     = 99      # largest group size scored
MAX_ROLLING_VARS   = 99
GDP_LAGS_SCREEN    = 2       # GDP lags in the t-stat / OCMT screening regressions
HAC_LAGS           = 4       # Newey-West window
RANK_ON_TRAIN_ONLY = False   # static ranking on the full quarterly panel (beta14b convention)
CV_MIN_TRAIN, CV_STEP, CV_EVAL_SIZE = 16, 2, 4   # expanding-window folds of the race
CHOSEN_METHODS       = None  # None = every screen with at least one benchmark-beating group
MAX_SPECS_PER_METHOD = 5     # a screen with more beating groups is thinned by the spread pick (user decision 2026-10-24: 5, was 10; six screens win on the clean panel)
KEEP_COMMON          = True  # the common group is a fixed row (watchlist reference)
K_REF                = 60    # size at which the overlap matrix between screens is reported

# ============================================================================
#  6. TRAINING LOOP (kit knobs at Jamaica's values)
# ============================================================================
ITER_LSTM, EPOCH_LSTM = 20, 100     # networks per cell and epochs (Jamaica 2026-10-20)
SAMPLE_WEIGHT_DECAY   = 0.95        # a quarter a quarters old weighs 0.95**a
NWCST_FILL            = "ffill"     # ragged-edge fill
SCALE_LINEAR          = 1           # StandardScaler around ridge (elastic net and LASSO always scale)
RIDGE_ALPHAS          = [0.1, 1, 10, 50, 100]
SHOCK_WINSOR_Z        = 3.0         # winsorise shock features at +/- 3 sigma for the linear models
LOO_SHOCK_DATES       = ["2020-06-01"]   # leave-one-out refit on COVID: seeds the shock record

# ============================================================================
#  7. REGIME AND WATCHLIST
# ============================================================================
READING_GUARD          = 20.0       # pp; a cell reading beyond +/- this is dropped from the pool for that quarter and
                                    # vintage (user decision 2026-10-25: 91 of 956 production cells, all linear on raw
                                    # designs, gave readings of up to 859 that the weighted mean followed; no Trinidad
                                    # quarter has exceeded 12.3 in absolute value)
BURN_IN_MIN_PRINTS     = 3          # equal weights until three published prints are on record
REGIME_OVERRIDE_ADD    = {}         # {"YYYY-MM-01": "reason"}; every use is logged
REGIME_OVERRIDE_REMOVE = {}
# Breadth rule (user decision 2026-10-25): on the clean panel the 5 % count never reached its threshold at COVID (max 10 of
# 295 at V+2) while 2022-Q4 reached 13, so it cannot trigger the regime on its own. It stays as the corroboration of a
# cluster alert (>= MIN_BROKEN_FOR_ALERT broken variables); set to True to restore the Jamaica "either" rule.
BREADTH_TRIGGER = False
# Alert rule (2026-10-26 night, PENDING USER CONFIRMATION in the morning): the clusters were identified as crash readers
# (median <= -4 at COVID), and every alert they raised on the other side was a rebound or a false one (identification pool:
# 2020-Q3 V+2/V+4 and 2020-Q4 V+1 at +4.2..+4.6; production: 2023-Q4 V+1/V+2/V+4 at +4.5..+5.5 with actual -0.1), so the
# alarm is one-sided. "both" restores the two-sided rule used until 2026-10-26.
ALERT_PP   = 4.0
ALERT_SIDE = "below"     # side that can switch the regime: "below" (crash alarm) or "both"
FLAG_BOTH  = True        # 2026-10-27 (user decision): a divergence of ALERT_PP on either side is flagged on the
                         # watchlist (column "flag": -1 below, +1 above, 0 inside the corridor); only the ALERT_SIDE
                         # side becomes an alert and can switch the regime. Upward flags are information only.

# Clusters frozen 2026-10-26 on the X-13 identification run (output/20261026_tto_ident, COVID 2020-Q2 as the test
# shock, actual -12.3, pooled median -0.2..-0.6 at every vintage; search `_build/cluster_search.py`). Criterion as in
# Jamaica: cluster median <= -4 at the shock, never more than 4 pp from the pooled median in the 12 ordinary test
# quarters. The clusters point at the FROZEN groups (section 3), not at the race's own groups, whose membership rolls.
# The alert rule in [tto:post] is cluster median - pooled median <= -ALERT_PP (ALERT_SIDE), backed by >= MIN_BROKEN broken variables.
# An entry with "alerts": False is drawn on the panel but never raises an alert (the reference line).
# The 2026-10-25 set (fsis17/fsis24/fshap20 on the classical target) is in _build/config.py.bak-20261026-185204.
_LIN = ["OLSR", "ENET", "LASSO"]
_FAC = ["p3", "p5", "p5l2"]
WATCHLIST = {
    "early_vm1":   {"vintage": -1, "methods": ["fshap"], "variants": ["base", "flats"], "estimators": _LIN, "K": [20, 20],
                    "note": "6 cells; COVID -4.5 at V-1, divergence -3.9 (NEAR MISS of the 4 pp alert); max ordinary divergence 2.8 (2020-Q4)"},
    "zero_v0":     {"vintage": 0,  "methods": ["fibma"], "variants": ["base", "flats"], "estimators": _LIN, "K": [6, 6],
                    "note": "6 cells; COVID -15.5 at V0, divergence -15.1; max ordinary divergence 3.3 (2021-Q1)"},
    "support_vp1": {"vintage": 1,  "methods": ["fshap"], "variants": _FAC, "estimators": ["OLSR"], "K": [6, 6],
                    "note": "3 cells; COVID -6.7 at V+1, divergence -6.4; max ordinary divergence 3.1 (2020-Q3)"},
    "confirm_vp2": {"vintage": 2,  "methods": ["fshap"], "variants": _FAC, "estimators": _LIN, "K": [6, 6],
                    "note": "9 cells; COVID -6.6 at V+2, divergence -6.3; max ordinary divergence 3.0 (2020-Q3)"},
    "late_vp4":    {"vintage": 4,  "methods": ["fshap"], "variants": _FAC, "estimators": ["ENET", "LASSO"], "K": [6, 6],
                    "note": "6 cells; COVID -6.6 at V+4, divergence -6.2; max ordinary divergence 3.5 (2022-Q2)"},
    "reference":   {"vintage": None, "methods": ["human"], "variants": ["base"], "estimators": None, "K": [2, 2], "alerts": False,
                    "note": "economist activity group human2 (7 cells); COVID -0.4..-0.8; the reference line, never alerts"},
}

# ============================================================================
#  4. RUN SETTINGS (approved with cells 1-2, 2026-10-24)
# ============================================================================
import os as _os, json as _json
from datetime import date as _date
DATE_STR = _os.environ.get("TTO_LAUNCH_DATE_STR") or _date.today().strftime("%Y%m%d")   # the orchestrator locks it
RUN_TAG  = _os.environ.get("TTO_RUN_TAG", "prod")        # "ident" = identification run (COVID as a test quarter)
OUT_SUFFIX = _os.environ.get("TTO_OUT_SUFFIX", "")       # variant runs (2026-10-25): "_x13", "_pseudo", "_chain", "_frz"
OUTPUT_SUBFOLDER = f"{DATE_STR}_tto_{RUN_TAG}{OUT_SUFFIX}"   # output/<date>_tto_prod[_variant] or _ident

# ---- variant switches (2026-10-25), read from the environment so the overnight queue flips them without editing this file ----
SA_METHOD    = _os.environ.get("TTO_SA_METHOD", "x13")           # "x13": X-13ARIMA-SEATS on every non-SA series, target included (Barbados beta15 block);
                                                                 # ADOPTED 2026-10-26 (user decision) after the overnight comparison: RMSE 0.88 of the target's
                                                                 # swing vs 0.91 classical, 0.54 of no-change vs 0.59, better 95 % band coverage.
                                                                 # "classical": statsmodels additive decomposition (the method until 25 October 2026)
GDP_LAG_MODE = _os.environ.get("TTO_GDP_LAG_MODE", "vintage")    # GDP-lag features of the p5l2 and economist designs (helper gdp_lag_features):
                                                                 # "vintage": lag l = the l-th most recent PUBLISHED quarter at the reading (training: two quarters back)
                                                                 # "chain":   lag l = quarter t-l: the print if published, else the pseudo-print, else the pipeline's own
                                                                 #            nowcast of t-l at its latest vintage on or before the reading date (CHAIN_SOURCE_RUN)
CHAIN_SOURCE_RUN = _os.environ.get("TTO_CHAIN_SOURCE_RUN")       # output folder of the base run whose regime-aware nowcasts feed the chain, e.g. "20261026_tto_prod"
PSEUDO_PRINTS = {k: float(v) for k, v in _json.loads(_os.environ.get("TTO_PSEUDO_PRINTS", "{}")).items()}
                                                                 # {"2026-06-01": value}: written into the target before the seasonal adjustment and treated as
                                                                 # published by the lag features; EXCLUDED from the test window (so never scored or on the record)

# the two runs differ only here
TEST_START_ACTIVE   = TEST_START_IDENTIFICATION if RUN_TAG == "ident" else TEST_START
TEST_END_ACTIVE     = "2022-12-01" if RUN_TAG == "ident" else None     # None = last GDP print
DESIRED_DATE_ACTIVE = None if RUN_TAG == "ident" else DESIRED_DATE     # no live readings in the identification run

RUN  = {"ols": 0, "olsr": 1, "enet": 1, "lasso": 1, "dt": 0, "dt4": 1, "rf": 1, "gbt": 1, "lstm": 1}   # dt off: DT4 replaces DT (user, 2026-10-25)
ITER = 10                      # bag size of DT / DT4 / RF / GBT (Jamaica)
NWCST_SEED, SEEDS = 42, [42, 43, 44]
MAX_WORKERS, SAVE_BATCH_SIZE = 4, 25

PROJECT_ROOT = ".."            # tto/, relative to code/
def derive_paths(d, make=True):
    """Project folders given d = the folder holding this file (kit function, lights folders removed)."""
    root = _os.path.abspath(_os.path.join(_os.path.abspath(d), PROJECT_ROOT))
    P = {"code": _os.path.abspath(d), "raw": _os.path.join(root, "raw"), "data": _os.path.join(root, "data"),
         "output": _os.path.join(root, "output")}
    if make:
        for k, p in P.items():
            if k != "code": _os.makedirs(p, exist_ok=True)
    return P
