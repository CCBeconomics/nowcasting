"""
config.py  --  Trinidad and Tobago workshop settings (participant copy).

Modelled on 2026_NWCST_BOL_Workshop/code/config.py (2026-10-19), cut down to
the NTL and Google Trends blocks. Edit THIS file, not the notebooks.
"""
import os

# NASA LAADS DAAC download token. Expires after ~60 days.
# Regenerate at https://ladsweb.modaps.eosdis.nasa.gov/ -> My Account -> Generate Token
NASA_TOKEN = ""   # paste your own token here (see the install guide)

# ============================================================================
#  COUNTRY
# ============================================================================
COUNTRY   = "Trinidad and Tobago"
CODE      = "TT"                    # ISO2, also the Google Trends geo
CODE_LC   = CODE.lower()
GADM_ISO3 = "TTO"

# ---- Night-time light tiles (VIIRS Black Marble) ---------------------------
TILES = ["h11v07"]
DOWNLOAD_TILES = None               # subset of TILES to download; None = all

# ---- File names (all in raw/) ----------------------------------------------
COORDINATES_XLSX = "Coordinates.xlsx"   # NTL sites (lat/lon)
NTL_OUTPUT_CSV   = "ntl_data.csv"       # daily NTL per site
NTL_SHAPE_CSV    = "ntl_shape.csv"      # daily NTL, country mean
# meta.csv registers the country series as shape_mean / shape_sum
# (UNTL0000 / UNTL0001), so the shape CSV keeps those names.
NTL_SHAPE_COLS   = {"ntl_mean": "shape_mean", "ntl_sum": "shape_sum"}

# ============================================================================
#  NIGHT-TIME LIGHTS (1_data_ntl)
# ============================================================================
# Window to download and extract. One file (~25 MB) per day.
#   "2026-04-01 2026-04-30"  -> from - to
#   "2026-04-01"             -> from that date to the latest available
#   None                     -> from NTL_FIRST_DATE to the latest available
# Dates already in raw/ntl_*.csv are never re-read, so a window that overlaps
# the extracted history costs nothing but the download of tiles not on disk.
# Workshop: two weeks = 14 files, ~350 MB. The full 2012-2026 history is
# shipped in data/ntl_shape.csv -- do not try to rebuild it on a laptop.
NTL_DATE_RANGE = "2026-09-01 2026-09-14"

# Safety limits, applied even when NTL_DATE_RANGE is None.
NTL_FIRST_DATE = "2012-01-19"       # first VNP46A2 date; never download anything earlier
NTL_MAX_FILES  = 30                 # refuse a run above this (~750 MB); None = no limit

# Event marked on the series and centred in the before/after maps.
NTL_EVENT_DATE   = None             # e.g. "2025-10-28"; None = first/last day of the window
NTL_EVENT_LABEL  = None             # e.g. "Hurricane Melissa"
NTL_EVENT_WINDOW = 7                # days before/after the event shown on the maps

# The maps are the one step that opens a tile after extraction. hfiles/ is
# Dropbox online-only, so each map date pulls its tile back down (~25 MB).
NTL_DRAW_MAPS = True


def ntl_window():
    """(start, end) of the NTL window as datetimes; end None = latest available."""
    from datetime import datetime

    floor = datetime.strptime(NTL_FIRST_DATE, "%Y-%m-%d") if NTL_FIRST_DATE else None
    if not NTL_DATE_RANGE:
        return floor, None

    parts = NTL_DATE_RANGE.split()
    if len(parts) == 1:
        start, end = datetime.strptime(parts[0], "%Y-%m-%d"), None
    elif len(parts) == 2:
        start = datetime.strptime(parts[0], "%Y-%m-%d")
        end   = datetime.strptime(parts[1], "%Y-%m-%d")
    else:
        raise ValueError("NTL_DATE_RANGE must be None, one date, or two dates "
                         "separated by a space; got %r" % (NTL_DATE_RANGE,))

    if floor and start < floor:
        start = floor
    return start, end

# ============================================================================
#  GOOGLE TRENDS (1_data_gtrends)
# ============================================================================
GT_KEYWORDS = "TT_GTrends_Keywords.csv"   # in raw/: keyword, geo, column
GT_START    = "2004-01-01"                # Google Trends starts here ("all" on the website)

# ============================================================================
#  PATHS AND HELPERS -- normally no need to edit below
# ============================================================================
GADM_ZIP     = "gadm41_" + GADM_ISO3 + "_shp.zip"
PROJECT_ROOT = ".."                 # public/, relative to this file
NL_ROOT      = None               # None = public/hfiles/, created on first download


def download_tiles():
    """Tiles to download: DOWNLOAD_TILES if set, else TILES."""
    return list(DOWNLOAD_TILES) if DOWNLOAD_TILES else list(TILES)


def derive_paths(d, make=True):
    """Project folders, given d = the folder holding this file. Creates missing ones."""
    d = os.path.abspath(d)
    root = os.path.abspath(os.path.join(d, PROJECT_ROOT))
    up = lambda *p: os.path.join(root, *p)
    P = {
        "code":    d,
        "raw":     up("raw"),
        "data":    up("data"),
        "output":  up("output"),
        "ntl_tmp": up("ntl_tmp"),
        "nl_root": os.path.abspath(os.path.join(d, NL_ROOT)) if NL_ROOT else up("hfiles"),
    }
    if make:
        # nl_root is never created here: a wrong NL_ROOT should fail loudly,
        # not start a fresh, empty tile store.
        for k, p in P.items():
            if k not in ("code", "nl_root"):
                os.makedirs(p, exist_ok=True)
    return P


# ============================================================================
# ==== NOWCAST (act5, act10–act12) ====
# ============================================================================
# Settings for the build, selection, model and nowcast activities. Adapted from
# the DC workshop config (2026-08) to Trinidad and Tobago.
from datetime import date

# ---- Files (all in data/) ---------------------------------------------------
DATA_CSV     = "data.csv"           # monthly panel, quarterly GDP on quarter-end months
METADATA_CSV = "meta.csv"           # one row per ticker: freq, months_lag, nominal, sa, agg_method
GDP_CSV      = "gdp.csv"            # quarterly real GDP alone (act5)
GTRENDS_CSV  = "gtrends.csv"        # Google Trends, monthly, in raw/ (made in act3, read by act5)
NTL_CSV      = "ntl_shape.csv"      # daily country-mean night lights (act5)

# ---- Key series --------------------------------------------------------------
GDP_TICKER    = "RGDP0000"   # real GDP, CBTT production approach (constant prices)
CPI_TICKER    = "RCPI0000"   # CPI all items, index Jan 2015 = 100
CPI_BASE_YEAR = 2018         # deflate nominal series to 2018 dollars (as in the research pipeline)

# ---- Sample and vintages ----------------------------------------------------
# GDP is published to 2026Q1 (2026-03-01). The live nowcast covers the next
# unpublished quarters up to DESIRED_DATE.
DESIRED_DATE = "2026-09-01"                      # last quarter to nowcast (2026Q3)
DATE_STR     = date.today().strftime("%Y%m%d")   # output/<DATE_STR>/ = today's run
TRAIN_START  = "2013-03-01"                      # first quarter with a growth rate
TEST_START   = "2023-06-01"                      # hold-out: 2023Q2 onward
LAGS         = list(range(-2, 3))                # vintages -2..+2 (two months early .. two late)
LARS_FACTORS = 1

# ---- Seasonal adjustment: 0 = classical seasonal_decompose, 1 = X-13 ---------
SEASONAL_FILTER = 0
X13_PATH        = None

# ---- Models and ensemble sizes (small, so act11 runs inside a session) --------
RUN = {"ols": 1, "olsr": 1, "enet": 1, "lasso": 1,
       "gbt": 1, "dt": 1, "rf": 1, "lstm": 0}   # LSTM off by default: slow on laptops
ITER       = 5
ITER_LSTM  = 5
EPOCH_LSTM = 50

# ---- Extensions (see the DC config for the full explanation) ----------------
SCALE_LINEAR     = 1
SHOCK_FEATURES   = 0      # built for hurricanes; T&T has none in the test window
ROLLING_LARS     = 0      # 1 = re-rank variables at every test quarter (slower)
SELECTION_CV     = 1
REGIME_ENSEMBLE  = 0      # needs SHOCK_DATES inside the test window
WATCHLIST        = 0
DISPERSION_BANDS = 0

SAMPLE_WEIGHT_DECAY = 0.95
NWCST_SEED          = 42
NWCST_FILL          = "ffill"
RIDGE_ALPHAS        = [0.1, 1, 10, 50, 100]
MAX_WORKERS         = 4

SHOCK_WINSOR_Z   = 3.0
SHOCK_BREAK_Z    = -3.0
SHOCK_MIN_BROKEN = 2

ROLLING_MAX_VARS = 99
CV_MIN_TRAIN     = 16
CV_STEP          = 2
CV_EVAL_SIZE     = 4
LARS_SIZE_RANGE  = (5, 30)

LOO_SHOCK_DATES   = ["2020-06-01"]   # COVID, inside the training window
SHOCK_DATES       = []
SHOCK_TARGET_DATE = None

ORACLE_KS              = []
ORACLE_ESTIMATORS      = ["ENET", "LASSO"]
ORACLE_VINTAGE         = 1
ORACLE_KS_TREE         = []
ORACLE_TREE_ESTIMATORS = ["DT", "RF", "GBT"]
ORACLE_TREE_VINTAGE    = 2

