"""
====================================================================
 /config.py  —  the key file you edit
====================================================================

Every notebook reads its country-specific values from THIS file.
The notebooks themselves are country-agnostic.

If you find yourself editing notebook logic to change a country, that
knob belongs here.

====================================================================
"""

import os
from datetime import date

# ============================================================================
#  1. IDENTITY  --  set the country.
# ============================================================================
# The Bahamas, Barbados, Jamaica, Belize, Trinidad and Tobago, Guyana, Suriname
COUNTRY = "Jamaica"

# ============================================================================
#  2. THE NIGHTTIME-LIGHTS WINDOW  (act3_ntl.ipynb)
# ============================================================================
NTL_DATE_RANGE = None

# Event to mark on the time series and to centre the before/after maps:
#   None -> use the first & last dates of the window
NTL_EVENT_DATE   = None
NTL_EVENT_LABEL  = ""
NTL_EVENT_WINDOW = 7      # days each side of the event for the before/after maps

# Download + analysis date window:
#   None                    -> incremental: fetch every missing .h5 up to the
#                              latest available. Years of data. Do NOT do this
#                              in a workshop session.
#   "YYYY-MM-DD YYYY-MM-DD" -> restrict BOTH download and analysis to
#                              [start, end], inclusive. Use this.
#
# A two-week window is about 14 files per tile: a couple of minutes to download
# and a few seconds to extract. Start small, widen once it works.

# ---- Worked examples: uncomment ONE block, with the matching COUNTRY ---------
# Hurricane Dorian, The Bahamas (this reproduces the figure in the Session 2 deck)
#   COUNTRY = "The Bahamas"
#   NTL_DATE_RANGE   = "2019-08-01 2019-10-15"
#   NTL_EVENT_DATE   = "2019-09-01"
#   NTL_EVENT_LABEL  = "H. Dorian"
#
# Hurricane Melissa, Jamaica
#   COUNTRY = "Jamaica"
#   NTL_DATE_RANGE   = "2025-10-01 2025-12-15"
#   NTL_EVENT_DATE   = "2025-10-28"
#   NTL_EVENT_LABEL  = "H. Melissa"
#
# Hurricane Beryl, Barbados
#   COUNTRY = "Barbados"
#   NTL_DATE_RANGE   = "2024-06-15 2024-08-15"
#   NTL_EVENT_DATE   = "2024-07-01"
#   NTL_EVENT_LABEL  = "H. Beryl"

# ============================================================================
#  3. NASA LAADS DAAC TOKEN  --  you must supply your own
# ============================================================================
# 1. Register (free) at https://urs.earthdata.nasa.gov
# 2. Go to https://ladsweb.modaps.eosdis.nasa.gov -> My Account -> Generate Token
# 3. Paste the token string below, between the quotes.
#
# Tokens EXPIRE (typically after 60 days). If downloads start returning 401,
# generate a fresh one. Do not share a token or commit it anywhere public --
# it authenticates as you.
NASA_TOKEN = ""

# ============================================================================
#  4. FILE NAMES
# ============================================================================
DATA_CSV     = "data.csv"                      # merged modelling table, in data/
METADATA_CSV = "meta.csv"                      # series metadata, in data/

# ---- Inputs act5_build merges ----------------------------------------------
GTRENDS_CSV = "gtrends.csv"      # act4's merged Google Trends            -> raw/
# act3 writes only the window you asked it for (NTL_DATE_RANGE), which is far
# too short for the monthly seasonal decomposition act5 runs. So act5 reads the
# full-history country series that ships with the workshop bundle instead.
NTL_BUILD_CSV = "ntl_shape.csv"  # daily NTL, full history (2012-)        -> data/

# The country-code-stamped file names (Coordinates_bs.xlsx, ntl_bs.csv, ...) are
# built from COUNTRY further down, once CODE_LC exists.

# ---- Key series ------------------------------------------------------------
GDP_TICKER    = "RGDP0000"   # real-GDP target (project-wide convention)
CPI_TICKER    = "RCPI0000"   # CPI used to DEFLATE nominal vars (JM=RCPI0000, BB=MCPI0000)
CPI_BASE_YEAR = 2019         # base year of that CPI index (differs by country)

# ============================================================================
#  5. NOWCAST RUN SETTINGS  (act6 - act9)
# ============================================================================
# act8_nwcst and act9_results MUST share DATE_STR -- it selects output/<DATE_STR>/.
# DATE_STR defaults to today, so act9 reads the run act8 just wrote. To go back
# and look at an older run, pin it here: DATE_STR = "20260801".
DESIRED_DATE = "2026-06-01"                      # target quarter start (YYYY-MM-01)
DATE_STR     = date.today().strftime("%Y%m%d")   # output/<DATE_STR>/ = today's run

TRAIN_START = "2015-03-01"          # training sample start
TEST_START  = "2023-06-01"          # test / holdout sample start
LAGS        = list(range(-2, 3))    # vintages evaluated: -2..+2 (backcast..forecast)

# PCA components used in the LARS pre-selection benchmark (act8, "factors").
LARS_FACTORS = 1

# ---- Seasonal adjustment (act8) --------------------------------------------
#   0 -> classical statsmodels seasonal_decompose: subtract the seasonal component.
#        Fast (in-process) and has no external dependency.
#   1 -> X-13ARIMA-SEATS, the US Census Bureau's production method. Better
#        real-time behaviour, but statsmodels only WRAPS it -- it shells out to an
#        external executable once per series, so it is far slower and it must be
#        installed. Series X-13 cannot handle fall back to the classical method,
#        and act8 prints how many series took each route so a fallback is never
#        silent.
#
# Cost, measured on this project's table (863 series): about 2 SECONDS for the
# classical route, about 7.5 MINUTES for X-13. That makes X-13 the single most
# expensive step in act8 -- more than the nowcasting loop itself. Worth it for a
# production run, usually not for a live workshop session.
SEASONAL_FILTER = 0
# Where the X-13 executable is.
#   None -> look for x13as / x13as_ascii on PATH. Installing the binary into the
#           environment (envs/nwcst/Library/bin/) puts it there whenever the
#           nwcst env is active, which is the normal setup.
#   "..." -> the FOLDER holding x13as.exe. See the warning below: a path to the
#            .exe itself does not work.
X13_PATH = None

# ----------------------------------------------------------------------------
#  INSTALLING X-13ARIMA-SEATS  (only needed for SEASONAL_FILTER = 1)
# ----------------------------------------------------------------------------
# It is NOT a conda or pip package on Windows -- statsmodels ships only the
# Python wrapper. You download the executable from the US Census Bureau and drop
# it into the environment. Already done on Diego's machine (2026-08-05, v1.1
# build 62); these are the steps for a new one.
#
# Windows, from an Anaconda Prompt with the nwcst env active:
#
#   $z = "$env:TEMP\x13as.zip"
#   Invoke-WebRequest -UseBasicParsing -OutFile $z `
#     -Uri "https://www2.census.gov/software/x-13arima-seats/x13as/windows/program-archives/x13as_ascii-v1-1-b62.zip"
#   Expand-Archive $z -DestinationPath "$env:TEMP\x13" -Force
#   $dest = "$env:CONDA_PREFIX\Library\bin"
#   Copy-Item "$env:TEMP\x13\x13as\x13as_ascii.exe" "$dest\x13as.exe"       -Force
#   Copy-Item "$env:TEMP\x13\x13as\x13as_ascii.exe" "$dest\x13as_ascii.exe" -Force
#
# macOS / Linux: same idea, but take the build from
# https://www2.census.gov/software/x-13arima-seats/x13as/unix-linux/ , chmod +x
# it, and copy it to $CONDA_PREFIX/bin/x13as .
#
# TWO TRAPS, both of which cost an afternoon here:
#
#  1. Install it under the name "x13as", not "x13as_ascii". The Census zip is
#     named x13as_ascii.exe, but statsmodels only ever probes for "x12a" and
#     "x13as". Copying it under both names, as above, satisfies either lookup.
#
#  2. If you set X13_PATH by hand, give the FOLDER, never the .exe. statsmodels
#     does try to strip a filename off the path, but only when it ends in a
#     lower-case binary name -- and Windows reports the file as "x13as.EXE",
#     which slips past that check and produces a nonsense lookup. Handing it the
#     file path yields X13NotFoundError on EVERY series while the binary sits
#     right there in the folder. Leaving X13_PATH = None avoids this entirely,
#     because act8 resolves the directory itself.
#
# To confirm it works, set SEASONAL_FILTER = 1 and read act8's line:
#     Seasonal adjustment: 682 series by X-13, 181 by classical fallback, ...
# A first number of 0 means it was not found, and every series was silently
# adjusted the classical way -- which is exactly the bug this reporting exists
# to make visible.
# ----------------------------------------------------------------------------

# Which models to fit (1 = on, 0 = off).
RUN = {
    "ols": 1, "olsr": 1, "enet": 1, "lasso": 1,
    "gbt": 1, "dt": 1, "rf": 1, "lstm": 1,
}

# Iterations / ensemble sizes. The workshop defaults are deliberately small so a
# full run finishes inside a session; the production pipeline uses 100.
ITER       = 10    # bagging ensemble size for DT / RF / GBT
ITER_LSTM  = 10    # LSTM ensemble size
EPOCH_LSTM = 50    # LSTM training episodes

# ============================================================================
#  6. BETA13/14 EXTENSIONS  (act8_nwcst)  --  every switch defaults to OFF.
# ============================================================================
# These are the research-branch methods developed on Jamaica in
# act8_nwcst_beta14a.ipynb (kept in activity/ as the reference copy). As shipped,
# act8_nwcst behaves EXACTLY as it did on 2026-08-01 -- a workshop run is
# unaffected. Switch them on one at a time; the commented values reproduce
# beta14a's Jamaica configuration.
#
# Turning any of these on changes the numbers act9_results reports. It also
# costs time: ROLLING_LARS and SELECTION_CV are the two expensive ones.

# ---- MASTER SWITCHES -------------------------------------------------------
# Every extension act8 can turn on, in one place. 0 = off, 1 = on. The tuning
# values for each live in the lettered groups below and stay inert while their
# switch is 0 -- so you can leave a group configured between runs and flip only
# the line here.
SCALE_LINEAR     = 0    # A  scale predictors before Ridge / ENET / LASSO
SHOCK_FEATURES   = 0    # B  quarterly-min "...N" / range "...R" series + SHOK0000
ROLLING_LARS     = 0    # C  re-rank the variable pool at every test quarter
SELECTION_CV     = 0    # C  expanding-window CV instead of one train/test split
REGIME_ENSEMBLE  = 0    # E  inverse-RMSE consensus, calibrated per regime
WATCHLIST        = 0    # F  model-spread and oracle-cluster diagnostic panels
DISPERSION_BANDS = 0    # F  weighted-quantile prediction bands

# Four more knobs act as switches but carry the setting in the value itself, so
# they stay down in their groups:
#     SAMPLE_WEIGHT_DECAY   (A)  None = off, else the decay factor   e.g. 0.95
#     NWCST_SEED            (A)  None = off, else the seed           e.g. 42
#     NWCST_FILL            (A)  "mean" = off, "ffill" = on
#     LOO_SHOCK_DATES       (D)  []     = off, else TRAINING-period shock quarters
#
# What depends on what. Each of these degrades with a printed message, never an
# error, so an unmet dependency shows up as an empty panel rather than a crash:
#     DISPERSION_BANDS   needs REGIME_ENSEMBLE = 1
#     REGIME_ENSEMBLE    needs a non-empty SHOCK_DATES  -- TEST-window quarters
#     WATCHLIST panel 3  needs a non-empty LOO_SHOCK_DATES
#     WATCHLIST panel 2  needs ORACLE_KS values that exist in YOUR spec list
#
# SHOCK_DATES and LOO_SHOCK_DATES are NOT interchangeable: leave-one-out refits a
# shock inside the TRAINING sample, the regime ensemble calibrates on shocks
# inside the TEST window. A date in the wrong list is silently inert.
# ----------------------------------------------------------------------------

# ---- A. Estimation core ----------------------------------------------------
# Exponential decay on the training sample: weight = decay ** (age in quarters),
# so the most recent quarter weighs 1 and older ones fade. None = unweighted.
SAMPLE_WEIGHT_DECAY = 0.95          # beta14a: 0.95
# Seed for DT / RF / GBT / LSTM. None keeps today's behaviour (each bagged member
# is drawn freshly, so two runs differ). An int makes a run reproducible.
NWCST_SEED          = 42          # beta14a: 42
# How ragged edges in the test vintage are filled.
#   "mean"  -> mean_fill_dataset  (training-sample mean; current)
#   "ffill" -> ffill_dataset      (carry the last observation forward; beta14a)
NWCST_FILL          = "ffill"        # beta14a: "ffill"
# Penalty grid for Ridge. With SCALE_LINEAR = 1 the predictors are standardised
# first, so the useful grid sits higher: on scaled data an alpha of 0.0001 is no
# penalty at all. Lower the floor again if you turn scaling back off.
RIDGE_ALPHAS        = [0.1, 1, 10, 50, 100]   # beta14a: [0.1, 1, 10, 50, 100] or [0.0001, 0.001, 0.01, 0.1, 1, 10, 20] 
# Parallel workers in the nowcasting loop. None -> derive from cpu_count.
MAX_WORKERS         = None          # beta14a: 4 (capped for memory)

# ---- B. Shock features -----------------------------------------------------
# For every LARS-selected monthly series X, add its quarterly MINimum (ticket
# "XN") and quarterly RANGE (ticket "XR"), plus a binary SHOK0000 flag for
# quarters where many series break down together. Built to let the models see
# hurricane / COVID collapses that a quarterly average smooths away.
# Switch: SHOCK_FEATURES, in the master block above.
SHOCK_WINSOR_Z   = 3.0              # clip shock columns at +/- Z sd, LINEAR models only
SHOCK_BREAK_Z    = -3.0             # a qmin series counts as "broken" below this z-score
# SHOCK_MIN_BROKEN must be no larger than the number of MONTHLY series LARS
# selects -- only those get a quarterly minimum, so a threshold above that count
# can never be met and SHOK0000 stays flat zero. The pre-selection cell prints
# "Computing shock features for N ... monthly variables"; keep this below N.
# Jamaica selected enough monthly series for 4; on the Barbados workshop table
# only 3 are monthly, so 4 flags nothing and 2 is the sensible value there.
SHOCK_MIN_BROKEN = 4                # this many broken series flag the quarter as a shock

# ---- C. Variable selection -------------------------------------------------
# ROLLING_LARS re-ranks the candidate pool at every test quarter using only data
# strictly before it, so the evaluation carries no look-ahead in the selection
# step either. SELECTION_CV replaces the single train/test split benchmark with
# an expanding-window cross-validation.
# Switches: ROLLING_LARS, SELECTION_CV, in the master block above.
ROLLING_MAX_VARS = 99
CV_MIN_TRAIN     = 16               # quarters in the smallest CV training window
CV_STEP          = 2                # quarters between fold origins
CV_EVAL_SIZE     = 4                # quarters scored per fold
LARS_SIZE_RANGE  = (10, 50)         # size sweep, single-split path. CV path: (5, K_max)

# ---- D/E. Shock calibration and the regime-aware ensemble -------------------
# LOO_SHOCK_DATES: quarters refitted leave-one-out, so the model's error on a
# known shock is a genuine out-of-sample residual. Written to
# "<date> 3_nwcst Tab shock_residuals.xlsx".
LOO_SHOCK_DATES   = ["2020-06-01"]              # beta14a: ["2020-06-01"]   (COVID)
# REGIME_ENSEMBLE (master block above) weights specs by inverse RMSE, computed
# separately for normal and shock quarters, and writes one consensus nowcast per
# (date, vintage). These are the quarters it treats as shocks -- they must fall
# inside the TEST window, or there is nothing in the forecast table to calibrate on.
SHOCK_DATES       = ["2024-12-01", "2025-12-01"]              # beta14a: ["2024-12-01", "2025-12-01"]
# Excluded from the shock calibration so the target quarter cannot help predict
# itself. Set it to the shock you are actually nowcasting.
SHOCK_TARGET_DATE = "2025-12-01"            # beta14a: "2025-12-01"     (H. Melissa) or None

# ---- F. Watchlist and dispersion bands -------------------------------------
# WATCHLIST reports how far the models disagree (P25/P50/P75 across the whole
# spec x estimator pool) and tracks named "oracle" clusters that did well before.
# The K values are country- and run-specific: read them off a completed run,
# do not carry Jamaica's over.
# Switches: WATCHLIST, DISPERSION_BANDS, in the master block above.
ORACLE_KS              = [43, 47, 48, 49, 50]         # beta14a: [43, 47, 48, 49, 50]
ORACLE_ESTIMATORS      = ["ENET", "LASSO"]
ORACLE_VINTAGE         = 1
ORACLE_KS_TREE         = [29, 30, 31]         # beta14a: [29, 30, 31]
ORACLE_TREE_ESTIMATORS = ["DT", "RF", "GBT"]
ORACLE_TREE_VINTAGE    = 2

# ============================================================================
#  Derived values + helpers (normally no need to edit below)
# ============================================================================

# ---- Country registry ------------------------------------------------------
# name -> iso2 (the project's CODE), iso3 (GADM), bbox, and the VIIRS tiles the
# country's ACTUAL GEOMETRY touches.
#
# `tiles` is not hand-written: it was derived by intersecting each GADM level-0
# polygon with the 10x10-degree VIIRS grid (see refresh_registry() below, and
# tiles_from_bbox() for the cheap approximation). Deriving from the polygon
# rather than the bounding box matters -- a bounding box is a rectangle, so for
# an irregularly shaped country it can reach into a tile the land never touches,
# and you would then download and scan a tile holding nothing you care about.
#
# To add a country: put the name here with its iso2/iso3, then run
# refresh_registry() once in the `geo` env and paste in the tiles it prints.
COUNTRIES = {
    "Barbados":            dict(iso2="BB", iso3="BRB", bbox=(-59.65, 13.04, -59.42, 13.34),
                                tiles=["h12v07"]),
    "Jamaica":             dict(iso2="JM", iso3="JAM", bbox=(-78.37, 17.02, -75.97, 18.53),
                                tiles=["h10v07"]),
    "The Bahamas":         dict(iso2="BS", iso3="BHS", bbox=(-80.48, 20.91, -72.71, 27.27),
                                tiles=["h09v06", "h10v06"]),
    "Belize":              dict(iso2="BZ", iso3="BLZ", bbox=(-89.22, 15.89, -87.49, 18.50),
                                tiles=["h09v07"]),
    "Guyana":              dict(iso2="GY", iso3="GUY", bbox=(-61.39, 1.18, -56.48, 8.53),
                                tiles=["h11v08", "h12v08"]),
    "Suriname":            dict(iso2="SR", iso3="SUR", bbox=(-58.09, 1.83, -53.98, 6.02),
                                tiles=["h12v08"]),
    "Trinidad and Tobago": dict(iso2="TT", iso3="TTO", bbox=(-61.93, 10.04, -60.49, 11.36),
                                tiles=["h11v07"]),
}

if COUNTRY not in COUNTRIES:
    raise KeyError(
        f"{COUNTRY!r} is not in COUNTRIES. Add it (iso2, iso3, bbox) and run "
        f"refresh_registry() to derive its tiles. Known: {sorted(COUNTRIES)}")

_C = COUNTRIES[COUNTRY]

CODE      = _C["iso2"]          # -> "metadata - <CODE>.csv"
CODE_LC   = CODE.lower()        # -> bs_gdp.csv, bs_cpi.csv ...
GADM_ISO3 = _C["iso3"]          # -> gadm41_<ISO3>_shp.zip
BBOX      = _C["bbox"]          # (lon_min, lat_min, lon_max, lat_max)

# ---- File names that carry the country code --------------------------------
# Site lat/lon for the point-based NTL extraction. One file per country ships in
# raw/ (Coordinates_bs.xlsx, Coordinates_jm.xlsx, ...) -- all seven are included.
COORDINATES_XLSX = f"Coordinates.xlsx"

# NTL outputs are named after the country code, so switching COUNTRY does not
# silently overwrite the series you built for the previous one.
GDP_CSV        = f"gdp_{CODE_LC}.csv"          # quarterly real GDP     -> raw/
NTL_OUTPUT_CSV = f"ntl_{CODE_LC}.csv"          # site-level daily NTL   -> raw/
NTL_SHAPE_CSV  = f"ntl_shape_{CODE_LC}.csv"    # country-boundary daily -> raw/
BLK_NTL_CSV    = f"blk_ntl_{CODE_LC}.csv"      # monthly composite      -> data/

# ---- VIIRS Black Marble NTL tiles ------------------------------------------
# Derived from COUNTRY. Extraction reads EVERY tile here; single- and multi-tile
# countries use the same code path.
TILES = list(_C["tiles"])

# Set only to OVERRIDE the derived list (rare -- e.g. to skip a tile that touches
# the country but holds nothing you care about). None = use the derived TILES.
#
# Example: The Bahamas touches h09v06 only along a thin strip west of longitude
# -80, which is open water and sandbank. TILES_OVERRIDE = ["h10v06"] would halve
# the download at essentially no cost in coverage. Left off by default so you can
# see the multi-tile code path work.
TILES_OVERRIDE = None
if TILES_OVERRIDE:
    TILES = list(TILES_OVERRIDE)

# Subset of TILES to DOWNLOAD. Leave None to download all TILES. Set this only
# when a tile is already populated by a neighbour that shares it
# (e.g. Guyana = ["h11v08"] because its h12v08 is downloaded by Suriname).
DOWNLOAD_TILES = None

GADM_ZIP = "gadm41_" + GADM_ISO3 + "_shp.zip"

# ---- Where the folders are ---------------------------------------------------
# The workshop root owns raw/, data/, output/, logs/, ntl_tmp/ and hfiles/.
# Give its location RELATIVE TO THIS FILE (or an absolute path).
#   ".."     -> this config.py sits in <workshop root>/activity/     <- current
#   "../.."  -> ... in <workshop root>/workshop_code/activity/
PROJECT_ROOT = ".."

# Where the VNP46A2 .h5 tiles live, as <NL_ROOT>/<tile>/*.h5 .
#   None -> <PROJECT_ROOT>/hfiles, i.e. a self-contained workshop bundle.
#   NL_ROOT = "../../hfiles"
NL_ROOT = "../../hfiles"


def tiles_from_bbox(bbox=None):
    """VIIRS tiles a bounding box touches. Cheap, no geometry, no dependencies.

    The VIIRS Black Marble grid is 10x10 degrees:
        h = floor((lon + 180) / 10)     v = floor((90 - lat) / 10)

    This OVER-COVERS for countries whose bounding box reaches into a tile the
    land does not actually touch, so the registry stores geometry-derived tiles.
    Use this only as a first guess for a new country.
    """
    import math
    lo, la, hi, ha = bbox or BBOX
    h0, h1 = math.floor((lo + 180) / 10), math.floor((hi + 180) / 10)
    v0, v1 = math.floor((90 - ha) / 10), math.floor((90 - la) / 10)
    return [f"h{h:02d}v{v:02d}" for v in range(v0, v1 + 1) for h in range(h0, h1 + 1)]


def refresh_registry(shape_dir=None, countries=None):
    """Re-derive bbox + tiles for every country from its GADM polygon.

    This is how you add a country that is not in COUNTRIES above: add the name
    with its iso2/iso3 and any rough bbox, then run this and paste back what it
    prints.

    Needs geopandas/shapely, so run it in the `geo` env -- NOT at import time:

        conda run -n geo python -c "import config; config.refresh_registry()"
    """
    import os, urllib.request
    import geopandas as gpd
    from shapely.geometry import box

    shape_dir = shape_dir or os.path.join(os.getcwd(), "shapes")
    os.makedirs(shape_dir, exist_ok=True)
    out = {}
    for name in (countries or COUNTRIES):
        iso3 = COUNTRIES[name]["iso3"]
        zp = os.path.join(shape_dir, f"gadm41_{iso3}_shp.zip")
        if not os.path.exists(zp):
            urllib.request.urlretrieve(
                f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{iso3}_shp.zip", zp)
        g = gpd.read_file(f"zip://{zp}!gadm41_{iso3}_0.shp")
        geom = g.geometry.union_all() if hasattr(g.geometry, "union_all") else g.geometry.unary_union
        b = geom.bounds
        cand = tiles_from_bbox(b)
        exact = [t for t in cand
                 if geom.intersects(box(10 * int(t[1:3]) - 180,
                                        90 - 10 * int(t[4:6]) - 10,
                                        10 * int(t[1:3]) - 170,
                                        90 - 10 * int(t[4:6])))]
        out[name] = (b, exact)
        drop = set(cand) - set(exact)
        print(f'    {name!r:24}: dict(iso2="{COUNTRIES[name]["iso2"]}", iso3="{iso3}", '
              f'bbox=({b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}),')
        print(f'{"":30}tiles={exact}),'
              + (f'   # bbox rule also suggested {sorted(drop)} -- not touched' if drop else ""))
    return out


def download_tiles():
    """Tiles to fetch from LAADS -- DOWNLOAD_TILES if set, else all TILES."""
    return list(DOWNLOAD_TILES) if DOWNLOAD_TILES else list(TILES)


def derive_paths(d, make=True):
    """Standard project paths given d = the folder holding this config.py.

    Everything hangs off PROJECT_ROOT, resolved relative to this file, so the
    only thing to edit when the folder moves is that one string.

    Any folder that does not exist yet is CREATED (make=False to just look).
    """
    d = os.path.abspath(d)
    root = os.path.abspath(os.path.join(d, PROJECT_ROOT))
    up = lambda *p: os.path.join(root, *p)
    P = {
        "code":    d,
        "raw":     up("raw"),
        "data":    up("data"),
        "output":  up("output"),
        "logs":    up("logs"),
        "ntl_tmp": up("ntl_tmp"),        # transient .tif during extraction
        "nl_root": os.path.abspath(os.path.join(d, NL_ROOT))
                   if NL_ROOT else up("hfiles"),   # <nl_root>/<tile>/*.h5
    }
    if make:
        for k, p in P.items():
            if k != "code":
                os.makedirs(p, exist_ok=True)
    return P
