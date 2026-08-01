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
COUNTRY = "The Bahamas"

# ============================================================================
#  2. THE NIGHTTIME-LIGHTS WINDOW  (act3_ntl.ipynb)
# ============================================================================
NTL_DATE_RANGE = "2026-01-05 2026-01-18"

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

# The country-code-stamped file names (Coordinates_bs.xlsx, ntl_bs.csv, ...) are
# built from COUNTRY further down, once CODE_LC exists.

# ---- Key series ------------------------------------------------------------
GDP_TICKER    = "RGDP0000"   # real-GDP target (project-wide convention)
CPI_TICKER    = "MCPI0000"   # CPI used to DEFLATE nominal vars (BB=MCPI0000, JM=RCPI0000)
CPI_BASE_YEAR = 2018         # base year of that CPI index (differs by country)

# ============================================================================
#  5. NOWCAST RUN SETTINGS  (act5 - act8)
# ============================================================================
# act7_nwcst and act8_results MUST share DATE_STR -- it selects output/<DATE_STR>/.
DESIRED_DATE = "2026-06-01"                      # target quarter start (YYYY-MM-01)
DATE_STR     = date.today().strftime("%Y%m%d")   # output/<DATE_STR>/ = today's run

TRAIN_START = "2013-06-01"          # training sample start
TEST_START  = "2023-06-01"          # test / holdout sample start
LAGS        = list(range(-2, 3))    # vintages evaluated: -2..+2 (backcast..forecast)

# Which models to fit (1 = on, 0 = off).
RUN = {
    "ols": 1, "olsr": 1, "enet": 1, "lasso": 1,
    "gbt": 1, "dt": 1, "rf": 1, "lstm": 1,
}

# Iterations / ensemble sizes. The workshop defaults are deliberately small so a
# full run finishes inside a session; the production pipeline uses 100.
ITER       = 25    # bagging ensemble size for DT / RF / GBT
ITER_LSTM  = 10    # LSTM ensemble size
EPOCH_LSTM = 25    # LSTM training episodes

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
#   Otherwise a path, absolute or relative to THIS FILE. To reuse the shared
#   project-wide store (years of tiles already downloaded, nothing to fetch):
#       NL_ROOT = "../../hfiles"
NL_ROOT = None


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
