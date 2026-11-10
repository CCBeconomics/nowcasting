"""
config_data.py  --  Trinidad and Tobago settings for 1_data_ntl and 1_data_gtrends (the production pipeline).

Modelled on 2026_NWCST_BOL_Workshop/code/config.py (2026-10-19), cut down to
the NTL and Google Trends blocks. Edit THIS file, not the notebooks.
"""
import os

# NASA LAADS DAAC download token. Expires after ~60 days.
# Regenerate at https://ladsweb.modaps.eosdis.nasa.gov/ -> My Account -> Generate Token
NASA_TOKEN = ""   # paste your own token here (see the install guide); never share a file that holds it

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
# raw/ntl_shape.csv ends 2026-04-19 (as of 2026-10-19).
NTL_DATE_RANGE = "2026-04-15"

# Safety limits, applied even when NTL_DATE_RANGE is None.
NTL_FIRST_DATE = "2012-01-19"       # first VNP46A2 date; never download anything earlier
NTL_MAX_FILES  = 400                # refuse a run above this (~10 GB); None = no limit

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
GT_KEYWORDS = "../raw/TT_GTrends_Keywords.csv"   # read from data/, so this points to raw/TT_GTrends_Keywords.csv
GT_START    = "2004-01-01"                # Google Trends starts here ("all" on the website)

# ============================================================================
#  PATHS AND HELPERS -- normally no need to edit below
# ============================================================================
GADM_ZIP     = "gadm41_" + GADM_ISO3 + "_shp.zip"
PROJECT_ROOT = ".."                 # tto/, relative to this file: raw/, data/, output/ are shared with activity/
NL_ROOT      = None                 # None = tto/hfiles/<tile>/, created on first download; or a path to an existing tile store


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
        # The default tile store (tto/hfiles/) is created; a store named in
        # NL_ROOT never is, so a wrong path fails loudly instead of starting
        # a fresh, empty store.
        for k, p in P.items():
            if k == "code" or (k == "nl_root" and NL_ROOT):
                continue
            os.makedirs(p, exist_ok=True)
    return P
