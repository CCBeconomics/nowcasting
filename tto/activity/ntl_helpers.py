"""
ntl_helpers.py  —  NTL (VIIRS Black Marble) download + extraction helpers.

Extracted from the baseline 1_data_ntl.ipynb so the notebook stays a thin
orchestration layer. Functions that used notebook globals now take them as
explicit arguments (api_key, path_root). Two extraction back-ends:
    processHD5        point-based (lat/lon list, Coordinates.xlsx)
    processHD5_shape  polygon/shape-based (renamed from processHD5_island)

Trinidad copy (2026-10-19) of 2026_NWCST_BOL_Workshop/code/ntl_helpers.py.
One change: shape_dates_done / update_shape_csv take `out_cols`, so the shape
CSV keeps Trinidad's registered column names (shape_mean, shape_sum).
"""
import os, re, sys, glob, time, requests
import threading
from datetime import datetime
import numpy as np
import statistics as stat


def _bootstrap_gdal_env():
    """Point GDAL at its data and plugin folders inside this interpreter's env.

    Conda normally sets GDAL_DATA and GDAL_DRIVER_PATH from an ACTIVATION
    script. A Jupyter kernel is frequently launched by running the environment's
    python.exe directly, which skips activation -- so GDAL comes up without
    them.

    That matters here because gdal 3.11 (conda-forge) no longer ships the HDF5
    driver in libgdal-core: it lives in a separate `libgdal-hdf5` package, as
    gdal_HDF5.dll in <env>/Library/lib/gdalplugins. Unfound, GDAL does not raise
    -- gdal.Open() just returns None on every .h5, the notebook's per-file
    try/except swallows the resulting AttributeError, and the whole run finishes
    "successfully" with an empty series. Two failure modes, in order:

      1. plugin folder not searched  -> "not recognized as being in a
         supported file format ... plugin gdal_HDF5.dll is not available"
      2. folder found, DLL won't load -> "Can't load requested DLL ... 126"
         because the plugin's own dependencies (libhdf5 and friends) live in
         <env>/Library/bin, which is only on PATH after activation.

    Setting all three below fixes both. Harmless when activation already ran,
    and a no-op on an env that has no such folders.

    If the HDF5 driver is genuinely absent:  conda install -n geo -c conda-forge libgdal-hdf5
    """
    root = sys.prefix
    win  = os.path.join(root, "Library")
    base = win if os.path.isdir(win) else root          # Windows vs unix layout

    for var, sub in (("GDAL_DATA",        ("share", "gdal")),
                     ("GDAL_DRIVER_PATH", ("lib", "gdalplugins")),
                     ("PROJ_LIB",         ("share", "proj"))):
        path = os.path.join(base, *sub)
        if not os.environ.get(var) and os.path.isdir(path):
            os.environ[var] = path

    # The plugin is loaded with a plain LoadLibrary, which searches PATH --
    # os.add_dll_directory() does NOT cover it, so prepend the folder instead.
    libbin = os.path.join(base, "bin")
    if os.path.isdir(libbin) and libbin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = libbin + os.pathsep + os.environ.get("PATH", "")


_bootstrap_gdal_env()

# Heavy geospatial deps (present in the `geo` env). Wrapped so the module can be
# imported / syntax-checked in the base env; the functions need them at runtime.
try:
    from osgeo import gdal, ogr, gdalnumeric
    import h5py
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import mapping
except Exception:
    gdal = ogr = gdalnumeric = h5py = None
    rasterio = rio_mask = mapping = None


def check_hdf5_driver(raise_on_missing=True):
    """Confirm GDAL can actually open HDF5 before a long run starts.

    Call this FIRST. Without the driver every .h5 read fails silently (see
    _bootstrap_gdal_env), so an overnight job can burn hours downloading tiles
    it will then be unable to read.
    """
    if gdal is None:
        msg = ("GDAL is not importable in this kernel. Run this notebook with "
               "the `geo` kernel, not the default one.")
    elif gdal.GetDriverByName("HDF5") is None:
        msg = ("GDAL has no HDF5 driver, so every VIIRS tile would read as "
               "empty. Install it into the geo env and restart the kernel:\n"
               "    conda install -n geo -c conda-forge libgdal-hdf5")
    else:
        return True
    if raise_on_missing:
        raise RuntimeError(msg)
    print("WARNING:", msg)
    return False

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


def _note(msg):
    """Print a message WITHOUT breaking an active tqdm bar (uses tqdm.write so the
    bar stays put and messages scroll above it). Falls back to print if no tqdm."""
    if _tqdm is not None:
        _tqdm.write(str(msg))
    else:
        print(msg)


_LOG_LOCK = threading.Lock()   # serialize log appends when downloading in parallel


def _append_fail(log_path: str, line: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with _LOG_LOCK, open(log_path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def list_tile_rasters(nl_root, tiles, product="VNP46A2", make=True):
    """Inventory downloaded NTL files across ALL tiles (multi-tile safe).

    Ensures each tile's central subfolder ``nl_root/<tile>/`` exists (when
    ``make``) and returns a sorted list of that product's ``.h5`` filenames
    gathered from every tile subfolder. Basenames only — each file is later
    read via its full per-tile path (the tile is embedded in the filename).
    """
    files = []
    for t in tiles:
        tdir = os.path.join(nl_root, t)
        if make:
            os.makedirs(tdir, exist_ok=True)
        if os.path.isdir(tdir):
            files += [f for f in os.listdir(tdir) if product in f and f.endswith(".h5")]
    files.sort()
    return files


def conJDtoDate(JD):
    '''
    Anotations
    '''
    date = datetime.strptime(JD, '%y%j').date()
    return date


def getRasterData(lat, lon, window, xOrigin, yOrigin, pixelWidth, pixelHeight, data):
    '''
    Anotations:
    
    '''
    
    nrows, ncols = data.shape
    col = int((lon - xOrigin) / pixelWidth )
    row = int((yOrigin - lat) / pixelHeight)

    if not (0 <= row < nrows and 0 <= col < ncols):
        return float("nan")

    #Data AT THAT ROW COLUMN
    if(window == 3):
        '''
        This is a grid of 3x3 around the lat,lon.

        Clipped to the array so a site within one pixel of the tile edge yields
        the 2x3 or 2x2 block that exists instead of raising IndexError -- which
        the callers catch per FILE, so one edge site would otherwise throw away
        every site on that date. np.nanmean, not statistics.mean, so a block
        holding a fill pixel still returns the mean of the real ones.
        '''
        r0, r1 = max(row - 1, 0), min(row + 2, nrows)
        c0, c1 = max(col - 1, 0), min(col + 2, ncols)
        block  = data[r0:r1, c0:c1]
        if np.all(np.isnan(block)):
            return float("nan")
        return round(float(np.nanmean(block)), 2)

    else:
        #print(window)
        value = data[row][col]
        return float(value)


def download_vnp46a2(year, day_of_year, tile, api_key, path_root, output_dir=None,
                     file_list=None, log_path=None, max_retries=3, retry_sleep=1.0):
    """
    Download the VNP46A2 file(s) for one date + tile, with retries.

    Transient failures (an HTTP error or a network exception) are retried up to
    `max_retries` times, sleeping `retry_sleep` seconds between attempts. Anything
    that still fails is appended to `log_path` and skipped, so the caller's loop
    keeps going instead of crashing.

    Args:
        year, day_of_year, tile: which file(s) to fetch
        path_root: central store root; files land in path_root/<tile>/
        log_path:  where to append 'missing / failed' notes (None = don't log)
        max_retries, retry_sleep: retry budget per request
    """

    # Route each tile to its own central subfolder and skip files already there
    if output_dir is None:
        output_dir = os.path.join(path_root, tile)
    os.makedirs(output_dir, exist_ok=True)
    if file_list is None:
        file_list = os.listdir(output_dir)
    base_url = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP46A2"
    url = f"{base_url}/{year}/{day_of_year:03d}"
    archive_dir = f"{base_url}/{year}/{int(day_of_year):03d}/"   # browsable dir for manual download
    headers = {"Authorization": f"Bearer {api_key}"}

    # --- List the day's files (retry transient errors, then give up gracefully) ---
    filenames = None
    last_err = "unknown"
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(f"{url}.json", headers=headers, timeout=60)
            if response.status_code == 200:
                files = response.json()
                if isinstance(files, dict):
                    filenames = files.get('content') or files.get('files') or list(files.keys())
                break
            last_err = f"HTTP {response.status_code}"
        except Exception as e:
            last_err = repr(e)
        if attempt < max_retries:
            time.sleep(retry_sleep)

    # Failures below are LOGGED (with URLs) but not printed per-file — the caller
    # reports a single summary at the end. Returns "ok" / "unavailable" / "failed".
    if filenames is None:
        msg = f"{year}-{int(day_of_year):03d} {tile}: file listing failed after {max_retries} tries ({last_err}) | {archive_dir}"
        if log_path:
            _append_fail(log_path, msg)
        return "failed"

    matching_files = [f['downloadsLink'] for f in filenames if tile in f['downloadsLink'] and f['downloadsLink'].endswith('.h5') ]

    if not matching_files:
        # The day's directory exists but this tile has no acquisition (a genuine
        # archive gap) — not an error, just unavailable.
        msg = f"{year}-{int(day_of_year):03d} {tile}: no file on server | {archive_dir}"
        if log_path:
            _append_fail(log_path, msg)
        return "unavailable"

    status = "ok"
    for file_url in matching_files:
        filename = file_url.split("/")[-1]
        output_path = os.path.join(output_dir, filename)

        if filename in file_list:
            continue                       # already downloaded — check the next matching file

        # Download with retries; write to a .part file and rename on success so a
        # failed/partial transfer never leaves a file that looks complete.
        ok = False
        last_err = "unknown"
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.get(file_url, headers=headers, stream=True, timeout=120)
                if r.status_code == 200:
                    tmp_path = output_path + ".part"
                    with open(tmp_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    os.replace(tmp_path, output_path)   # atomic: appears only when complete
                    ok = True
                    break
                last_err = f"HTTP {r.status_code}"
            except Exception as e:
                last_err = repr(e)
            if attempt < max_retries:
                time.sleep(retry_sleep)

        if not ok:
            status = "failed"
            msg = f"{year}-{int(day_of_year):03d} {tile} {filename}: download failed after {max_retries} tries ({last_err}) | {file_url}"
            if log_path:
                _append_fail(log_path, msg)
            try:
                if os.path.exists(output_path + ".part"):
                    os.remove(output_path + ".part")
            except Exception:
                pass
    return status


class TooManyTargets(RuntimeError):
    """Raised when one run would download more files than max_files allows."""


def download_many(targets, api_key, path_root, log_path=None, max_workers=8,
                  max_retries=3, retry_sleep=1.0, progress=True, max_files=None):
    """Download many (tile, year, day) targets CONCURRENTLY.

    Downloading is I/O-bound, so a thread pool overlaps the network waits and is
    much faster than a sequential loop. Each target is handed to download_vnp46a2
    (which retries transient errors and logs failures), so nothing raised here
    aborts the batch. `targets` is the list of (tile, year, day) tuples built in
    1_data_ntl. Lower `max_workers` if the server starts rate-limiting.

    max_files: refuse the whole batch when it is larger than this, raising
        TooManyTargets BEFORE a single byte is fetched. The guard lives here, in
        the function that does the downloading, and not in the notebook cell
        that calls it -- on 2026-09-13 the window was left at None and this ran
        unbounded, fetching 20,985 files and 490 GB and filling the disk. A
        check in the caller would have been skipped in exactly the same way.
        None disables it.

    Returns a dict of status counts: {'ok', 'unavailable', 'failed'}.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from collections import Counter

    if max_files is not None and len(targets) > max_files:
        gb = len(targets) * 25 / 1024
        raise TooManyTargets(
            "%s files to download (~%.1f GB), over the limit of %s. "
            "Nothing was downloaded. Either narrow cfg.NTL_DATE_RANGE, or raise "
            "cfg.NTL_MAX_FILES once you have checked the free disk space, or "
            "pass max_files=None to turn the check off."
            % (format(len(targets), ","), gb, format(max_files, ","))
        )

    def _one(t):   # t = (tile, year, day)
        return download_vnp46a2(t[1], t[2], t[0], api_key, path_root,
                                log_path=log_path, max_retries=max_retries,
                                retry_sleep=retry_sleep)

    counts = Counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_one, t) for t in targets]
        it = as_completed(futures)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, total=len(futures), desc=f"Downloading (x{max_workers})")
        for fut in it:
            counts[fut.result() or "ok"] += 1
    return dict(counts)


def list_years(last_year, last_day, tile, api_key, end_date=None, output_dir=None, collection="5200", product='VNP46A2'):
    # 1. Find all available years
    base_url = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/{collection}/{product}"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Get all available years
    response = requests.get(f"{base_url}.json", headers=headers)
    response = response.json()
    all_years = [item['name'] for item in response['content']]

    # Keep only years >= last_year (and <= end_date's year when a range is set)
    years_to_download = [year for year in all_years if int(year) >= int(last_year)]
    if end_date is not None:
        years_to_download = [year for year in years_to_download if int(year) <= end_date.year]

    return years_to_download


def get_target(target, last_year, last_day, tile, api_key, end_date=None, output_dir=None, collection="5200", product='VNP46A2'):
    """
    Download all available days from target years that are after last observation
    (and strictly before end_date when a custom range is set).

    Args:
        target: List of years to check (e.g., ['2024', '2025'])
        last_year: Last year you have (e.g., 2024)
        last_day: Last day of year you have (e.g., 100)
        tile: Tile identifier
        end_date: Optional datetime; stop before this date (used for custom ranges).
                  When None, fetch every available day up to the latest (incremental).
        output_dir: Output directory
    """

    base_url = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/{collection}/{product}"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Convert last observation to datetime for comparison
    last_observation = datetime.strptime(f"{last_year}-{last_day}", "%Y-%j")

    my_list = []
    for year in target:
        # Get all available days for this year
        response = requests.get(f"{base_url}/{year}.json", headers=headers)

        if response.status_code != 200:
            print(f"Error getting days for year {year}: {response.status_code}")
            continue

        response_data = response.json()
        all_days = [item['name'] for item in response_data['content']]

        # Filter to only numeric day values
        day_numbers = sorted([int(day) for day in all_days if day.isdigit()])
        for day in day_numbers:
            dt = datetime.strptime(f"{year}-{day}", "%Y-%j")
            if dt > last_observation and (end_date is None or dt < end_date):
                my_list.append(( year , day ))
    return my_list


def missing_targets(tiles, api_key, nl_root, start_date=None, end_date=None,
                    start_year="2012", product="VNP46A2"):
    """(tile, year, day) files the server lists as available but are NOT on disk.

    This is a DIFF, not a resume-from-last: it compares the server's available-day
    list (optionally bounded to [start_date, end_date]) against the .h5 files
    already in nl_root/<tile>/. So it backfills HOLES left by earlier failed days
    as well as picking up new dates. Feed the result straight to download_many().

    start_year: earliest year to query when start_date is None (VNP46A2 -> 2012).
    """
    targets = []
    for tile in tiles:
        anchor = str(start_date.year) if start_date is not None else start_year
        years  = list_years(anchor, "001", tile, api_key, end_date=end_date)
        # anchor at Jan-1 so get_target returns EVERY available day in the window
        avail  = set(get_target(years, anchor, "001", tile, api_key, end_date=end_date))
        if start_date is not None:
            avail = {(y, d) for (y, d) in avail
                     if datetime.strptime(f"{y}-{d}", "%Y-%j") >= start_date}

        # (year, day) already on disk for this tile
        on_disk = set()
        tdir = os.path.join(nl_root, tile)
        if os.path.isdir(tdir):
            for f in os.listdir(tdir):
                if product in f and f.endswith(".h5"):
                    m = re.search(r'\.A(\d{4})(\d{3})\.', f)
                    if m:
                        on_disk.add((m.group(1), int(m.group(2))))

        for (y, d) in sorted(avail - on_disk):
            targets.append((tile, y, d))
    return targets


def processHD5(inputHD5, layer, OutputFolder, coords, Date ):
    """Read one .h5 at each of `coords` and return one record per site.

    OutputFolder is accepted and IGNORED -- kept so the call sites keep matching
    processHD5_shape(), which does still need a scratch folder. Nothing is
    written to disk here any more; see the note further down.
    """
    ## Open HDF file
    hdflayer = gdal.Open(inputHD5, gdal.GA_ReadOnly)
    subhdflayer = hdflayer.GetSubDatasets()[layer][0]
    rlayer = gdal.Open(subhdflayer, gdal.GA_ReadOnly)

    if rlayer is None:
        print("Could not open image", Date)
        return []          # skip this file rather than crashing on None.ReadAsArray()

    HorizontalTileNumber = int(rlayer.GetMetadata_Dict()["HorizontalTileNumber"])
    VerticalTileNumber = int(rlayer.GetMetadata_Dict()["VerticalTileNumber"])
    WestBoundCoord = (10*HorizontalTileNumber) - 180
    NorthBoundCoord = 90-(10*VerticalTileNumber)

    EastBoundCoord = WestBoundCoord + 10
    SouthBoundCoord = NorthBoundCoord - 10

    # The grid is read STRAIGHT out of the .h5, with no GeoTIFF in between.
    #
    # This used to gdal.Translate() the subdataset to a ~23 MB GeoTIFF in
    # ntl_tmp/, reopen it, and read the array back -- to obtain a geotransform
    # that -a_ullr had just written from these same four bound coordinates. The
    # detour cost a 23 MB write + read per .h5 and, because the housekeeping was
    # commented out, left every one of them on disk: ~5 GB for a two-month
    # four-tile Bolivian window, ~450 GB for a full-archive run.
    #
    # A VNP46A2 tile is a plain 10-degree x 10-degree grid, 2400 x 2400 pixels,
    # row 0 at the north edge, so the geotransform is exact arithmetic:
    #   origin = (WestBoundCoord, NorthBoundCoord),  pixel = 10/2400 degrees.
    # Identical numbers, no temporary file. processHD5_shape() still writes one
    # small .tif because rasterio's polygon mask needs a georeferenced source.
    rows, cols  = rlayer.RasterYSize, rlayer.RasterXSize
    xOrigin     = WestBoundCoord
    yOrigin     = NorthBoundCoord
    pixelWidth  = 10.0 / cols
    pixelHeight = 10.0 / rows

    data = rlayer.ReadAsArray().astype(float)
    rlayer   = None
    hdflayer = None
    # VIIRS fill -> NaN BEFORE any pixel is read, so a filled pixel cannot enter
    # a 3x3 mean as a number. VNP46A2 v002 fills the NTL layers with -999.9;
    # older collections used 65535. Both are caught here.
    data[data >= 65535] = np.nan
    data[data < 0]      = np.nan

    _datalist_ = []
    
    # Here starts the lat lon part.
    for coord in coords :
        
        lat = coord['lat']
        lon = coord['lon']
        
        if lat < SouthBoundCoord or lat > NorthBoundCoord or lon < WestBoundCoord or lon > EastBoundCoord:
            #print(f"Latitude {lat} and longitude {lon} are outside tile bounds. Skipping file.")
            continue

        value1 = getRasterData(lat, lon, 1 , xOrigin, yOrigin, pixelWidth, pixelHeight, data)
        value3 = getRasterData(lat, lon, 3 , xOrigin, yOrigin, pixelWidth, pixelHeight, data)
        
        _d_ = {
            'sector' : coord['sector'],
            'location' : coord['location'],
            'city' : coord['city'],
            'JD' : Date,            
            'DNBvalue1' : value1 ,
            'DNBvalue3' : value3 ,
        }        
        
        _datalist_.append(_d_)

    return _datalist_


def processHD5_shape(inputHD5, layer, OutputFolder, gdf_boundary, Date):
    """
    Extract mean NTL radiance over the full island boundary from one h5 file.
    Returns a dict with date, mean, sum, and pixel count.
    """
    hdflayer    = gdal.Open(inputHD5, gdal.GA_ReadOnly)
    subhdflayer = hdflayer.GetSubDatasets()[layer][0]
    rlayer      = gdal.Open(subhdflayer, gdal.GA_ReadOnly)

    meta = rlayer.GetMetadata_Dict()
    H = int(meta["HorizontalTileNumber"])
    V = int(meta["VerticalTileNumber"])
    west, north = (10 * H) - 180, 90 - (10 * V)
    east, south = west + 10, north - 10

    tmp = os.path.join(OutputFolder, "_island_tmp.tif")
    # -ot Float32 is REQUIRED, not cosmetic -- same reason as in raster_to_array().
    # Older VNP46A2 files (collection 001, i.e. anything before ~2020) store the DNB
    # layer as uint16. The rio_mask call below fills with np.nan, which cannot be
    # written into an integer array:
    #     TypeError: Cannot convert fill_value nan to dtype uint16
    # Without this, EVERY pre-2020 date is silently skipped by the caller's
    # try/except and the historical series comes back empty.
    opts = gdal.TranslateOptions(gdal.ParseCommandLine(
        f"-a_srs EPSG:4326 -ot Float32 -a_ullr {west} {north} {east} {south}"
    ))
    gdal.Translate(tmp, rlayer, options=opts)
    rlayer = None  # release handle

    shapes = [mapping(geom) for geom in gdf_boundary.geometry]
    with rasterio.open(tmp) as src:
        out_image, _ = rio_mask(src, shapes, crop=True, nodata=np.nan, filled=True)

    try:
        os.remove(tmp)
    except Exception:
        pass

    data = out_image[0].astype(float)
    data[data >= 65535] = np.nan   # VIIRS fill value
    data[data < 0]      = np.nan

    valid = ~np.isnan(data)
    return {
        "date"    : Date,
        "ntl_mean": float(np.nanmean(data)) if valid.any() else np.nan,
        "ntl_sum" : float(np.nansum(data)),
        "ntl_n"   : int(valid.sum()),
    }


def h5_full_path(nl_root, fname):
    """Resolve a VNP46A2 basename to its full path in the central per-tile store
    (hfiles/<tile>/<fname>). The tile is field 2 of the dotted filename."""
    return os.path.join(nl_root, fname.split(".")[2], fname)


def rasters_in_range(nl_root, tiles, start=None, end=None, product="VNP46A2"):
    """Full paths of product .h5 files across ALL tiles, optionally restricted to
    a [start, end] date window (inclusive). start/end are date/datetime or None.
    Returns a list of (full_path, date) sorted by date."""
    def _d(x):
        return x.date() if hasattr(x, "date") else x
    s = _d(start) if start is not None else None
    e = _d(end) if end is not None else None
    out = []
    for t in tiles:
        tdir = os.path.join(nl_root, t)
        if not os.path.isdir(tdir):
            continue
        for f in os.listdir(tdir):
            if product in f and f.endswith(".h5"):
                dt = conJDtoDate(f[11:16])
                if (s is None or dt >= s) and (e is None or dt <= e):
                    out.append((os.path.join(tdir, f), dt))
    out.sort(key=lambda p: p[1])
    return out


def find_h5s_for_date(target_date, nl_root, tiles, product="VNP46A2"):
    """Full paths of the .h5 files whose embedded date == target_date, ONE PER
    TILE. Returns [] if nothing matches. target_date is a date/datetime/Timestamp.

    A country that spans several VIIRS tiles (The Bahamas = h09v06 + h10v06) has
    one file per tile per date, and a map needs ALL of them -- taking just the
    first gives you whatever 10-degree square happens to sort first, which for
    The Bahamas is the near-empty strip of ocean west of longitude -80.
    """
    tgt = target_date.date() if hasattr(target_date, "date") else target_date
    hits = []
    for t in tiles:
        tdir = os.path.join(nl_root, t)
        if not os.path.isdir(tdir):
            continue
        for f in sorted(os.listdir(tdir)):
            if product in f and f.endswith(".h5") and conJDtoDate(f[11:16]) == tgt:
                hits.append(os.path.join(tdir, f))
                break            # one file per tile per date
    return hits


def find_h5_for_date(target_date, nl_root, tiles, product="VNP46A2"):
    """First tile's .h5 for target_date, else None.

    Single-tile convenience wrapper. For MAPS use find_h5s_for_date -- see the
    warning in its docstring about multi-tile countries.
    """
    hits = find_h5s_for_date(target_date, nl_root, tiles, product=product)
    return hits[0] if hits else None


def _h5_layer_to_geotiff(h5_path, layer, out_tif):
    """Translate one h5 subdataset to a georeferenced Float32 GeoTIFF (EPSG:4326).

    The .h5 carries no CRS, so the tile's own H/V grid numbers are read from its
    metadata and turned into the corner coordinates handed to -a_ullr.
    """
    hdflayer    = gdal.Open(h5_path, gdal.GA_ReadOnly)
    subhdflayer = hdflayer.GetSubDatasets()[layer][0]
    rlayer      = gdal.Open(subhdflayer, gdal.GA_ReadOnly)

    meta  = rlayer.GetMetadata_Dict()
    H, V  = int(meta["HorizontalTileNumber"]), int(meta["VerticalTileNumber"])
    west, north = (10 * H) - 180, 90 - (10 * V)
    east, south = west + 10, north - 10

    # -ot Float32 is REQUIRED, not cosmetic. Older VNP46A2 files (collection 001 --
    # e.g. Bahamas tiles before ~2020) store the DNB layer as uint16, and the
    # rio_mask call below fills with np.nan, which cannot be written into an
    # integer array:
    #     TypeError: Cannot convert fill_value nan to dtype uint16
    # Without this, raster_to_array raises on every pre-collection-002 file.
    opts = gdal.TranslateOptions(gdal.ParseCommandLine(
        f"-a_srs EPSG:4326 -ot Float32 -a_ullr {west} {north} {east} {south}"
    ))
    gdal.Translate(out_tif, rlayer, options=opts)
    rlayer = None
    return out_tif


def raster_to_array(h5_paths, layer, gdf_boundary, out_folder):
    """Clip an h5 layer to a boundary polygon and return (data2d, lons1d, lats1d)
    with VIIRS fill/negative values masked to NaN. Used to draw NTL maps.

    h5_paths is ONE path or a LIST of paths (the same date across several tiles).
    Several tiles are mosaicked into a single raster BEFORE the clip, so a country
    straddling a tile boundary comes out whole instead of cut at the seam.
    """
    import numpy as np
    paths = [h5_paths] if isinstance(h5_paths, (str, bytes, os.PathLike)) else list(h5_paths)
    if not paths:
        raise ValueError("raster_to_array: no .h5 path given")

    os.makedirs(out_folder, exist_ok=True)
    tmps = [_h5_layer_to_geotiff(p, layer, os.path.join(out_folder, f"_map_tmp_{i}.tif"))
            for i, p in enumerate(paths)]

    try:
        if len(tmps) == 1:
            src_path = tmps[0]
        else:
            # Adjacent VIIRS tiles do not overlap, so a plain merge stitches them.
            from rasterio.merge import merge as rio_merge
            srcs = [rasterio.open(t) for t in tmps]
            mosaic, mosaic_transform = rio_merge(srcs)
            profile = srcs[0].profile
            for s in srcs:
                s.close()
            profile.update(driver="GTiff", dtype="float32", count=1,
                           height=mosaic.shape[1], width=mosaic.shape[2],
                           transform=mosaic_transform)
            src_path = os.path.join(out_folder, "_map_tmp_mosaic.tif")
            with rasterio.open(src_path, "w", **profile) as dst:
                dst.write(mosaic[0].astype("float32"), 1)
            tmps.append(src_path)

        shapes = [mapping(geom) for geom in gdf_boundary.geometry]
        with rasterio.open(src_path) as src:
            out_image, out_transform = rio_mask(src, shapes, crop=True, nodata=np.nan, filled=True)
            nrows, ncols = out_image.shape[1], out_image.shape[2]
            lons = out_transform.c + (np.arange(ncols) + 0.5) * out_transform.a
            lats = out_transform.f + (np.arange(nrows) + 0.5) * out_transform.e
    finally:
        for t in tmps:
            try:
                os.remove(t)
            except Exception:
                pass

    data = out_image[0].astype(float)
    data[data >= 65535] = np.nan
    data[data < 0]      = np.nan
    return data, lons, lats


def save_fig(fig, P, name, notebook="act4", date_str=None, dpi=150):
    """Write a figure to output/<YYYYMMDD>/ under the project naming convention:

        output/20260912/20260912 act4 Fig ntl_daily.png

    Returns the path. An overnight run then leaves reviewable files behind
    instead of pictures that exist only inside the .ipynb -- which matters here
    because notebook outputs are committed, so a re-run shows up as a huge diff.
    """
    from datetime import date as _date
    date_str = date_str or _date.today().strftime("%Y%m%d")
    out_dir  = os.path.join(P["output"], date_str)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{date_str} {notebook} Fig {name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ===========================================================================
#  RESUME SUPPORT
#  Extract only the dates the CSV does not already carry.
#
#  Why this exists: the .h5 tiles are intermediate. A full Bolivian build is
#  ~18,000 files and several hundred GB, so the working pattern is to extract
#  the series and then evict the tiles to cloud storage. Dropbox leaves an
#  online-only placeholder behind: the NAME and SIZE stay readable (so
#  missing_targets() correctly skips re-downloading from LAADS), but the first
#  byte anything READS pulls the whole file back from the cloud.
#
#  So a date already in the CSV must never reach gdal.Open(). Every function
#  below filters on the date alone -- from the filename and the CSV, never
#  from file contents.
# ===========================================================================


def _read_existing(csv_path):
    """An existing CSV as a DataFrame, or None if absent, empty or unreadable."""
    import pandas as pd
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        _note(f"WARNING: {os.path.basename(csv_path)} could not be read ({e}); "
              f"starting that series from scratch.")
        return None
    return df if len(df) else None


def _date_set(series):
    """A column of dates as a set of datetime.date, junk dropped."""
    import pandas as pd
    return set(pd.to_datetime(series, errors="coerce").dropna().dt.date)


def _from_file_cols(df, out_cols):
    """Rename a shape CSV's columns back to the internal ntl_* names."""
    if not out_cols:
        return df
    return df.rename(columns={v: k for k, v in out_cols.items()})


def shape_dates_done(csv_path, n_tiles, out_cols=None):
    """Dates already COMPLETE in the country-shape CSV.

    Complete means every tile contributed. A date extracted while only some
    tiles were on disk has a mean computed over part of the country; skipping
    it on a later run would freeze that error into the series forever. The
    ntl_tiles column records how many tiles went into each row.

    A CSV written before ntl_tiles existed cannot be checked, so its dates are
    taken on trust and a warning is printed -- delete the file to force a
    clean rebuild.

    out_cols: the same {internal: file column} map given to update_shape_csv.
    """
    df = _read_existing(csv_path)
    if df is None or "date" not in df.columns:
        return set()
    df = _from_file_cols(df, out_cols)
    if "ntl_tiles" not in df.columns and "ntl_n" not in df.columns:
        # No count columns at all (a legacy schema such as Trinidad's
        # shape_mean/shape_sum). With ONE tile a row can only exist if that tile
        # was read, so every date is complete; with several it cannot be told.
        if n_tiles > 1:
            _note(f"NOTE: {os.path.basename(csv_path)} has no ntl_tiles/ntl_n column; "
                  f"its dates are taken on trust.")
        return _date_set(df["date"])
    if "ntl_tiles" not in df.columns:
        # Legacy CSV, written before ntl_tiles existed. Do NOT simply trust it:
        # fall back to the pixel count, which is the same evidence by another
        # route. Every complete date clips the same number of Bolivian pixels,
        # so a date well below the maximum was extracted while a tile was still
        # missing and its mean covers only part of the country.
        import pandas as pd
        n = pd.to_numeric(df["ntl_n"], errors="coerce")
        full = n.max()
        keep = n >= 0.99 * full
        short = int((~keep).sum())
        _note(f"NOTE: {os.path.basename(csv_path)} predates the ntl_tiles column. "
              f"Judging its {len(df):,} row(s) by pixel count instead"
              + (f"; {short} incomplete date(s) will be re-read and replaced."
                 if short else " -- all complete."))
        return _date_set(df.loc[keep, "date"])
    return _date_set(df.loc[df["ntl_tiles"] >= n_tiles, "date"])


def site_dates_done(csv_path, locations):
    """Dates whose site CSV rows already cover EVERY requested location.

    The same completeness test as shape_dates_done, expressed per site: a tile
    that was missing leaves its sites with no row at all for that date (
    processHD5 skips coordinates outside the tile), so the date fails this
    test and is read again.

    locations: the site names being requested this run, normalised the same
    way the notebook normalises them before building `coords`.
    """
    import pandas as pd
    want = set(locations)
    if not want:
        return set()
    df = _read_existing(csv_path)
    if df is None or not {"date", "location"}.issubset(df.columns):
        return set()
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    if not len(d):
        return set()
    covered = d.groupby(d["date"].dt.date)["location"].nunique()
    return set(covered[covered >= len(want)].index)


def pending_rasters(files, done_dates, label="extraction"):
    """The (path, date) pairs whose date is NOT already in the CSV.

    This is the filter that keeps an evicted .h5 from being rehydrated: it
    decides on the date parsed out of the FILENAME, so a skipped file is never
    opened, never read, and never pulled back from the cloud.
    """
    pending = [(p, d) for (p, d) in files if d not in done_dates]
    skipped = len(files) - len(pending)
    _note(f"{label}: {len(files):,} raster(s) in window | {skipped:,} skipped "
          f"(already extracted) | {len(pending):,} to read")
    return pending


def update_shape_csv(records, csv_path, n_tiles, out_cols=None):
    """Fold new per-file readings into the shape CSV; return the FULL table.

    Across tiles, totals and pixel counts ADD and the mean is then RECOMPUTED
    as sum / n -- never the average of the tile means. Bolivia's tiles hold
    78%, 15%, 7% and 0.3% of the country, so averaging their means would give
    a sliver of Pando the same weight as the 78% holding La Paz, Cochabamba
    and Santa Cruz.

    out_cols: {internal: file column} -- write ONLY these columns, under these
    names. Trinidad passes {"ntl_mean": "shape_mean", "ntl_sum": "shape_sum"}:
    its meta.csv registers the series under those names, and 2_data_build
    turns every column of a raw CSV into a series, so ntl_n / ntl_tiles would
    enter data.csv as unregistered predictors. None = the full ntl_* schema.

    Returns the whole series (old rows plus new) under the INTERNAL names, so
    the plots downstream see the complete history even on a run that
    extracted nothing.
    """
    import pandas as pd
    COLS = ["ntl_mean", "ntl_sum", "ntl_n", "ntl_tiles"]
    frames = []

    old = _read_existing(csv_path)
    if old is not None:
        old = _from_file_cols(old, out_cols)
    if old is not None and "date" in old.columns:
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
        old = old.dropna(subset=["date"]).set_index("date")
        if "ntl_tiles" not in old.columns:
            old["ntl_tiles"] = n_tiles          # see shape_dates_done()
        frames.append(old.reindex(columns=COLS))

    if records:
        new = pd.DataFrame(records)
        new["date"] = pd.to_datetime(new["date"])
        # size() counts the tile files that reported for each date.
        new = (new.groupby("date")
                  .agg(ntl_sum=("ntl_sum", "sum"),
                       ntl_n=("ntl_n", "sum"),
                       ntl_tiles=("ntl_sum", "size")))
        new["ntl_mean"] = new["ntl_sum"] / new["ntl_n"].replace(0, np.nan)
        frames.append(new.reindex(columns=COLS))

    if not frames:
        return pd.DataFrame(columns=COLS)

    out = pd.concat(frames)
    # A re-extracted date supersedes the stored one: `new` is concatenated
    # last, so keep="last" lets a complete re-read replace a partial row.
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "date"
    if out_cols:
        out[list(out_cols)].rename(columns=out_cols).to_csv(csv_path)
    else:
        out.to_csv(csv_path)
    return out


def update_site_csv(rows, csv_path):
    """Fold new site readings into the site CSV; return the FULL table."""
    import pandas as pd
    frames = []

    old = _read_existing(csv_path)
    if old is not None and "date" in old.columns:
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
        frames.append(old.dropna(subset=["date"]))

    if rows:
        new = pd.DataFrame(rows).rename(columns={"JD": "date"})
        new["date"] = pd.to_datetime(new["date"])
        frames.append(new)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # One row per site per date; a re-extracted date replaces the stored one.
    out = out.drop_duplicates(subset=["location", "date"], keep="last")
    out = out.sort_values(["location", "date"]).reset_index(drop=True)
    out.to_csv(csv_path, index=False)
    return out
