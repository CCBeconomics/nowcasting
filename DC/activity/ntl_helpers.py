"""
ntl_helpers.py  —  NTL (VIIRS Black Marble) download + extraction helpers.

Extracted from the baseline 1_data_ntl.ipynb so the notebook stays a thin
orchestration layer. Functions that used notebook globals now take them as
explicit arguments (api_key, path_root). Two extraction back-ends:
    processHD5        point-based (lat/lon list, Coordinates.xlsx)
    processHD5_shape  polygon/shape-based (renamed from processHD5_island)
"""
import os, re, glob, time, requests
import threading
from datetime import datetime
import numpy as np
import statistics as stat

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
    
    col = int((lon - xOrigin) / pixelWidth )
    row = int((yOrigin - lat) / pixelHeight)
    
    #Data AT THAT ROW COLUMN
    if(window == 3):
        '''
        This is a grid of 3x3 around the lat,lon. 
        '''
        indexX = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        indexY = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        newIndexX = indexX + row
        newIndexY = indexY + col
        
        Totalvalue = []
        for i in range(0, 3):
            for j in range(0, 3):
                Totalvalue.append(data[newIndexX[i][j]][newIndexY[i][j]])
                
        value = format(stat.mean(map(float, Totalvalue)),'.2f')
        return float(value)

    else:
        #print(window)
        value = data[row][col]
        return value


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


def download_many(targets, api_key, path_root, log_path=None, max_workers=8,
                  max_retries=3, retry_sleep=1.0, progress=True):
    """Download many (tile, year, day) targets CONCURRENTLY.

    Downloading is I/O-bound, so a thread pool overlaps the network waits and is
    much faster than a sequential loop. Each target is handed to download_vnp46a2
    (which retries transient errors and logs failures), so nothing raised here
    aborts the batch. `targets` is the list of (tile, year, day) tuples built in
    1_data_ntl. Lower `max_workers` if the server starts rate-limiting.

    Returns a dict of status counts: {'ok', 'unavailable', 'failed'}.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from collections import Counter

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
    rasterFilePre = os.path.basename(inputHD5)[:-3]
    ## Open HDF file
    hdflayer = gdal.Open(inputHD5, gdal.GA_ReadOnly)
    subhdflayer = hdflayer.GetSubDatasets()[layer][0]
    rlayer = gdal.Open(subhdflayer, gdal.GA_ReadOnly)
    # Layer name = last path segment of the subdataset descriptor
    # (HDF5:"<file>"://HDFEOS/.../<LayerName>). Using split(...) instead of a
    # hardcoded offset keeps the .tif name short + valid regardless of how long
    # the input .h5 path is (a fixed offset breaks on long absolute paths and can
    # blow past the Windows 260-char path limit).
    outputLayerName = subhdflayer.split("/")[-1]
    clean_layer_name = re.sub(r'[^\w\-_.]', '_', outputLayerName)    #Get File Name Prefix

    #outputFile  (os.path.join is robust to OutputFolder with/without a trailing slash)
    outputRaster = os.path.join(OutputFolder, rasterFilePre + "_" + clean_layer_name + ".tif")
    
    HorizontalTileNumber = int(rlayer.GetMetadata_Dict()["HorizontalTileNumber"])
    VerticalTileNumber = int(rlayer.GetMetadata_Dict()["VerticalTileNumber"])
    WestBoundCoord = (10*HorizontalTileNumber) - 180
    NorthBoundCoord = 90-(10*VerticalTileNumber)
    
    EastBoundCoord = WestBoundCoord + 10
    SouthBoundCoord = NorthBoundCoord - 10
    
    EPSG = "-a_srs EPSG:4326" #WGS84
    
    translateOptionText = EPSG+" -a_ullr " + str(WestBoundCoord) + " " + str(NorthBoundCoord) + " " + str(EastBoundCoord) + " " + str(SouthBoundCoord)
    translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine(translateOptionText))
    #gdal.Translate(outputRaster,rlayer, options=translateoptions)
    result = gdal.Translate(outputRaster, rlayer, options=translateoptions)
    
    raster = gdal.Open(outputRaster,gdal.GA_ReadOnly)
    
    if raster is None:
        print("Could not open image", Date)
        return []          # skip this file rather than crashing on None.GetRasterBand()

    band = raster.GetRasterBand(1)

    cols = raster.RasterXSize
    rows = raster.RasterYSize

    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]
    data = band.ReadAsArray(0, 0, cols, rows)
    
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
    
    raster = None
    # housekeeping
    #try:
    #    os.remove(outputRaster)
    #except :
    #    [ os.remove(f"{outputFolder}/{T}") for T in os.listdir(outputFolder) if ".tif" in T ]
    
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
