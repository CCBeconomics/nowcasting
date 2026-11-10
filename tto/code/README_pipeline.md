# The production pipeline: data (code/) and nowcast (pipeline/)

`activity/` is the teaching version of the tool: short notebooks, one idea each. `code/` and `pipeline/` are the tool
itself, the one run to download the data, rebuild the panel and produce the Trinidad and Tobago nowcast. `code/` holds the
data notebooks; `pipeline/` holds the nowcast (see `pipeline/README.md`). Both read and write the same `raw/`, `data/`
and `output/` folders as the activities, so a rebuild here updates the panel the activities use.

## Run order

| Step | File | What it does | Writes |
|---|---|---|---|
| 1 | `code/1_data_down.ipynb` | Downloads the structured series: CBTT data centre and Economic Datapack, FRED, EIA (Brent), US Census, Statistics Canada, Eurostat and UK trade | `raw/*.csv` |
| 1 | `code/1_data_gtrends.ipynb` | Google Trends for the keyword list in `raw/TT_GTrends_Keywords.csv` | `raw/gtrends.csv`, `data/gtrends_log.csv` |
| 1 | `code/1_data_ntl.ipynb` | VIIRS Black Marble night-time lights (tile h11v07): downloads the daily tiles, extracts the country and the sites in `raw/Coordinates.xlsx` (environment `geo`) | `raw/ntl_shape.csv`, `raw/ntl_data.csv` |
| 2 | `code/2_data_build.ipynb` | Merges every `raw/*.csv` into one monthly panel and updates the metadata | `data/data.csv`, `data/meta.csv` |
| 3 | `pipeline/run_tto.py` (or `pipeline/3_nwcst_tto.ipynb`) | The nowcast: seasonal adjustment, the six-screen preselection race, specifications, estimation over nine vintages, the pooled nowcast with its band and the regime watchlist | `output/<date>_tto_<run>/` |
| 4 | `pipeline/4_results_tto.ipynb` | Accuracy tables, fan chart and watchlist charts from a finished run (`run_tto.py` runs it at the end) | the same output folder |

The three step-1 notebooks are independent. Each only adds new dates to what is already in `raw/`.

## Settings of the data notebooks

| File | Role |
|---|---|
| `config.py` | Night-time lights and Google Trends: NASA token, tile, date window, keyword file, tile store (`NL_ROOT`). |
| `ntl_helpers.py` | Tile download and pixel extraction for `1_data_ntl`. |

The settings of the nowcast are in `pipeline/config.py`.

**Credentials:** `1_data_ntl` needs a free NASA LAADS token in `config.py` (`NASA_TOKEN`); without one it works with the
tiles already on disk. Paste it into your own copy only: never into a file you share.
