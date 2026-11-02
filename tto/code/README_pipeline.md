# The production pipeline (code/)

`activity/` is the teaching version of the tool: short notebooks, one idea each. `code/` is the pipeline itself, the
one run to download the data, rebuild the panel and produce the Trinidad and Tobago nowcast. It reads and writes the
same `raw/`, `data/` and `output/` folders as the activities, so a rebuild here updates the panel the activities use.

## Run order

| Step | File | What it does | Writes |
|---|---|---|---|
| 1 | `1_data_down.ipynb` | Downloads the structured series: CBTT data centre and Economic Datapack, FRED, EIA (Brent), US Census, Statistics Canada, Eurostat and UK trade | `raw/*.csv` |
| 1 | `1_data_gtrends.ipynb` | Google Trends for the keyword list in `raw/TT_GTrends_Keywords.csv` | `raw/gtrends.csv`, `data/gtrends_log.csv` |
| 1 | `1_data_ntl.ipynb` | VIIRS Black Marble night-time lights (tile h11v07): downloads the daily tiles, extracts the country and the sites in `raw/Coordinates.xlsx` (environment `geo`) | `raw/ntl_shape.csv`, `raw/ntl_data.csv` |
| 2 | `2_data_build.ipynb` | Merges every `raw/*.csv` into one monthly panel and updates the metadata | `data/data.csv`, `data/meta.csv` |
| 3 | `run_tto.py` (or `3_nwcst_tto.ipynb`) | The nowcast: seasonal adjustment, the six-screen preselection race, specifications, estimation over nine vintages, the pooled nowcast with its band and the regime watchlist | `output/<date>_tto_<run>/` |
| 4 | `4_results_tto.ipynb` | Accuracy tables, fan chart and watchlist charts from a finished run | the same output folder |

The three step-1 notebooks are independent. Each only adds new dates to what is already in `raw/`.

## Settings

| File | Role |
|---|---|
| `config.py` | Night-time lights and Google Trends: NASA token, tile, date window, keyword file, tile store (`NL_ROOT`). |
| `config_nwcst.py` | Every setting of the nowcast: data files, target, candidate screens, the preselection race, frozen groups, estimators, clusters and the watchlist rule, vintages, output folder naming. |
| `nwcst_helpers.py` | Helpers for the nowcast (vintage alignment, GDP-lag features, X-13 handling, the regime machinery). Not the same file as `activity/nwcst_helpers.py`. |
| `ntl_helpers.py` | Tile download and pixel extraction for `1_data_ntl`. |
| `run_tto_seed.py` | One seed of the estimation loop; launched by `run_tto.py`. |

**Credentials:** `1_data_ntl` needs a free NASA LAADS token in `config.py` (`NASA_TOKEN`); without one it works with the
tiles already on disk. Paste it into your own copy only: never into a file you share.

## Data

The nowcast reads `data/data.csv` and `data/meta.csv` (set in `config_nwcst.py`). As shipped, that is the October 2026
panel: 178 months (January 2012 to October 2026), 638 series. The target is `RGDP0000`, real GDP by the production approach,
quarterly, published about six months after the quarter ends. `meta.csv` holds each series' publication lag
(`months_lag`), which decides what every vintage can see, and the `nominal` and `sa` flags used for deflation (CPI
`RCPI0000`, base 2018) and seasonal adjustment.

The results presented in the workshop were estimated on the same panel before the night-time lights correction of
22 October 2026 (13 `UNTL*` site series, August 2025 to April 2026), so a rerun moves the pooled numbers somewhat.

## Running the nowcast

From `code/`, with the `nwcst` environment active (scikit-learn 1.8: the orchestrator refuses other versions, because
LARS output differs across versions and the frozen groups were identified on 1.8):

```
TTO_RUN_TAG=ident python run_tto.py      # identification run: test window from 2019-Q4, COVID as a test quarter, no live readings
TTO_RUN_TAG=prod  python run_tto.py      # production run: test window from 2023-Q2, live quarters
```

Environment switches (all optional): `TTO_LAUNCH_DATE_STR=YYYYMMDD` fixes the output folder's date; `TTO_SA_METHOD=x13|classical`
(default `x13`; the X-13 binary comes from the conda-forge package `x13as`, and without it every series takes the classical
decomposition automatically); `TTO_OUT_SUFFIX` names a variant run; `TTO_SPEC_FILTER` runs a subset of specifications.
Three seeds in parallel take several hours on a laptop (about 1,030 cells per seed at nine vintages). Outputs go to
`output/<date>_tto_<run><suffix>/`; logs and checkpoints to `output/logs/` and `output/checkpoints/`. Then run
`4_results_tto.ipynb` with `TTO_RUN_TAG` set to the folder's tag for the tables and charts.

The method is described in the slides `s5_methodology.pdf` (the pipeline) and `theory_5_preselection.pdf` (the six screens).
