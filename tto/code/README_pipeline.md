# The production pipeline (code/)

`activity/` is the teaching version of the tool: short notebooks, one idea each. `code/` is the pipeline itself, the one
run to download the data, rebuild the panel and produce the Trinidad and Tobago nowcast. It reads and writes the same
`raw/`, `data/` and `output/` folders as the activities, so a rebuild here updates the panel the activities use.

## Run order

| Step | File | What it does | Writes |
|---|---|---|---|
| 1 | `1_data_down.ipynb` | Downloads the structured series: CBTT data centre and Economic Datapack, FRED, EIA (Brent), US Census, Statistics Canada, Eurostat and UK trade | `raw/*.csv` |
| 1 | `1_data_gtrends.ipynb` | Google Trends for the keyword list in `raw/TT_GTrends_Keywords.csv` | `raw/gtrends.csv`, `data/gtrends_log.csv` |
| 1 | `1_data_ntl.ipynb` | VIIRS Black Marble night-time lights (tile h11v07): downloads the daily tiles, extracts the country and the sites in `raw/Coordinates.xlsx` (environment `geo`) | `raw/ntl_shape.csv`, `raw/ntl_data.csv` |
| 2 | `2_data_build.ipynb` | Merges every `raw/*.csv` into one monthly panel and updates the metadata | `data/data.csv`, `data/meta.csv` |
| 3 | `run_tto.py` (or `3_nwcst_tto.ipynb`) | The nowcast: seasonal adjustment, the six-screen preselection race, specifications, estimation over nine vintages, the pooled nowcast with its band and the regime watchlist | `output/<date>_tto_<run>/` |
| 4 | `4_results_tto.ipynb` | Accuracy tables, fan chart and watchlist charts from a finished run (`run_tto.py` runs it at the end) | the same output folder |
| 5 (optional) | `5_timeseries_models.R` | Classical mixed-frequency benchmarks in R on the same panel and sample: DFM, MIDAS, MF-BVAR and MIDAS-ML | `output/<date>_tto_timeseries/` |

The three step-1 notebooks are independent. Each only adds new dates to what is already in `raw/`.

## Files

| File | Role |
|---|---|
| `config_data.py` | Settings of the data notebooks (steps 1 and 2): NASA token, night-time lights tile, date window, Google Trends keyword file, tile store (`NL_ROOT`). |
| `ntl_helpers.py` | Tile download and pixel extraction for `1_data_ntl`. |
| `config.py` | Every setting of the nowcast (steps 3 and 4), seven annotated sections, each naming the notebook cell that reads it: data files and sample, redundancy rules, the preselection race, groups outside the race, estimators, pool and watchlist, run settings. |
| `3_nwcst_tto.ipynb` | The method, in tagged cells (`# [tto:<name>]`), each opened by a header stating what it reads and writes. |
| `nwcst_engine.py` | The estimation of one specification (six stages) and the estimator registry. |
| `nwcst_helpers.py` | Screens, estimators, cross-validation, seasonal adjustment, regime machinery. Not the same file as `activity/nwcst_helpers.py`. |
| `run_tto.py`, `run_tto_seed.py` | Orchestrator (setup once, three seeds in parallel, merge, post-estimation, charts) and the per-seed runner. |
| `5_timeseries_models.R` | Optional R benchmarks: DFM, MIDAS, MF-BVAR, MIDAS-ML (see below). |

**Credentials:** `1_data_ntl` needs a free NASA LAADS token in `config_data.py` (`NASA_TOKEN`); without one it works with
the tiles already on disk. Paste it into your own copy only: never into a file you share.

## Data

The nowcast reads `data/data.csv` and `data/meta.csv` (set in `config.py`, section 1). As shipped, that is the October 2026
panel: 178 months (January 2012 to October 2026), 638 series. The target is `RGDP0000`, real GDP by the production approach,
quarterly, published about six months after the quarter ends. `meta.csv` holds each series' publication lag
(`months_lag`), which decides what every vintage can see, and the `nominal` and `sa` flags used for deflation (CPI
`RCPI0000`, base 2018) and seasonal adjustment.

The results presented in the workshop were estimated on the same panel before the night-time lights correction of
22 October 2026 (13 `UNTL*` site series, August 2025 to April 2026), so a rerun moves the pooled numbers somewhat.

## Running the nowcast

From `code/`, with the `nwcst` environment active. The scripts need scikit-learn 1.8 and refuse other versions, because
LARS output differs across versions and the frozen groups were identified on 1.8. X-13ARIMA-SEATS comes from the
conda-forge package `x13as`; without it every series takes the classical decomposition automatically.

```
TTO_RUN_TAG=ident python run_tto.py     # identification run: test window 2019-Q4 to 2022-Q4, COVID as a test quarter
TTO_RUN_TAG=prod  python run_tto.py     # production run: test window from 2023-Q2, live quarters after the last print
```

On Windows (Anaconda Prompt): `set TTO_RUN_TAG=prod`, then `python run_tto.py`.

The nine setup cells run once (about 15 minutes), then three seeds run in parallel; a full run takes several hours on a
laptop. Outputs go to `output/<date>_tto_<run>/`, one folder per seed beside it, logs to `output/logs/` and checkpoints
to `output/checkpoints/`. A stopped run resumes from its checkpoint when relaunched with the same date.

Optional environment switches: `TTO_LAUNCH_DATE_STR` fixes the output folder's date; `TTO_SA_METHOD=x13|classical`;
`TTO_SEEDS=42` runs fewer seeds; `TTO_SPEC_FILTER` (a regular expression on specification names, e.g. `^(human1|fshap6)$`)
runs a subset of specifications. The notebook can also be run cell by cell; it reads `TTO_RUN_TAG` from the environment.

The method is described in the slides `s5_methodology.pdf` (the pipeline) and `theory_5_preselection.pdf` (the six screens).

## Time-series benchmarks in R (step 5, optional)

`5_timeseries_models.R` estimates the models of the time-series theory block (`theory_4_timeseries.pdf`) on
`data/data.csv` and `data/meta.csv`, with the nowcast's training and test windows (training from 2013-Q1, test from
2023-Q2). It adjusts each series with the classical decomposition, takes growth rates, and then estimates:

- **DFM** (`nowcastDFM`): one monthly factor behind 15 indicators, Kalman-filter nowcasts; also nowcasts the quarters after the last GDP print.
- **MIDAS** (`midasr`): GDP on six monthly lags of one indicator, one model per indicator, nowcasts averaged.
- **MF-BVAR** (`mfbvar`): a small monthly VAR with GDP, Minnesota prior, re-estimated every test quarter.
- **MIDAS-ML** (`midasml`): sparse-group LASSO over every indicator and its monthly lags.

The indicator lists, lags and dates are in the SETTINGS block at the top of the script. From `code/`:

```
Rscript 5_timeseries_models.R
```

It takes about a minute and writes the test nowcasts, the test RMSE of each model against the training mean, the DFM
live nowcasts and a chart to `output/<date>_tto_timeseries/`. These models are a benchmark beside the pipeline: they
do not enter its pooled nowcast. The packages are those of the R environment check (`install/act0_env_check_R.Rmd`);
`mfbvar` needs Rtools on Windows; `midasml` is optional (`install.packages("midasml")`), and MIDAS-ML is skipped without it.
