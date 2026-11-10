# The nowcast pipeline (pipeline/)

The production tool behind the Trinidad and Tobago GDP nowcasts. It sits beside the data notebooks (`code/`) and reads
the panel they build in `data/` (`data.csv`, `meta.csv`, named in `config.py` section 1); it writes to
`output/<date>_tto_<run>/`.

| File | Role |
|---|---|
| `config.py` | Every knob, seven annotated sections, each naming the notebook cell that reads it. The only file to edit. |
| `3_nwcst_tto.ipynb` | The method: data, filters, seasonal adjustment, the six-screen race, groups, specifications, estimation loop, post-estimation, live quarters. Tagged cells (`# [tto:<name>]`), each opened by a header stating what it reads and writes. |
| `nwcst_engine.py` | The estimation of one specification (six stages) and the estimator registry. |
| `nwcst_helpers.py` | Screens, estimators, cross-validation, seasonal adjustment, regime machinery. Not the same file as `activity/nwcst_helpers.py`. |
| `run_tto.py`, `run_tto_seed.py` | Orchestrator (setup once, three seeds, merge, post-estimation, charts) and the per-seed runner. |
| `4_results_tto.ipynb` | Tables and charts from a finished run. |
| `tests/test_engine.py` | Engine identity test: re-estimates four specifications and compares them with a reference run. |

## Data

The nowcast reads `data/data.csv` and `data/meta.csv`. As shipped, that is the October 2026 panel: 178 months
(January 2012 to October 2026), 638 series. The target is `RGDP0000`, real GDP by the production approach, quarterly,
published about six months after the quarter ends. `meta.csv` holds each series' publication lag (`months_lag`), which
decides what every vintage can see, and the `nominal` and `sa` flags used for deflation (CPI `RCPI0000`, base 2018) and
seasonal adjustment. Running `code/2_data_build.ipynb` rebuilds both files.

The results presented in the workshop were estimated on the same panel before the night-time lights correction of
22 October 2026 (13 `UNTL*` site series, August 2025 to April 2026), so a rerun moves the pooled numbers somewhat.

## Running the nowcast

From `pipeline/`, with the `nwcst` environment active. The scripts need scikit-learn 1.8 and refuse other versions,
because LARS output differs across versions and the frozen groups were identified on 1.8. X-13ARIMA-SEATS comes from the
conda-forge package `x13as`; without it every series takes the classical decomposition automatically.

```
TTO_RUN_TAG=ident python run_tto.py     # identification run: test window 2019-Q4 to 2022-Q4, COVID as a test quarter
TTO_RUN_TAG=prod  python run_tto.py     # production run: test window from 2023-Q2, live quarters after the last print
```

On Windows (Anaconda Prompt): `set TTO_RUN_TAG=prod` and then `python run_tto.py`.

The nine setup cells run once (about 15 minutes), then three seeds run in parallel; a full run takes several hours on a
laptop. Outputs go to `output/<date>_tto_<run>/`, one folder per seed beside it, logs to `output/logs/` and checkpoints
to `output/checkpoints/`. A stopped run resumes from its checkpoint when relaunched with the same date.

Optional environment switches: `TTO_LAUNCH_DATE_STR` fixes the output folder's date; `TTO_SA_METHOD=x13|classical`;
`TTO_SEEDS=42` runs fewer seeds; `TTO_SPEC_FILTER` (a regular expression on specification names, e.g. `^(human1|fshap6)$`)
runs a subset of specifications. The notebook can also be run cell by cell; it reads `TTO_RUN_TAG` from the environment.

The method is described in the slides `s5_methodology.pdf` (the pipeline) and `theory_5_preselection.pdf` (the six screens).
