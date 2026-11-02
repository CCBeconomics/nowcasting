"""
Trinidad single-seed trainer (port of Jamaica's run_beta16_seed.py, 2026-10-24). Takes the seed as CLI arg.
Writes to ../output/{date}_tto_{tag}_s{seed}/. Launched by run_tto.py (three seeds in parallel); usable standalone.

Usage:  TTO_RUN_TAG=ident python run_tto_seed.py 42 [n_jobs]      (from tto/pipeline/, nwcst env active)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"   # one core per worker (Jamaica 2026-10-20)
import sys, json, re, warnings, time, pickle
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sklearn
if not sklearn.__version__.startswith("1.8"):
    sys.stderr.write(f"[run_tto_seed] ABORT: sklearn=={sklearn.__version__}; use the nowcst env.\n"); sys.exit(2)
import matplotlib; matplotlib.use('Agg')
if len(sys.argv) < 2:
    sys.stderr.write("Usage: run_tto_seed.py <seed> [n_jobs]\n"); sys.exit(2)
SEED = int(sys.argv[1]); N_JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SPEC_FILTER = os.environ.get("TTO_SPEC_FILTER")      # regex on spec names: run only these (partial runs, e.g. the frozen groups, 2026-10-25)
OUT_SUFFIX  = os.environ.get("TTO_OUT_SUFFIX", "")   # keeps a partial run's output folders and checkpoints apart from the full run
NB_PATH = '3_nwcst_tto.ipynb'

def _cells(nb):
    out = {}
    for c in nb['cells']:
        if c.get('cell_type') != 'code': continue
        src = ''.join(c['source']); m = re.match(r"# \[tto:(\w+)\]", src)
        if m: out[m.group(1)] = src.replace('plt.show()', 'plt.close()')
    return out
with open(NB_PATH) as f: CELLS = _cells(json.load(f))
SETUP = ["setup", "data", "filters", "helpers", "preselection", "specs"]

print(f"[{time.strftime('%H:%M:%S')}] [seed {SEED}] setup cells {SETUP} ...", flush=True); t0 = time.time()
for tag in SETUP:
    try: exec(CELLS[tag])
    except Exception as e:
        print(f"[cell {tag}] FAILED: {e}", flush=True); import traceback; traceback.print_exc(); sys.exit(1)

# the loop cell: everything before the orchestration (imports, opts, rows, the worker), nothing after
_loop = CELLS["loop"]; _cut = _loop.index("perf_out, fcst_out, oos_out, loo_out = [], [], [], []")
exec(_loop[:_cut])
opts_seed = dict(opts, seed=SEED)
if SPEC_FILTER:
    rows = [r for r in rows if re.search(SPEC_FILTER, str(r["spec"]))]
    print(f"[seed {SEED}] spec filter {SPEC_FILTER!r}: {len(rows)} specs kept", flush=True)
SPATH = os.path.join(os.path.dirname(PATH_OUTPUT), f"{cfg.OUTPUT_SUBFOLDER}_s{SEED}"); os.makedirs(SPATH, exist_ok=True)   # config.OUTPUT_SUBFOLDER already carries TTO_OUT_SUFFIX (2026-10-26 fix)
print(f"[{time.strftime('%H:%M:%S')}] [seed {SEED}] setup done in {time.time()-t0:.0f}s | specs={len(rows)} | run={cfg.RUN_TAG} | SPATH={SPATH} | n_jobs={N_JOBS}", flush=True)

import pandas as pd
from joblib import Parallel, delayed
CKPT = os.path.join("..", "output", "checkpoints", f"tto_{cfg.RUN_TAG}{OUT_SUFFIX}_s{SEED}_checkpoint.pkl"); os.makedirs(os.path.dirname(CKPT), exist_ok=True)
if os.path.exists(CKPT):
    with open(CKPT, 'rb') as f: done_specs, perf_parts, fcst_parts, oos_parts, loo_parts = pickle.load(f)
    print(f"[seed {SEED}] resuming: {len(done_specs)} specs already done", flush=True)
else:
    done_specs, perf_parts, fcst_parts, oos_parts, loo_parts = set(), [], [], [], []
rows_todo = [r for r in rows if r['spec'] not in done_specs]
print(f"[seed {SEED}] {len(rows_todo)} specs to process", flush=True)

def _save():
    try:
        if perf_parts: pd.concat(perf_parts, axis=0).to_excel(f"{SPATH}/{date_str} 3_nwcst Tab perf.xlsx")
        if fcst_parts:
            tf = pd.concat(fcst_parts, axis=0); tf.index = pd.to_datetime(tf.index).strftime("%Y-%m-%d")
            tf.to_excel(f"{SPATH}/{date_str} 3_nwcst Tab fcst.xlsx", index_label="date")
        if oos_parts:
            to_ = pd.concat(oos_parts, axis=0); to_.index = pd.to_datetime(to_.index).strftime("%Y-%m-%d")
            to_.to_excel(f"{SPATH}/{date_str} 3_nwcst Tab oos.xlsx", index_label="date")
        if loo_parts: pd.DataFrame(loo_parts).to_excel(f"{SPATH}/{date_str} 3_nwcst Tab shock_residuals.xlsx", index=False)
    except Exception as e: print(f"[seed {SEED} SAVE ERROR] {e}", flush=True)
def _ckpt():
    with open(CKPT, 'wb') as f: pickle.dump((done_specs, perf_parts, fcst_parts, oos_parts, loo_parts), f)

t_seed = time.time(); completed = 0; BATCH = 5
for i in range(0, len(rows_todo), BATCH):
    batch = rows_todo[i:i + BATCH]
    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
        delayed(_run_one_spec_worker)(r, opts_seed, data, metadata, target_variable, lags, run,
                                      train_start_date, test_start_date, test_end_date,
                                      _iter_, _iter_lstm, _epoch_lstm, desired_date) for r in batch)
    for spec, perf_df, fcst_df, oos_df, loo_list, err in results:
        completed += 1
        if err:
            print(f"[seed {SEED}][{completed}/{len(rows_todo)}] {spec} FAILED\n{err}", flush=True); done_specs.add(spec); continue
        if perf_df is not None and not perf_df.empty: perf_parts.append(perf_df)
        if fcst_df is not None and not fcst_df.empty: fcst_parts.append(fcst_df)
        if oos_df is not None and not oos_df.empty: oos_parts.append(oos_df)
        if loo_list: loo_parts.extend(loo_list)
        done_specs.add(spec)
        elapsed = time.time() - t_seed; rate = completed / elapsed if elapsed > 0 else 0
        eta = (len(rows_todo) - completed) / rate / 60 if rate > 0 else 0
        print(f"[seed {SEED}][{completed}/{len(rows_todo)}] {spec} OK | ETA {eta:.1f}min", flush=True)
    _ckpt(); _save(); print(f"[seed {SEED}][CHECKPOINT] {len(done_specs)} specs done", flush=True)
_ckpt(); _save()
print(f"[{time.strftime('%H:%M:%S')}] [seed {SEED}] DONE in {(time.time()-t_seed)/60:.1f}min", flush=True)
