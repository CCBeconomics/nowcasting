"""
Trinidad single-seed trainer. Takes the seed as CLI arg.
Writes to ../output/{date}_tto_{tag}_s{seed}/. Launched by run_tto.py (three seeds in parallel); usable standalone.

Usage:  TTO_RUN_TAG=ident python run_tto_seed.py 42 [n_jobs]      (nwcst env active)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"   # one core per worker
import sys, json, re, warnings, time, pickle
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sklearn
if not sklearn.__version__.startswith("1.8"):
    sys.stderr.write(f"[run_tto_seed] ABORT: sklearn=={sklearn.__version__}; use the nwcst env.\n"); sys.exit(2)
import matplotlib; matplotlib.use('Agg')
if len(sys.argv) < 2:
    sys.stderr.write("Usage: run_tto_seed.py <seed> [n_jobs]\n"); sys.exit(2)
SEED = int(sys.argv[1]); N_JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SPEC_FILTER = os.environ.get("TTO_SPEC_FILTER")      # regex on spec names: run only these (partial runs, e.g. the frozen groups)
NB_PATH = '3_nwcst_tto.ipynb'

def _cells(nb):
    out = {}
    for c in nb['cells']:
        if c.get('cell_type') != 'code': continue
        src = ''.join(c['source']); m = re.match(r"# \[tto:(\w+)\]", src)
        if m: out[m.group(1)] = src.replace('plt.show()', 'plt.close()')
    return out
with open(NB_PATH) as f: CELLS = _cells(json.load(f))
SETUP = ["setup", "data", "filters", "helpers", "presel_panel", "presel_race", "presel_groups", "presel_shock", "specs"]   # the loop cell is not executed: the engine is imported

CTX = os.environ.get("TTO_CONTEXT")     # hand-off written by run_tto.py's setup pass; absent = standalone run
t0 = time.time()
if CTX and os.path.exists(CTX):
    print(f"[{time.strftime('%H:%M:%S')}] [seed {SEED}] setup cells ['setup', 'data', 'helpers'] + context {CTX} ...", flush=True)
    for tag in ["setup", "data", "helpers"]:          # paths, constants, pool_mask/reading_guard, imports (seconds)
        try: exec(CELLS[tag])
        except Exception as e:
            print(f"[cell {tag}] FAILED: {e}", flush=True); import traceback; traceback.print_exc(); sys.exit(1)
    with open(CTX, "rb") as f: _ctx = pickle.load(f)
    globals().update(_ctx)                             # data, metadata (processed), spec_list, rolling_selections, benchmark, ...
    print(f"[seed {SEED}] context loaded: {len(spec_list)} specs, panel {data.shape}, benchmark {benchmark:.3f}", flush=True)
else:
    print(f"[{time.strftime('%H:%M:%S')}] [seed {SEED}] setup cells {SETUP} ...", flush=True)
    for tag in SETUP:
        try: exec(CELLS[tag])
        except Exception as e:
            print(f"[cell {tag}] FAILED: {e}", flush=True); import traceback; traceback.print_exc(); sys.exit(1)

# the engine: the notebook's loop cell and this runner call the same nwcst_engine.run_one_spec
from nwcst_engine import run_one_spec, make_context, make_opts, make_rows
ctx = make_context(data, metadata, target_variable, lags, run, train_start_date, test_start_date, test_end_date,
                   _iter_, cfg.ITER_LSTM, cfg.EPOCH_LSTM, desired_date)
opts_seed = dict(make_opts(cfg, data, metadata, target_variable), seed=SEED)
rows = make_rows(spec_list, rolling_selections)
if SPEC_FILTER:
    rows = [r for r in rows if re.search(SPEC_FILTER, str(r["spec"]))]
    print(f"[seed {SEED}] spec filter {SPEC_FILTER!r}: {len(rows)} specs kept", flush=True)
SPATH = os.path.join(os.path.dirname(PATH_OUTPUT), f"{cfg.OUTPUT_SUBFOLDER}_s{SEED}"); os.makedirs(SPATH, exist_ok=True)
print(f"[{time.strftime('%H:%M:%S')}] [seed {SEED}] setup done in {time.time()-t0:.0f}s | specs={len(rows)} | run={cfg.RUN_TAG} | SPATH={SPATH} | n_jobs={N_JOBS}", flush=True)

import pandas as pd
from joblib import Parallel, delayed
# Checkpoint keyed by run, launch date and seed; a spec counts as done only if its name AND its variable list match
CKPT = os.path.join("..", "output", "checkpoints", f"tto_{cfg.RUN_TAG}_{date_str}_s{SEED}_checkpoint.pkl"); os.makedirs(os.path.dirname(CKPT), exist_ok=True)
_norm = lambda v: " ".join(sorted(str(v).split()))          # the variable list as a sorted string: the group order may vary between passes
_sig_by_name = {r["spec"]: _norm(r["selected_vars"]) for r in rows}
if os.path.exists(CKPT):
    with open(CKPT, 'rb') as f: done_specs, perf_parts, fcst_parts, oos_parts, loo_parts = pickle.load(f)
    if done_specs and isinstance(next(iter(done_specs)), str):   # legacy cache keyed by name only -> upgrade to (name, variables)
        done_specs = {(n, _sig_by_name.get(n)) for n in done_specs}
    done_specs = {(n, _norm(v)) for n, v in done_specs}
    print(f"[seed {SEED}] resuming from {os.path.basename(CKPT)}: {len(done_specs)} specs already done", flush=True)
else:
    done_specs, perf_parts, fcst_parts, oos_parts, loo_parts = set(), [], [], [], []
rows_todo = [r for r in rows if (r['spec'], _norm(r['selected_vars'])) not in done_specs]
pd.DataFrame([{k: v for k, v in r.items() if k != "rolling_selections"} for r in rows]).to_csv(f"{SPATH}/{date_str} 3_nwcst Tab spec_rows.csv", index=False)   # persist the spec list
print(f"[seed {SEED}] {len(rows_todo)} specs to process", flush=True)

def write_xlsx(df, path, **kw):
    """Write an Excel table through a local temporary file, then move it into place (retried three times). A direct
    write into a folder watched by a sync client can fail with 'Operation timed out' on large files."""
    import tempfile, shutil, time as _time
    for attempt in range(3):
        tmp = os.path.join(tempfile.gettempdir(), f"nwcst_{os.getpid()}_{attempt}.xlsx")
        try:
            df.to_excel(tmp, **kw); shutil.move(tmp, path); return
        except OSError as e:
            if os.path.exists(tmp): os.remove(tmp)
            if attempt == 2: raise
            print(f"[write_xlsx] retry {attempt + 1} for {os.path.basename(path)}: {e}", flush=True); _time.sleep(5)

def _save():
    try:
        if perf_parts: write_xlsx(pd.concat(perf_parts, axis=0), f"{SPATH}/{date_str} 3_nwcst Tab perf.xlsx")
        if fcst_parts:
            tf = pd.concat(fcst_parts, axis=0); tf.index = pd.to_datetime(tf.index).strftime("%Y-%m-%d")
            write_xlsx(tf, f"{SPATH}/{date_str} 3_nwcst Tab fcst.xlsx", index_label="date")
        if oos_parts:
            to_ = pd.concat(oos_parts, axis=0); to_.index = pd.to_datetime(to_.index).strftime("%Y-%m-%d")
            write_xlsx(to_, f"{SPATH}/{date_str} 3_nwcst Tab oos.xlsx", index_label="date")
        if loo_parts: write_xlsx(pd.DataFrame(loo_parts), f"{SPATH}/{date_str} 3_nwcst Tab shock_residuals.xlsx", index=False)
    except Exception as e: print(f"[seed {SEED} SAVE ERROR] {e}", flush=True)
def _ckpt():
    with open(CKPT, 'wb') as f: pickle.dump((done_specs, perf_parts, fcst_parts, oos_parts, loo_parts), f)

t_seed = time.time(); completed = 0; BATCH = 5
for i in range(0, len(rows_todo), BATCH):
    batch = rows_todo[i:i + BATCH]
    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
        delayed(run_one_spec)(r, opts_seed, ctx) for r in batch)
    for spec, perf_df, fcst_df, oos_df, loo_list, err in results:
        completed += 1
        if err:
            print(f"[seed {SEED}][{completed}/{len(rows_todo)}] {spec} FAILED\n{err}", flush=True); done_specs.add((spec, _sig_by_name.get(spec))); continue
        if perf_df is not None and not perf_df.empty: perf_parts.append(perf_df)
        if fcst_df is not None and not fcst_df.empty: fcst_parts.append(fcst_df)
        if oos_df is not None and not oos_df.empty: oos_parts.append(oos_df)
        if loo_list: loo_parts.extend(loo_list)
        done_specs.add((spec, _sig_by_name.get(spec)))
        elapsed = time.time() - t_seed; rate = completed / elapsed if elapsed > 0 else 0
        eta = (len(rows_todo) - completed) / rate / 60 if rate > 0 else 0
        print(f"[seed {SEED}][{completed}/{len(rows_todo)}] {spec} OK | ETA {eta:.1f}min", flush=True)
    _ckpt(); _save(); print(f"[seed {SEED}][CHECKPOINT] {len(done_specs)} specs done", flush=True)
_ckpt(); _save()
print(f"[{time.strftime('%H:%M:%S')}] [seed {SEED}] DONE in {(time.time()-t_seed)/60:.1f}min", flush=True)
