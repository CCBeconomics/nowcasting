"""
Trinidad orchestrator: three seeds in parallel, merge, post-estimation
cells on the merged pool, charts if 4_results_tto.ipynb exists. The run (ident / prod) is chosen by TTO_RUN_TAG.

  TTO_RUN_TAG=ident python run_tto.py     # identification run (nwcst env active)
  TTO_RUN_TAG=prod  python run_tto.py     # production run

The nine setup cells (data, filters, seasonal adjustment, race, groups, shock features, specs) run ONCE
here, before the seeds; their result is written to <merged>/_context/ (race_context.pkl for the seeds, spec_list.csv,
rolling_rankings.json and a copy of config.py for people) and the seeds load it through TTO_CONTEXT instead of recomputing
it. Optional: TTO_SEEDS=42,43 overrides config.SEEDS (tests); TTO_SPEC_FILTER passes through to the seeds.
"""
import sys, os, json, re, warnings, time, shutil, subprocess
from datetime import date as _date
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sklearn
if not sklearn.__version__.startswith("1.8"):
    sys.stderr.write(f"[run_tto] ABORT: sklearn=={sklearn.__version__}; use the nwcst env.\n"); sys.exit(2)
RUN_TAG = os.environ.get("TTO_RUN_TAG", "prod")
import importlib.util
_spec = importlib.util.spec_from_file_location("config", "config.py"); cfg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cfg)
SEEDS = [int(x) for x in os.environ["TTO_SEEDS"].split(",")] if os.environ.get("TTO_SEEDS") else list(cfg.SEEDS); N_JOBS_PER_SEED = 3
PY = sys.executable                                   # the interpreter running this script (the nwcst env)
today = _date.today(); LOCK_DATE_STR = os.environ.get("TTO_LAUNCH_DATE_STR") or f"{today.year:04d}{today.month:02d}{today.day:02d}"
SUB = f"{LOCK_DATE_STR}_tto_{RUN_TAG}"
print(f"[{time.strftime('%H:%M:%S')}] Trinidad orchestrator | run {RUN_TAG} | seeds {SEEDS} | date_str {LOCK_DATE_STR}", flush=True)

# ---- setup pass: the nine setup cells once, then the hand-off for the seeds ----
import matplotlib; matplotlib.use('Agg')
import pickle
nb = json.load(open('3_nwcst_tto.ipynb'))
CELLS = {}
for c in nb['cells']:
    if c.get('cell_type') == 'code':
        src = ''.join(c['source']); m = re.match(r"# \[tto:(\w+)\]", src)
        if m: CELLS[m.group(1)] = src.replace('plt.show()', 'plt.close()')
os.environ["TTO_LAUNCH_DATE_STR"] = LOCK_DATE_STR
t_setup = time.time()
for tag in ["setup", "data", "filters", "helpers", "presel_panel", "presel_race", "presel_groups", "presel_shock", "specs"]:
    try: exec(CELLS[tag])
    except Exception as e: print(f"[cell {tag}] FAILED during setup: {e}", flush=True); sys.exit(1)
CTX_DIR = os.path.join("..", "output", SUB, "_context"); os.makedirs(CTX_DIR, exist_ok=True)
CTX_KEYS = ["data", "metadata", "spec_list", "rolling_selections", "benchmark", "broken_vars", "pre_selected_columns_monthly", "SHOCK_SHARE", "_vars"]
_ns = globals(); _missing = [k for k in CTX_KEYS if k not in _ns]
if _missing: print(f"[context] ABORT: setup did not define {_missing}", flush=True); sys.exit(1)
CTX_PATH = os.path.abspath(os.path.join(CTX_DIR, "race_context.pkl"))
with open(CTX_PATH, "wb") as f: pickle.dump({k: _ns[k] for k in CTX_KEYS}, f)
try:                                                   # human-readable copies; the pickle above is what the seeds need
    spec_list.to_csv(os.path.join(CTX_DIR, "spec_list.csv"), index=False)
    json.dump({str(m): {str(d): [str(x) for x in v] for d, v in sel.items()} if isinstance(sel, dict) else str(sel)
               for m, sel in rolling_selections.items()}, open(os.path.join(CTX_DIR, "rolling_rankings.json"), "w"))
    shutil.copy2("config.py", os.path.join(CTX_DIR, "config_at_launch.py"))
except Exception as e: print(f"[context] side files not written: {type(e).__name__}: {e}", flush=True)
print(f"[{time.strftime('%H:%M:%S')}] setup done in {(time.time()-t_setup)/60:.1f}min | {len(spec_list)} specs, panel {data.shape}, benchmark {benchmark:.3f} | context {CTX_PATH}", flush=True)

procs = []
for s in SEEDS:
    os.makedirs(os.path.join("..", "output", "logs"), exist_ok=True)
    log_path = os.path.join("..", "output", "logs", f"tto_{RUN_TAG}_s{s}.log")   # one log per seed
    env = os.environ.copy(); env["TTO_LAUNCH_DATE_STR"] = LOCK_DATE_STR; env["TTO_RUN_TAG"] = RUN_TAG; env["TTO_CONTEXT"] = CTX_PATH
    f = open(log_path, "w")
    p = subprocess.Popen([PY, "run_tto_seed.py", str(s), str(N_JOBS_PER_SEED)], stdout=f, stderr=subprocess.STDOUT, env=env)
    procs.append((s, p, f, log_path)); print(f"  [seed {s}] launched PID={p.pid}  log={log_path}", flush=True)
t_wait = time.time()
while any(p.poll() is None for (_, p, _, _) in procs):
    time.sleep(30); statuses = []
    for s, p, _, log_path in procs:
        if p.poll() is None:
            try:
                chks = [ln for ln in open(log_path).readlines() if "CHECKPOINT" in ln]
                statuses.append(f"s{s}:{chks[-1].strip().split(']')[-1].strip()}" if chks else f"s{s}:starting")
            except Exception: statuses.append(f"s{s}:?")
        else: statuses.append(f"s{s}:done(rc={p.returncode})")
    print(f"  [{time.strftime('%H:%M:%S')}] elapsed {(time.time()-t_wait)/60:.0f}min | " + "  ".join(statuses), flush=True)
for _, p, f, _ in procs: f.close()
for s, p, _, log_path in procs:
    if p.returncode != 0: print(f"  [seed {s}] FAILED (rc={p.returncode}) -- see {log_path}", flush=True)

# ---- merge per-seed folders ----
import pandas as pd
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
OUT = os.path.join("..", "output"); MERGED = os.path.join(OUT, SUB); os.makedirs(MERGED, exist_ok=True)
seed_paths = {s: os.path.join(OUT, f"{SUB}_s{s}") for s in SEEDS}
for tabname, index_col in [(f"{LOCK_DATE_STR} 3_nwcst Tab fcst.xlsx", "date"), (f"{LOCK_DATE_STR} 3_nwcst Tab perf.xlsx", None),
                           (f"{LOCK_DATE_STR} 3_nwcst Tab oos.xlsx", "date"), (f"{LOCK_DATE_STR} 3_nwcst Tab shock_residuals.xlsx", None)]:
    parts = []
    for s, sp in seed_paths.items():
        p = os.path.join(sp, tabname)
        if not os.path.exists(p): continue
        df = pd.read_excel(p, parse_dates=[index_col]) if index_col else pd.read_excel(p); df["seed"] = s; parts.append(df)
    if not parts: print(f"  [SKIP] {tabname}"); continue
    merged = pd.concat(parts, axis=0, ignore_index=True)
    if index_col and index_col in merged.columns:
        merged = merged.set_index(index_col); merged.index = pd.to_datetime(merged.index).strftime("%Y-%m-%d")
        write_xlsx(merged, os.path.join(MERGED, tabname), index_label=index_col)
    else: write_xlsx(merged, os.path.join(MERGED, tabname), index=False)
    print(f"  [OK] {tabname}: {len(merged)} rows across {len(parts)} seeds", flush=True)
first = list(seed_paths.values())[0]
for fn in os.listdir(first) if os.path.isdir(first) else []:
    if fn.endswith((".csv", ".json", ".png")) or "preselection" in fn or "data_quality" in fn:
        shutil.copy2(os.path.join(first, fn), os.path.join(MERGED, fn))

# ---- post-estimation cells on the merged pool (namespace built by the setup pass above) ----
PATH_OUTPUT = os.path.abspath(MERGED)
for tag in ["post", "live"]:
    print(f"  [cell {tag}] running...", flush=True)
    try: exec(CELLS[tag])
    except Exception as e: print(f"  [POST-CELL {tag}] {type(e).__name__}: {e}", flush=True)
RESULTS_NB = '4_results_tto.ipynb'
if os.path.exists(RESULTS_NB):
    rb = json.load(open(RESULTS_NB)); n_ok = n_err = 0
    for ci, c in enumerate(rb['cells']):
        if c.get('cell_type') != 'code': continue
        src = ''.join(c['source']).replace('plt.show()', 'plt.close()')
        if not src.strip(): continue
        try: exec(src); n_ok += 1
        except Exception as e: n_err += 1; print(f"  [chart cell {ci}] {type(e).__name__}: {e}", flush=True)
    print(f"  charts: {n_ok} ok, {n_err} errors", flush=True)
else: print(f"  {RESULTS_NB} not found -- charts skipped (step 9)", flush=True)
print(f"\n[{time.strftime('%H:%M:%S')}] DONE. Merged outputs in {MERGED}", flush=True)
