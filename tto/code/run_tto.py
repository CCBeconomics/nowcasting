"""
Trinidad orchestrator (port of Jamaica's run_beta16.py, 2026-10-24): three seeds in parallel, merge, post-estimation
cells on the merged pool, charts if 4_results_tto.ipynb exists. The run (ident / prod) is chosen by TTO_RUN_TAG.

  TTO_RUN_TAG=ident python run_tto.py     # identification run (from tto/pipeline/, nwcst env active)
  TTO_RUN_TAG=prod  python run_tto.py     # production run
  (macOS: prefix with `caffeinate -i` so the laptop does not sleep; the run takes hours)
"""
import sys, os, json, re, warnings, time, shutil, subprocess
from datetime import date as _date
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sklearn
if not sklearn.__version__.startswith("1.8"):
    sys.stderr.write(f"[run_tto] ABORT: sklearn=={sklearn.__version__}; use the nowcst env.\n"); sys.exit(2)
RUN_TAG = os.environ.get("TTO_RUN_TAG", "prod")
SEEDS = [42, 43, 44]; N_JOBS_PER_SEED = 3
PY = sys.executable                                   # the interpreter running this script (the nwcst env)
today = _date.today(); LOCK_DATE_STR = os.environ.get("TTO_LAUNCH_DATE_STR") or f"{today.year:04d}{today.month:02d}{today.day:02d}"
OUT_SUFFIX = os.environ.get("TTO_OUT_SUFFIX", ""); SUB = f"{LOCK_DATE_STR}_tto_{RUN_TAG}{OUT_SUFFIX}"   # variant runs (2026-10-25)
print(f"[{time.strftime('%H:%M:%S')}] Trinidad orchestrator | run {RUN_TAG} | seeds {SEEDS} | date_str {LOCK_DATE_STR}", flush=True)

procs = []
for s in SEEDS:
    os.makedirs(os.path.join("..", "output", "logs"), exist_ok=True)
    log_path = os.path.join("..", "output", "logs", f"tto_{RUN_TAG}{OUT_SUFFIX}_s{s}.log")   # one log per seed and variant
    env = os.environ.copy(); env["TTO_LAUNCH_DATE_STR"] = LOCK_DATE_STR; env["TTO_RUN_TAG"] = RUN_TAG
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
        merged.to_excel(os.path.join(MERGED, tabname), index_label=index_col)
    else: merged.to_excel(os.path.join(MERGED, tabname), index=False)
    print(f"  [OK] {tabname}: {len(merged)} rows across {len(parts)} seeds", flush=True)
first = list(seed_paths.values())[0]
for fn in os.listdir(first) if os.path.isdir(first) else []:
    if fn.endswith((".csv", ".json", ".png")) or "preselection" in fn or "data_quality" in fn:
        shutil.copy2(os.path.join(first, fn), os.path.join(MERGED, fn))

# ---- post-estimation cells on the merged pool ----
import matplotlib; matplotlib.use('Agg')
nb = json.load(open('3_nwcst_tto.ipynb'))
CELLS = {}
for c in nb['cells']:
    if c.get('cell_type') == 'code':
        src = ''.join(c['source']); m = re.match(r"# \[tto:(\w+)\]", src)
        if m: CELLS[m.group(1)] = src.replace('plt.show()', 'plt.close()')
os.environ["TTO_LAUNCH_DATE_STR"] = LOCK_DATE_STR
for tag in ["setup", "data", "filters", "helpers", "preselection", "specs"]:
    try: exec(CELLS[tag])
    except Exception as e: print(f"[cell {tag}] FAILED during merge setup: {e}", flush=True); sys.exit(1)
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
