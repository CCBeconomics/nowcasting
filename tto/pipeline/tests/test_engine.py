"""Engine identity test: four specs through nwcst_engine.run_one_spec, loaded from a run's hand-off file
(race_context.pkl), compared cell by cell with that run's seed-42 tables. Covers every branch of the registry:
a static all-estimator spec with the LSTM (human1), a frozen flats spec, a rolling PCA + GDP-lag spec and the
common group (or, in a run without one, a rolling base spec); the names are taken from the run's own spec list
(or TTO_TEST_SPECS). One spec also goes through joblib to prove the engine pickles.

    cd pipeline && TTO_RUN_TAG=prod python tests/test_engine.py [ctx_dir] [ref_seed_dir]
"""
import os, sys, time, pickle, importlib.util
import numpy as np, pandas as pd

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(CODE); sys.path.insert(0, CODE)
OUT = os.path.join(CODE, "..", "output")
CTX_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "20261106_tto_prod", "_context")
REF_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT, "20261106_tto_prod_s42")
REF_DATE = os.path.basename(os.path.normpath(REF_DIR)).split("_")[0]
ATOL = 1e-9

os.environ.setdefault("TTO_RUN_TAG", "ident" if "_ident" in os.path.basename(os.path.normpath(REF_DIR)) else "prod")
_spec = importlib.util.spec_from_file_location("config", os.path.join(CODE, "config.py"))
cfg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cfg)
from nwcst_engine import run_one_spec, make_context, make_opts, make_rows

ctx_file = pickle.load(open(os.path.join(CTX_DIR, "race_context.pkl"), "rb"))
data, metadata, spec_list, rolling_selections = ctx_file["data"], ctx_file["metadata"], ctx_file["spec_list"], ctx_file["rolling_selections"]
target = cfg.TARGET
test_end_date = data[target].dropna().index.max()
if cfg.TEST_END_ACTIVE: test_end_date = min(test_end_date, pd.to_datetime(cfg.TEST_END_ACTIVE))
desired_date = pd.to_datetime(cfg.DESIRED_DATE_ACTIVE) if cfg.DESIRED_DATE_ACTIVE else None
ctx = make_context(data, metadata, target, cfg.LAGS, cfg.RUN, cfg.TRAIN_START, cfg.TEST_START_ACTIVE, test_end_date,
                   cfg.ITER, cfg.ITER_LSTM, cfg.EPOCH_LSTM, desired_date)
opts = dict(make_opts(cfg, data, metadata, target), seed=42)
rows = {r["spec"]: r for r in make_rows(spec_list, rolling_selections)}
# The four specs: TTO_TEST_SPECS="a,b,c,d" or, by default, the ones of this run that match each branch
_names = set(spec_list["spec"].astype(str))
def _pick(*pats):
    import re as _re
    for p in pats:
        hits = sorted(n for n in _names if _re.fullmatch(p, n))
        if hits: return hits[0]
    raise SystemExit(f"test_engine: no spec in {CTX_DIR} matches {pats}")
SPECS = (os.environ["TTO_TEST_SPECS"].split(",") if os.environ.get("TTO_TEST_SPECS") else
         [_pick(r"fshap\d+flats", r"bench\d+flats"), _pick(r"lars\d+p5l2", r"\w+?\d+p5l2"),
          _pick(r"common\d+p3", r"common\d+flats", r"common\d+", r"sis\d+", r"\w+?\d+"), _pick(r"human1")])
print(f"specs under test: {SPECS}")
print(f"context {CTX_DIR}: {len(spec_list)} specs, panel {data.shape}; test window {cfg.TEST_START_ACTIVE} -> {test_end_date.date()}, live {desired_date}")

ref = {t: pd.read_excel(os.path.join(REF_DIR, f"{REF_DATE} 3_nwcst Tab {t}.xlsx")) for t in ["perf", "fcst", "oos", "shock_residuals"]}

def norm(df, datecol=None):
    df = df.copy()
    if datecol and datecol not in df.columns:            # the engine's fcst/oos tables carry the date as their index
        df = df.reset_index().rename(columns={df.index.name or "index": datecol})
    if datecol and datecol in df.columns: df[datecol] = pd.to_datetime(df[datecol]).dt.strftime("%Y-%m-%d")
    return df

def compare(name, got, want, keys):
    got, want = norm(got, "date" if "date" in keys else None), norm(want, "date" if "date" in keys else None)
    if "shock_date" in keys: got["shock_date"] = got["shock_date"].astype(str); want["shock_date"] = want["shock_date"].astype(str)
    vals = [c for c in want.columns if c not in keys and pd.api.types.is_numeric_dtype(want[c]) and c in got.columns]
    m = got.merge(want, on=keys, suffixes=("_g", "_w"), how="outer", indicator=True)
    unmatched = int((m["_merge"] != "both").sum())
    diff = max((float(np.nanmax(np.abs(m[f"{c}_g"].values.astype(float) - m[f"{c}_w"].values.astype(float)))) for c in vals), default=0.0)
    ok = unmatched == 0 and diff <= ATOL and len(got) > 0
    print(f"   {name:16s} rows {len(got):4d} vs {len(want):4d} | unmatched {unmatched} | max |diff| {diff:.2e} | {'OK' if ok else 'FAIL'}")
    return ok

CACHE = os.path.join(OUT, "logs", f"test_engine_results_{REF_DATE}.pkl")   # engine output cached, so a comparison fix never re-estimates
if os.path.exists(CACHE) and os.environ.get("TTO_TEST_REUSE", "1") == "1":
    results = pickle.load(open(CACHE, "rb")); print(f"engine results loaded from {CACHE} (set TTO_TEST_REUSE=0 to re-estimate)")
else:
    results = {}
    # one spec through joblib (the engine must pickle into loky workers)
    from joblib import Parallel, delayed
    t0 = time.time()
    par = Parallel(n_jobs=2, backend="loky")(delayed(run_one_spec)(rows[s], opts, ctx) for s in SPECS[:2])
    for r in par: results[r[0]] = r
    print(f"joblib: {SPECS[:2]} in {time.time()-t0:.0f}s")
    for s in SPECS[2:]:
        t0 = time.time(); results[s] = run_one_spec(rows[s], opts, ctx); print(f"sequential: {s} in {time.time()-t0:.0f}s")
    pickle.dump(results, open(CACHE, "wb"))

all_ok = True
for s in SPECS:
    spec, perf, fcst, oos, loo, err = results[s]
    print(f"\n{s}:")
    if err: print(err); all_ok = False; continue
    all_ok &= compare("perf", perf, ref["perf"][ref["perf"].spec == s], ["Vintage", "estimator", "spec"])
    all_ok &= compare("fcst", fcst, ref["fcst"][ref["fcst"].spec == s], ["date", "estimator", "spec"])
    if desired_date is not None: all_ok &= compare("oos", oos, ref["oos"][ref["oos"].spec == s], ["date", "estimator", "spec"])
    w = ref["shock_residuals"][ref["shock_residuals"].spec == s]
    if len(w) or loo: all_ok &= compare("shock_residuals", pd.DataFrame(loo), w, ["spec", "estimator", "shock_date"])
print("\nENGINE TEST:", "IDENTICAL" if all_ok else "DIFFERENCES FOUND")
sys.exit(0 if all_ok else 1)
