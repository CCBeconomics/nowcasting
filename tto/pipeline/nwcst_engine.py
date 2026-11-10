"""
nwcst_engine.py  --  the estimation engine of the Trinidad and Tobago pipeline: one specification, every test
quarter, every publication lag (the estimation worker of 3_nwcst_tto.ipynb, in six stages that read as the method does).

    run_one_spec(row, opts, ctx)  ->  (spec, perf_df, fcst_df, oos_df, loo_rows, err)

is the only entry point the notebook's loop cell and run_tto_seed.py use. `row` is one spec_list row as a dict with
its screen's rolling selections attached (make_rows); `opts` are the config section 5 knobs (make_opts); `ctx` is
everything the engine needs from the data and specs cells (make_context). Any exception is caught and returned as
`err` (a traceback string), so one bad spec never stops a run.

Stages:
    build_frame         the spec's variables (union of the rolling top-K for a rolling spec), its estimator map,
                        parameters, the sliced frame, the test dates and the actuals
    train_quarter       one test quarter: this quarter's variables, training window to the quarter before, fill,
                        scaler + PCA fitted ONCE on the training window and kept, flatten, GDP lags, winsorised copy
                        for the linear models, decay weights, one fit per enabled estimator
    read_at_lags        the same quarter read at every publication lag with the quarter's transformers (never refit)
    loo_shock_residuals leave-one-out refit of each LOO_SHOCK_DATES quarter (COVID) on the first test quarter's
                        real-time training table; every enabled estimator except the LSTM
    make_tables         the performance and forecast tables of this spec
    oos_reading         the unprinted quarters to desired_date, read with the LAST test quarter's models

ESTIMATORS is the registry: one row per estimator, iterated in the order the former if-blocks had. `linear` rows
are fitted and read on the winsorised design; the LSTM (`frame="monthly"`) fits and reads the unflattened monthly
frame; DT4 is the depth-4 tree, carried by fit kwargs and an out-of-sample label. Adding an estimator = one row here
plus its fit_/predict_/oos_ triplet in nwcst_helpers.py.
"""
import os
import traceback

import numpy as np
import pandas as pd

from nwcst_helpers import (ffill_dataset, mean_fill_dataset, flatten_data, gen_lagged_data, gdp_lag_features,
                           perform_tab, forecast_table,
                           fit_ols, predict_ols, oos_ols, fit_olsr, predict_olsr, oos_olsr,
                           fit_enet, predict_enet, oos_enet, fit_lasso, predict_lasso, oos_lasso,
                           fit_dt, predict_dt, oos_dt, fit_rf, predict_rf, oos_rf,
                           fit_gbt, predict_gbt, oos_gbt, fit_lstm, predict_lstm, oos_lstm)

# ---------------------------------------------------------------------------------------------------------------
# Estimator registry (order = the former if-block order). fit(yt, xt, sw, p): xt is the winsorised design for
# linear rows and the raw design otherwise; p = the per-spec parameters (alphas, scale, seed, dt_n, rf_n, gbt_n).
# loo: refitted in the leave-one-out stage. frame="monthly": the LSTM path (fit on the monthly frame, predict on
# the lagged monthly frame, oos on the extended monthly frame).
# ---------------------------------------------------------------------------------------------------------------
ESTIMATORS = [
    dict(key="ols",   label="OLS",   linear=True,  loo=True,
         fit=lambda yt, xt, sw, p: fit_ols(yt, xt, sample_weight=sw),
         predict=predict_ols,  oos=oos_ols,  oos_kwargs={}),
    dict(key="olsr",  label="OLSR",  linear=True,  loo=True,
         fit=lambda yt, xt, sw, p: fit_olsr(yt, xt, alphas=p["alphas"], sample_weight=sw, scale=p["scale"]),
         predict=predict_olsr, oos=oos_olsr, oos_kwargs={}),
    dict(key="enet",  label="ENET",  linear=True,  loo=True,
         fit=lambda yt, xt, sw, p: fit_enet(yt, xt, sample_weight=sw, scale=p["scale"]),
         predict=predict_enet, oos=oos_enet, oos_kwargs={}),
    dict(key="lasso", label="LASSO", linear=True,  loo=True,
         fit=lambda yt, xt, sw, p: fit_lasso(yt, xt, sample_weight=sw, scale=p["scale"]),
         predict=predict_lasso, oos=oos_lasso, oos_kwargs={}),
    dict(key="dt",    label="DT",    linear=False, loo=True,
         fit=lambda yt, xt, sw, p: fit_dt(yt, xt, ModelN=p["dt_n"], sample_weight=sw, seed=p["seed"]),
         predict=predict_dt,   oos=oos_dt,   oos_kwargs={}),
    dict(key="dt4",   label="DT4",   linear=False, loo=True,
         fit=lambda yt, xt, sw, p: fit_dt(yt, xt, ModelN=p["dt_n"], sample_weight=sw, seed=p["seed"], min_leaf=4),
         predict=predict_dt,   oos=oos_dt,   oos_kwargs={"label": "DT4"}),
    dict(key="rf",    label="RF",    linear=False, loo=True,
         fit=lambda yt, xt, sw, p: fit_rf(yt, xt, ModelN=p["rf_n"], sample_weight=sw, seed=p["seed"]),
         predict=predict_rf,   oos=oos_rf,   oos_kwargs={}),
    dict(key="gbt",   label="GBT",   linear=False, loo=True,
         fit=lambda yt, xt, sw, p: fit_gbt(yt, xt, ModelN=p["gbt_n"], sample_weight=sw, seed=p["seed"]),
         predict=predict_gbt,  oos=oos_gbt,  oos_kwargs={}),
    dict(key="lstm",  label="LSTM",  linear=False, loo=False, frame="monthly",
         fit=lambda frame, p: fit_lstm(frame, gdp_lags=p["gdp_lags"], n_models=p["iter_lstm"],
                                       train_episodes=p["epoch_lstm"], seed=p["seed"]),
         predict=predict_lstm, oos=oos_lstm, oos_kwargs={}),
]


# ---------------------------------------------------------------------------------------------------------------
# What the callers build once
# ---------------------------------------------------------------------------------------------------------------
def make_opts(cfg, data, metadata, target_variable):
    """The config section 5 knobs the engine reads (the former `opts` dict of the loop cell). The seed runner
    overrides `seed`. Empty / None values switch each feature off."""
    return dict(
        fill               = cfg.NWCST_FILL,                     # "ffill"
        weight_decay       = cfg.SAMPLE_WEIGHT_DECAY,            # 0.95
        seed               = cfg.NWCST_SEED,                     # 42; the seed runner overrides per seed
        scale_linear       = bool(cfg.SCALE_LINEAR),
        ridge_alphas       = list(cfg.RIDGE_ALPHAS),
        winsor_z           = cfg.SHOCK_WINSOR_Z,                 # shock features are always on in this pipeline
        loo_dates          = [pd.to_datetime(x) for x in cfg.LOO_SHOCK_DATES],
        # GDP-lag features: lag l = the l-th most recent PUBLISHED quarter at the reading date (gdp_lag_features)
        gdp_lag_mode       = "vintage",
        gdp_pub_lag        = int(metadata.set_index("ticket").loc[target_variable, "months_lag"]),
        pseudo_prints      = {},
        chain_estimates    = {},
        data_cut           = data.index[data.drop(columns=[target_variable]).notna().any(axis=1)].max(),
    )


def make_context(data, metadata, target_variable, lags, run, train_start_date, test_start_date, test_end_date,
                 iters, iter_lstm, epoch_lstm, desired_date):
    """Everything the engine needs from the data and specs cells, frozen in one dict and passed to every call."""
    return dict(data=data, metadata=metadata, target_variable=target_variable, lags=list(lags), run=dict(run),
                train_start_date=train_start_date, test_start_date=test_start_date, test_end_date=test_end_date,
                iters=iters, iter_lstm=iter_lstm, epoch_lstm=epoch_lstm, desired_date=desired_date)


def make_rows(spec_list, rolling_selections, done_specs=()):
    """spec_list rows as dicts with their screen's rolling selections attached (common/shared/human/frozen rows are
    static); rows whose spec name is in done_specs are skipped."""
    rows = []
    for _, row in spec_list.iterrows():
        if row["spec"] in done_specs:
            continue
        d = row.to_dict()
        _m = d.get("method")
        d["rolling_selections"] = rolling_selections.get(_m, {}) if isinstance(_m, str) else {}
        rows.append(d)
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------------------------
def run_one_spec(row_dict, opts, ctx):
    """Estimate one spec_list row at every test quarter and publication lag.
    Returns (spec, perf_df, fcst_df, oos_df, loo_rows, err_str); err_str is None on success, else a traceback."""
    try:
        _cap_threads()
        fr = build_frame(row_dict, opts, ctx)                                   # stage 1
        preds = {e["key"]: {k: [] for k in ctx["lags"]} for e in ESTIMATORS}
        ttrain_loo_base = None
        q = None
        for date in fr["dates"]:
            q = train_quarter(date, fr, opts, ctx)                              # stage 2
            if ttrain_loo_base is None:
                ttrain_loo_base = q["ttrain"].copy()    # the refit uses data to the quarter before the test window: real time for every test quarter
            read_at_lags(date, q, fr, preds, opts, ctx)                         # stage 3
        loo_rows = loo_shock_residuals(ttrain_loo_base, q, fr, opts, ctx)       # stage 4
        perf_df, fcst_df = make_tables(preds, fr, ctx)                          # stage 5
        oos_df = oos_reading(q, fr, opts, ctx)                                  # stage 6
        return fr["spec"], perf_df, fcst_df, oos_df, loo_rows, None
    except Exception:
        return row_dict.get("spec", "UNKNOWN"), None, None, None, None, traceback.format_exc()


def _cap_threads():
    # torch + loky deadlock/segfault if a worker spawns its own thread pool.
    try:
        import torch as _torch
        _torch.set_num_threads(1)
        _torch.set_num_interop_threads(1)
    except Exception:
        pass


# ---------------------------------------------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------------------------------------------
def build_frame(row_dict, opts, ctx):
    data, metadata, target_variable = ctx["data"], ctx["metadata"], ctx["target_variable"]
    train_start_date, test_start_date, test_end_date = ctx["train_start_date"], ctx["test_start_date"], ctx["test_end_date"]
    run = ctx["run"]

    spec = row_dict["spec"]
    XVars = row_dict["selected_vars"].split(" ")

    # ---- config.py section 5 knobs ----
    _rolling_sel = row_dict.get("rolling_selections") or {}   # per screen
    _wz          = opts.get("winsor_z")

    # Rolling LARS: K comes off the spec row; the union of every date's top-K
    # is sliced out of `data` up front so each date's subset is available.
    _K_rolling = None
    if _rolling_sel and row_dict.get("rolling_enabled"):
        _K_raw = row_dict.get("K")
        try:
            _K_rolling = int(_K_raw) if _K_raw is not None and not pd.isna(_K_raw) else None
        except (ValueError, TypeError):
            _K_rolling = None
    if _rolling_sel and _K_rolling:
        _uni = set()
        for _lst in _rolling_sel.values():
            _uni.update(_lst[:_K_rolling])
        XVars = sorted(_uni)

    # params
    gdp_lags = row_dict.get("gdp_lags", 0)   # spec_list column is "gdp_lags"
    # per-spec estimator map (spec_list column "estimators"); missing/empty -> all enabled estimators
    _allowed = row_dict.get("estimators")
    if isinstance(_allowed, str) and _allowed.strip():
        _allowed = set(_allowed.split())
        run = {k: (v if k in _allowed else 0) for k, v in run.items()}
    param_pca = row_dict.get("PCA", 3)
    data_flat = row_dict.get("data_flat", 3)

    dt_n = row_dict.get("dt_n", ctx["iters"])
    rf_n = row_dict.get("rf_n", dt_n)
    gbt_n = row_dict.get("gbt_n", dt_n)
    params = dict(alphas=opts.get("ridge_alphas"), scale=bool(opts.get("scale_linear")), seed=opts.get("seed"),
                  dt_n=dt_n, rf_n=rf_n, gbt_n=gbt_n, gdp_lags=gdp_lags,
                  iter_lstm=ctx["iter_lstm"], epoch_lstm=ctx["epoch_lstm"])

    # Winsorizing: shock columns (the 9-character "...N"/"...R" tickets and their flattened lags) are unbounded by
    # construction, and a single extreme quarter can dominate a least-squares fit. Trees and the LSTM are
    # untouched -- an extreme value is information to them. _wz is None when SHOCK_WINSOR_Z is None in config.
    _shock_base = set()
    if _wz is not None:
        _cols = set(data.columns)        # a companion = a ticket ending in N or R whose base ticket is also a column
        _shock_base = {c for c in data.columns             # (the former len(c) == 9 rule missed the companions of
                       if isinstance(c, str) and c[-1:] in ('N', 'R') and c[:-1] in _cols}   # 7- and 10-character tickets)

    # slice data for this spec
    df = data[[c for c in data.columns if c in XVars or c == target_variable]].copy()

    # GENERATE TRAIN/TEST DATASET
    df = df.reset_index(drop=False)
    df.dropna(axis=1, thresh=23, inplace=True)

    test = df.loc[(df.date >= train_start_date) & (df.date <= test_end_date), :].reset_index(drop=True)

    # DATES AND ACTUAL VALUES
    dates = pd.date_range(test_start_date, test_end_date, freq="3MS").strftime("%Y-%m-%d").tolist()
    actuals = list(test.loc[test.date.astype(str).isin(dates), target_variable].values)

    return dict(spec=spec, XVars=XVars, run=run, params=params, df=df, test=test, dates=dates, actuals=actuals,
                rolling_sel=_rolling_sel, K_rolling=_K_rolling, shock_base=_shock_base,
                param_pca=param_pca, data_flat=data_flat, gdp_lags=gdp_lags,
                y_true=data[target_variable].dropna())


def _wins_bounds(df, cols, wz):
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        mu, sd = df[c].mean(), df[c].std()
        if sd is not None and pd.notna(sd) and sd > 0:
            out[c] = (mu - wz * sd, mu + wz * sd)
    return out


def _wins_apply(df, bounds):
    if not bounds:
        return df
    df = df.copy()
    for c, (lo, hi) in bounds.items():
        if c in df.columns:
            df[c] = df[c].clip(lo, hi)
    return df


def _fill_fn(opts):
    return ffill_dataset if opts.get("fill") == "ffill" else mean_fill_dataset


# ---------------------------------------------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------------------------------------------
def train_quarter(date, fr, opts, ctx):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    target_variable = ctx["target_variable"]
    run, params, test = fr["run"], fr["params"], fr["test"]
    _rolling_sel, _K_rolling = fr["rolling_sel"], fr["K_rolling"]
    param_pca, data_flat, gdp_lags = fr["param_pca"], fr["data_flat"], fr["gdp_lags"]
    _wz, _decay = opts.get("winsor_z"), opts.get("weight_decay")
    _lag_mode = opts.get("gdp_lag_mode", "vintage"); _pub = int(opts.get("gdp_pub_lag", 6))
    fill = _fill_fn(opts)
    scaler = StandardScaler()

    # Rolling LARS: use the ranking computed at the latest test quarter
    # at or before this one -- never a later one.
    if _rolling_sel and _K_rolling:
        _td = pd.to_datetime(date)
        _avail = [t for t in _rolling_sel.keys() if t <= _td]
        _use_t = max(_avail) if _avail else min(_rolling_sel.keys())
        XVars_date = list(_rolling_sel[_use_t][:_K_rolling])
    else:
        XVars_date = fr["XVars"]

    # Restrict the WHOLE frame -- training and test alike -- to this quarter's variables (the LSTM predicts on the
    # frame it is handed and looks each column up in the fill table it stored at training time; every other
    # estimator survives a surplus column because predict_*() does X[train_vars] first).
    test_d = test
    if _rolling_sel and _K_rolling:
        test_d = test[[c for c in test.columns
                       if c in XVars_date or c == target_variable or c == "date"]]

    # define rolling-window training set
    train = test_d.loc[
        test_d.date <= str(pd.to_datetime(date) - pd.tseries.offsets.DateOffset(months=3))[:10],
        :
    ]
    transformed_train = fill(train, train)

    # PCA scaling (if enabled): fit the scaler and the PCA ONCE on this quarter's training window and KEEP them
    # (pca_q, pca_cols), so the test rows in the lag loop and the OOS frame are transformed into the SAME component
    # space the models are trained in (refitting on each test frame would leak the test window and make the
    # components incomparable: PCA signs/order can flip between fits).
    pca_q = pca_cols = None
    if param_pca not in (False, 0, None):
        ncomp = 1 if param_pca is True else int(param_pca)

        X = transformed_train.set_index("date").drop(target_variable, axis=1)
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns).dropna(axis=1)

        pca_q = PCA(n_components=ncomp).fit(X)
        pca_cols = X.columns
        Z = pd.DataFrame(pca_q.transform(X)).rename(columns=lambda x: f"PC_{x}")

        transformed_train = pd.concat([transformed_train[["date", target_variable]], Z], axis=1)

    ttrain = flatten_data(transformed_train, target_variable, data_flat)

    # keep quarterly observations and drop early observations
    # NOTE: the mean is taken BEFORE the quarter filter, as it always has been -- the original chained
    # .fillna(ttrain.mean()) evaluated its argument against the unfiltered frame. Do not fold this back into
    # the chain; it would change every stored number.
    _ttrain_mean = ttrain.mean(numeric_only=True)
    ttrain = ttrain.loc[ttrain.date.dt.month.astype(int).isin([3, 6, 9, 12]), :]
    ttrain = (ttrain.ffill().bfill() if opts.get("fill") == "ffill"
              else ttrain.fillna(_ttrain_mean))
    ttrain = ttrain.dropna(axis=1).reset_index(drop=True)
    ttrain = gdp_lag_features(ttrain, gdp_lags, fr["y_true"], _pub, _lag_mode)                        # training convention

    xt = ttrain.drop(["date", target_variable], axis=1)
    yt = ttrain[target_variable]

    # winsorized copy for the linear models only
    _shock_cols = [c for c in xt.columns
                   if c in fr["shock_base"] or any(c.startswith(b + '_') for b in fr["shock_base"])]
    _wb = _wins_bounds(xt, _shock_cols, _wz) if _shock_cols else {}
    xt_lin = _wins_apply(xt, _wb)

    # Exponential decay: a quarter `a` quarters old weighs decay**a, so the fit leans on the recent past without
    # discarding the distant one.
    if _decay is not None:
        _ages = (ttrain['date'].max() - ttrain['date']).dt.days.values / 90.0
        sample_weight = np.power(float(_decay), _ages)
    else:
        sample_weight = None

    # Apply model in train: one fit per enabled estimator, in registry order
    fits = {}
    for e in ESTIMATORS:
        if run.get(e["key"], 0) != 1:
            continue
        if e.get("frame") == "monthly":
            fits[e["key"]] = e["fit"](transformed_train, params)
        else:
            fits[e["key"]] = e["fit"](yt, xt_lin if e["linear"] else xt, sample_weight, params)

    return dict(date=date, fits=fits, scaler=scaler, pca_q=pca_q, pca_cols=pca_cols, wb=_wb, shock_cols=_shock_cols,
                transformed_train=transformed_train, ttrain=ttrain, train=train, test_d=test_d, XVars_date=XVars_date)


# ---------------------------------------------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------------------------------------------
def read_at_lags(date, q, fr, preds, opts, ctx):
    target_variable, metadata, lags = ctx["target_variable"], ctx["metadata"], ctx["lags"]
    run, param_pca, data_flat, gdp_lags = fr["run"], fr["param_pca"], fr["data_flat"], fr["gdp_lags"]
    _lag_mode = opts.get("gdp_lag_mode", "vintage"); _pub = int(opts.get("gdp_pub_lag", 6))
    _pseudo = opts.get("pseudo_prints") or {}; _chain = opts.get("chain_estimates") or {}
    fill = _fill_fn(opts)

    # Publication Lags in Test data
    for lag in lags:
        tmp_data = gen_lagged_data(metadata, q["test_d"], date, lag)
        tmp_data = fill(q["train"], tmp_data)

        if param_pca not in (False, 0, None):
            # transform with the TRAIN-fitted scaler/PCA -- never refit on the test frame (see the training block)
            X = tmp_data.set_index("date").drop(target_variable, axis=1)
            X = pd.DataFrame(q["scaler"].transform(X), columns=X.columns)[q["pca_cols"]]
            Z = pd.DataFrame(q["pca_q"].transform(X)).rename(columns=lambda x: f"PC_{x}")

            tmp_data = pd.concat([tmp_data[["date", target_variable]], Z], axis=1)

        test_data = flatten_data(tmp_data, target_variable, data_flat)
        test_data = gdp_lag_features(test_data, gdp_lags, fr["y_true"], _pub, _lag_mode,
                                     reading_date=pd.to_datetime(date) + pd.DateOffset(months=int(lag)),
                                     estimates=_chain, pseudo=_pseudo)
        test_data = test_data.loc[test_data["date"] == date, :].drop(["date", target_variable], axis=1)
        test_data_lin = _wins_apply(test_data, q["wb"])

        # Predicted values, one predict per enabled estimator, in registry order
        for e in ESTIMATORS:
            if run.get(e["key"], 0) != 1:
                continue
            fit_obj, train_vars = q["fits"][e["key"]]
            if e.get("frame") == "monthly":
                preds[e["key"]] = e["predict"](fit_obj, tmp_data, train_vars, date, preds[e["key"]], lag, gdp_lags=gdp_lags)
            else:
                preds[e["key"]] = e["predict"](fit_obj, test_data_lin if e["linear"] else test_data, train_vars,
                                               date, preds[e["key"]], lag)


# ---------------------------------------------------------------------------------------------------------------
# Stage 4
# ---------------------------------------------------------------------------------------------------------------
def loo_shock_residuals(ttrain_base, q, fr, opts, ctx):
    """A shock quarter that sits INSIDE the training sample is fitted, not forecast, so its in-sample residual says
    nothing about how the model would have coped. Refitting with that one quarter held out and then predicting it
    gives a genuine out-of-sample error for the one kind of quarter we have almost no test observations of."""
    target_variable = ctx["target_variable"]
    run, params, spec = fr["run"], fr["params"], fr["spec"]
    _wz, _decay = opts.get("winsor_z"), opts.get("weight_decay")
    _loo_dates_w = opts.get("loo_dates") or []
    _shock_cols = q["shock_cols"]

    loo_rows = []
    _tl = ttrain_base if ttrain_base is not None else q["ttrain"]   # real-time table: data to the quarter before the test window
    for _loo_date in _loo_dates_w:
        try:
            if "date" not in _tl.columns or not (_tl["date"] == _loo_date).any():
                continue
            _mask      = _tl["date"] == _loo_date
            _shock_row = _tl[_mask].reset_index(drop=True)
            _train_loo = _tl[~_mask].reset_index(drop=True)

            _yt_loo   = _train_loo[target_variable]
            _xt_loo   = _train_loo.drop(["date", target_variable], axis=1)
            _xt_shock = _shock_row.drop(["date", target_variable], axis=1)
            _y_actual = float(_shock_row[target_variable].iloc[0])

            # Re-derive the winsorizing bounds and the weights on the held-out sample, so nothing from the
            # excluded quarter leaks back in.
            _wb_loo       = _wins_bounds(_xt_loo, _shock_cols, _wz) if _shock_cols else {}
            _xt_lin_loo   = _wins_apply(_xt_loo,   _wb_loo)
            _xt_lin_shock = _wins_apply(_xt_shock, _wb_loo)

            if _decay is not None:
                _ages_loo = (_train_loo["date"].max() - _train_loo["date"]).dt.days / 90.0
                _sw_loo = np.power(float(_decay), _ages_loo).values
            else:
                _sw_loo = None

            for e in ESTIMATORS:
                if not e["loo"] or run.get(e["key"], 0) != 1:
                    continue
                _fit, _ = e["fit"](_yt_loo, _xt_lin_loo if e["linear"] else _xt_loo, _sw_loo, params)
                _pred = _loo_predict(_fit, _xt_lin_shock if e["linear"] else _xt_shock)
                if _pred is not None:
                    loo_rows.append({"spec": spec, "estimator": e["label"],
                                     "shock_date": str(pd.to_datetime(_loo_date).date()),
                                     "actual": _y_actual, "predicted": _pred,
                                     "residual": _pred - _y_actual})
        except Exception as _loo_err:
            print(f"[LOO WARNING spec={spec} date={_loo_date}] {_loo_err}")
    return loo_rows


def _loo_predict(fit_obj, x_df):
    try:
        if isinstance(fit_obj, list):      # dt / rf / gbt return a bag
            preds = []
            for _m in fit_obj:
                try:
                    preds.append(float(_m.predict(x_df)[0]))
                except Exception:
                    pass
            return float(np.nanmean(preds)) if preds else None
        return float(fit_obj.predict(x_df)[0])
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------------------------
# Stage 5
# ---------------------------------------------------------------------------------------------------------------
def make_tables(preds, fr, ctx):
    run, spec, actuals, dates, lags = fr["run"], fr["spec"], fr["actuals"], fr["dates"], ctx["lags"]
    perf_parts, fcst_parts = [], []
    for e in ESTIMATORS:
        if run.get(e["key"], 0) != 1:
            continue
        perf_parts.append(perform_tab(preds[e["key"]], e["label"], spec, actuals, lags))
        fcst_parts.append(forecast_table(preds[e["key"]], e["label"], spec, actuals, dates))
    perf_df = pd.concat(perf_parts, axis=0) if perf_parts else pd.DataFrame()
    fcst_df = pd.concat(fcst_parts, axis=0) if fcst_parts else pd.DataFrame()
    return perf_df, fcst_df


# ---------------------------------------------------------------------------------------------------------------
# Stage 6
# ---------------------------------------------------------------------------------------------------------------
def oos_reading(q, fr, opts, ctx):
    """The unprinted quarters to desired_date, read with the LAST test quarter's models (q). Empty when
    desired_date is None (identification run)."""
    desired_date = ctx["desired_date"]
    oos_df = pd.DataFrame()      # stays empty when desired_date is None
    if desired_date is None or q is None:
        return oos_df
    target_variable = ctx["target_variable"]
    run, spec, df = fr["run"], fr["spec"], fr["df"]
    param_pca, data_flat, gdp_lags = fr["param_pca"], fr["data_flat"], fr["gdp_lags"]
    _lag_mode = opts.get("gdp_lag_mode", "vintage"); _pub = int(opts.get("gdp_pub_lag", 6))
    _pseudo = opts.get("pseudo_prints") or {}; _chain = opts.get("chain_estimates") or {}
    fill = _fill_fn(opts)
    date = q["date"]

    # Same column-parity requirement as the test loop: the fitted models are the ones from the LAST test quarter,
    # so the OOS frame must carry that quarter's variables and no others.
    if fr["rolling_sel"] and fr["K_rolling"]:
        df = df[[c for c in df.columns
                 if c in q["XVars_date"] or c == target_variable or c == "date"]]

    # Extend dataframe to desired_date
    while desired_date > df['date'].max():
        df.loc[len(df), "date"] = df['date'].max() + pd.DateOffset(months=1)

    transformed_new_data = fill(df, df)

    # Apply PCA if specified -- transform with the scaler/PCA fitted on the LAST test quarter's training window:
    # the same component space the fitted models expect (never refit on the OOS frame)
    if param_pca not in (False, 0, None):
        X = transformed_new_data.set_index('date').drop(target_variable, axis=1)
        X = pd.DataFrame(q["scaler"].transform(X), columns=X.columns)[q["pca_cols"]]
        Z = pd.DataFrame(q["pca_q"].transform(X)).rename(columns=lambda x: f'PC_{x}')

        transformed_new_data = pd.concat([transformed_new_data[['date', target_variable]], Z], axis=1)

    # Prepare OOS data
    transformed_data = flatten_data(transformed_new_data, target_variable, data_flat)
    transformed_data = gdp_lag_features(transformed_data, gdp_lags, fr["y_true"], _pub, _lag_mode,
                                        reading_date=opts.get("data_cut"), estimates=_chain, pseudo=_pseudo)
    transformed_data = transformed_data[transformed_data['date'] > date]
    transformed_data = transformed_data[transformed_data['date'] <= desired_date]
    transformed_data = transformed_data.loc[transformed_data.date.dt.month.astype(int).isin([3, 6, 9, 12]), :]
    # transformed_train.mean() -- the TRAINING frame from the last test quarter, and called without numeric_only,
    # exactly as before.
    transformed_data = (transformed_data.ffill().bfill() if opts.get("fill") == "ffill"
                        else transformed_data.fillna(q["transformed_train"].mean()))
    transformed_data = transformed_data.dropna(axis=1)
    new_dates = transformed_data['date']
    transformed_data = transformed_data.drop(["date", target_variable], axis=1).reset_index(drop=True)
    transformed_data_lin = _wins_apply(transformed_data, q["wb"])

    outofsample_table = pd.DataFrame()
    # Run OOS forecasts, one oos_* call per enabled estimator, in registry order
    for e in ESTIMATORS:
        if run.get(e["key"], 0) != 1:
            continue
        fit_obj, train_vars = q["fits"][e["key"]]
        if e.get("frame") == "monthly":
            new_data = transformed_new_data
        else:
            new_data = transformed_data_lin if e["linear"] else transformed_data
        outofsample_table = e["oos"](fit_obj, new_data, train_vars=train_vars, table=outofsample_table,
                                     new_dates=new_dates, spec=spec, desired_date=desired_date, **e["oos_kwargs"])

    # Convert to DataFrame if needed
    return pd.DataFrame(outofsample_table) if isinstance(outofsample_table, list) else outofsample_table
