"""
nwcst_helpers.py -- the model library of the Trinidad and Tobago nowcasting pipeline: the six screens, the data
helpers, and fit_/predict_/oos_ for every estimator. fit_enet and fit_lasso choose their penalties by time-series CV;
fit_dt takes a min_leaf parameter (the depth-4 tree uses 4). Functions nothing calls are parked outside the package.

nwcst_helpers.py  —  nowcasting model library (factors, LARS ranking, data
shaping, and fit/predict/oos for OLS, OLSR, ENET, LASSO, DT, RF, GBT, LSTM).

  * lars_variable_ranking() fits the scaler and the PCA on the TRAINING rows only
    and transforms the test rows with them, so the ranking's RMSE carries no
    look-ahead leakage. It also returns a true RMSE (sqrt of the mean squared
    error), not the raw MSE.
  * fit_rf() caps max_features at the number of columns actually available.

Two module globals are read by the functions below as defaults:
  * target_variable : the GDP target column. Defaults to "RGDP0000" (the
        project-wide ticker = config.GDP_TICKER). If a country uses a different
        ticker, pass target_variable=... explicitly at the call sites.
  * scaler          : a shared StandardScaler() instance (re-fit on each call).
`metadata` (lagged_target), `lags` (perform_tab) and `desired_date` (oos_*) are
genuine runtime state — the orchestration notebook passes them in explicitly.

"""
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit
import os, json, time, warnings, re, traceback, timeit
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn import linear_model, tree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lars, LinearRegression, RidgeCV, ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.seasonal import seasonal_decompose

import torch                                  # required: used in fit_lstm default args
from nowcast_lstm.LSTM import LSTM
try:
    import pmdarima as pm                      # optional
except Exception:
    pm = None

# --- module globals read as defaults by the functions below ---
target_variable = "RGDP0000"
scaler = StandardScaler()


def lars_variable_ranking(df, target, max_vars=None, test=None, factors=5):
    """
    Stepwise LARS variable selection: rank variables by their entry into the model.
    - df: DataFrame with predictors and target (train period only)
    - target: name of the target variable
    - max_vars: number of variables to select; if None, continue until all are ranked
    - test: DataFrame with test period rows (same columns as df)
    Returns:
    - selected_vars: ordered list of variable names by entry into the LARS model
    - rmse: out-of-sample RMSE on the test period (or None)
    Predictors are z-scored on the rows passed in before ranking; the scoring call below uses the original df.

    Chinn, M. D., Meunier, B., & Stumpner, S. (2023). Nowcasting world trade with machine learning: a three-step approach (No. w31419). National Bureau of Economic Research.
    """
    df0 = deepcopy(df)

    # Standardise the predictors on the rows being ranked; the target is left as is.
    # LARS enters the variable with the largest covariance with the current residual, so on an
    # unscaled panel the entry order follows the series' volatility, not their signal (in a diagnostic
    # the raw top-10 were the ten most volatile of 398 candidates). Z-scoring makes
    # the order scale-free, like the other five screens, and restores the assumption of Efron et al.
    # (2004) that scikit-learn applied itself until its `normalize` option was removed in 1.2. Only
    # the rows passed in are used (the static panel or the rolling slice strictly before the test
    # quarter), so nothing leaks from the test period.
    _x_cols = [c for c in df0.columns if c != target]
    _sd = df0[_x_cols].std(ddof=0).replace(0, 1.0)        # constant columns stay at zero after centring
    df0[_x_cols] = (df0[_x_cols] - df0[_x_cols].mean()) / _sd

    selected_order = []
    count = 0
    max_vars = max_vars or (df0.shape[1] - 1)

    while count < max_vars:
        X = df0.drop(columns=[target])
        y = df0[target]

        # Stop if no variables remain
        if X.shape[1] == 0:
            break

        # Fit LARS model (drop NaN rows for fitting)
        mask = y.notna()
        model = Lars(n_nonzero_coefs=X.shape[1], jitter=1e-10, random_state=0)
        model.fit(X[mask], y[mask])

        # coef_path_ has shape (n_features, n_steps)
        coef_path = np.array(model.coef_path_)
        # Count how many times each variable is zero
        zero_counts = (coef_path == 0).sum(axis=1)

        # Summary table
        out = pd.DataFrame({'Variable': X.columns, 'ZeroCount': zero_counts})
        out = out.sort_values('ZeroCount')
        
        # Take variables that are uniquely at this ZeroCount level
        grouped = out.groupby('ZeroCount')
        selected = grouped.filter(lambda g: len(g) == 1)

        # If nothing uniquely enters, dump all remaining vars
        if selected.empty:
            remaining = X.columns.tolist()
            selected = pd.DataFrame({'Variable': remaining, 'ZeroCount': [count]*len(remaining)})
            selected_order.extend(remaining)
            print("Warning: Using fallback procedure for LARS")
            break
        else:
            selected_vars = selected['Variable'].tolist()
            selected_order.extend(selected_vars)
            df0 = df0.drop(columns=selected_vars)
            count += len(selected_vars)

    top_vars = selected_order[:max_vars]

    if test is None:
        return top_vars, None

    # Delegate scoring to the frozen evaluator shared by every method
    return top_vars, evaluate_selection(df, test, target, top_vars,
                                        factors=factors)

# =========================================================
# RANKER 1: SIS (Sure Independence Screening, Fan & Lv 2008)
# Rank candidates by absolute marginal correlation with the target,
# computed on the TRAINING window only.
# One of the four pre-selection techniques tested in:
# Chinn, M. D., Meunier, B., & Stumpner, S. (2023). Nowcasting world trade
# with machine learning: a three-step approach. NBER WP 31419.
# =========================================================
def rank_sis(train, target):
    # Absolute correlation of every candidate with the target.
    # .corrwith uses pairwise-complete observations, but the panel arrives
    # mean-filled from the prep block, so in practice every variable is
    # scored on the full training window (identical input treatment to
    # LARS -> clean comparability across methods).
    corr = train.drop(columns=[target]).corrwith(train[target]).abs()

    corr = corr.dropna().sort_values(ascending=False)  # best first
    # Returns: (a) ordered variable list, best first; rankings are NESTED,
    # so top-40 is the first 40 of top-100 and we rank ONCE, then slice.
    # (b) the scores themselves, useful for plots and tables.
    return corr.index.tolist(), corr


# =========================================================
# RANKER 2: T-STAT HARD THRESHOLDING (Bai & Ng 2008)
# Univariate screening regressions with GDP-lag controls:
#     y_t = const + sum_j( b_j * y_{t-j} ) + g * x_t + e_t
# Variables are ranked by |t-stat| on g, i.e. by the strength of the
# information a candidate carries BEYOND what GDP's own history predicts
# (not raw co-movement, which is what SIS measures).
# Origin: Bair et al. (2006, supervised PCA screening stage); in macro via
# Jurado, Ludvigson & Ng (2015); tested in Chinn, Meunier & Stumpner (2023).
# Adaptations vs the reference implementation:
#   - gdp_lags default 2 (quarterly target) instead of 4 (their monthly target)
#   - HAC / Newey-West standard errors (serially correlated residuals overstate
#     naive t-stats with ~60 quarterly observations)
#   - each screen runs on that variable's own available sample (dropna per
#     pair), so unequal histories are compared honestly through the t-stat;
#     on the mean-filled panel this is inert, but it makes the function
#     correct if unfilled data is ever passed
# =========================================================
def rank_tstat(train, target, gdp_lags=2, hac_lags=4):
    import statsmodels.api as sm                     # local import: only needed here

    y = train[target]                                # the target (GDP growth)

    # Build the GDP-lag control block once: y_{t-1} ... y_{t-gdp_lags}
    controls = pd.concat({f"{target}_l{j}": y.shift(j)
                          for j in range(1, gdp_lags + 1)}, axis=1)

    scores = {}
    for var in train.columns.drop(target):
        # Assemble the screening regression sample for THIS candidate:
        # target + its own lags + the candidate, rows with any NaN dropped
        block = pd.concat([y, controls, train[var]], axis=1).dropna()
        if len(block) <= gdp_lags + 3:               # not enough obs to estimate
            continue
        if block[var].std() == 0:                    # constant column, no signal
            continue

        X = sm.add_constant(block.drop(columns=[target]))   # const + lags + candidate
        try:
            res = sm.OLS(block[target], X).fit(
                cov_type='HAC',                       # Newey-West robust errors
                cov_kwds={'maxlags': hac_lags})
            scores[var] = abs(res.tvalues[var])       # |t| on the candidate only
        except Exception:
            continue                                  # singular / degenerate: skip

    scores = pd.Series(scores).sort_values(ascending=False)  # best first
    # Returns: nested ranking (rank once, slice at any size) + the |t| scores
    return scores.index.tolist(), scores


# =========================================================
# RANKER 3: SHAP IMPORTANCE (no lineal)
# Ajusta un Gradient Boosting sobre la ventana de ENTRENAMIENTO y ordena las
# variables por el promedio del valor absoluto de SHAP (atribucion de Shapley
# de cada prediccion entre las variables; TreeSHAP la calcula exacto para
# arboles). A diferencia de permutation importance o Boruta, NO hay shuffling:
# no se viola la estructura temporal de las series (clave con datos macro).
# Captura relevancia no lineal (umbrales, interacciones) que LARS/SIS/t-stat
# no ven por construccion.
# Refs: Lundberg & Lee (2017); Lundberg et al. (2020, TreeSHAP);
#       Marcilio & Eler (2020, SHAP como mecanismo de seleccion);
#       Chapman & Desai (2022, mean |SHAP| en nowcasting de PIB).
# Requiere: pip install shap
# =========================================================
def rank_shap(train, target, n_estimators=300, max_depth=2,
              learning_rate=0.05, random_state=0):
    from sklearn.ensemble import GradientBoostingRegressor
    try:
        import shap                                   # import local: solo aqui
    except ImportError:
        raise ImportError("Falta el paquete 'shap': pip install shap")

    X = train.drop(columns=[target])                  # panel ya viene mean-filled
    y = train[target]

    # Arboles poco profundos + learning rate bajo: hiperparametros conservadores
    # para n~60 trimestres (arboles profundos memorizan la muestra y el ranking
    # SHAP hereda ese sobreajuste). Semilla fija -> ranking reproducible.
    model = GradientBoostingRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      learning_rate=learning_rate,
                                      random_state=random_state)
    model.fit(X, y)

    # TreeSHAP: matriz (obs x variables) de atribuciones sobre el TRAIN
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Importancia global = promedio de |SHAP| por variable (Chapman & Desai)
    imp = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
    imp = imp[imp > 0].sort_values(ascending=False)   # variables nunca usadas
                                                      # por los arboles quedan
                                                      # fuera del ranking
    # Devuelve: ranking anidado (se ranquea UNA vez y se corta a cualquier
    # tamano) + los scores para graficos/tablas
    return imp.index.tolist(), imp


# =========================================================
# RANKER 4: OCMT (One Covariate at a time Multiple Testing)
# Chudik, Kapetanios & Pesaran (2018, Econometrica 86(4)).
# Misma regresion univariada que rank_tstat (const + rezagos del PIB +
# candidata, errores HAC), pero con DISCIPLINA DE TESTEO MULTIPLE:
#   - con p=358 tests al 5%, ~18 variables pasan por puro azar;
#   - el valor critico de OCMT crece con el numero de tests:
#         c(p, delta) = Phi^{-1}( 1 - alpha / (2 * p^delta) )
#     (delta=1 en la primera etapa, delta_star>1 en las siguientes);
#   - segunda etapa CONDICIONAL: las descartadas se re-testean incluyendo
#     las ya seleccionadas como controles -> recupera variables que solo
#     importan en conjunto (responde la objecion clasica a los metodos
#     marginales sin la inestabilidad de la estimacion conjunta completa).
# Salida natural de OCMT: un CONJUNTO seleccionado (trae su propio umbral,
# a diferencia de SIS/SHAP). Para el scaffold devolvemos ademas un ranking
# completo: seleccionadas primero (por |t| de su etapa de entrada), luego
# las demas por su |t| de ultima etapa.
# =========================================================
def rank_ocmt(train, target, gdp_lags=2, hac_lags=4,
              alpha=0.05, delta=1.0, delta_star=1.5, max_stages=5):
    import statsmodels.api as sm
    from scipy.stats import norm

    y = train[target]
    controls = pd.concat({f"{target}_l{j}": y.shift(j)
                          for j in range(1, gdp_lags + 1)}, axis=1)
    candidates = [c for c in train.columns if c != target]
    p = len(candidates)                                # numero de tests

    def tstat_given(selected, var):
        """|t| de 'var' en: y ~ const + rezagos PIB + seleccionadas + var"""
        cols = [train[s] for s in selected] + [train[var]]
        block = pd.concat([y, controls] + cols, axis=1).dropna()
        # grados de libertad suficientes
        if len(block) <= gdp_lags + len(selected) + 4:
            return np.nan
        if block[var].std() == 0:
            return np.nan
        X = sm.add_constant(block.drop(columns=[target]))
        try:
            res = sm.OLS(block[target], X).fit(cov_type='HAC',
                                               cov_kwds={'maxlags': hac_lags})
            return abs(res.tvalues[var])
        except Exception:
            return np.nan

    selected, entry_t = [], {}                         # orden y |t| de entrada
    last_t = {}                                        # ultimo |t| observado
    remaining = list(candidates)

    for stage in range(max_stages):
        # valor critico de la etapa: mas exigente cuanto mas tests (p^delta)
        d_ = delta if stage == 0 else delta_star
        crit = norm.ppf(1 - alpha / (2 * p ** d_))

        newly = []
        for var in remaining:
            t = tstat_given(selected, var)
            if np.isnan(t):
                continue
            last_t[var] = t
            if t > crit:
                newly.append((var, t))

        if not newly:                                  # nada nuevo: convergio
            break
        newly.sort(key=lambda x: -x[1])                # mas fuertes primero
        for var, t in newly:
            selected.append(var)
            entry_t[var] = t
            remaining.remove(var)

        # tope de tamano: no dejar que la etapa condicional agote los grados
        # de libertad (~60 obs trimestrales)
        if len(selected) > len(y.dropna()) - gdp_lags - 8:
            break

    # Ranking completo: seleccionadas (por |t| de entrada) + resto (por ultimo |t|)
    rest = (pd.Series({v: last_t.get(v, 0.0) for v in remaining})
              .sort_values(ascending=False))
    ranking = selected + rest.index.tolist()
    scores = pd.Series({**entry_t, **rest.to_dict()})
    # n_selected = tamano del conjunto que OCMT selecciona por si mismo
    return ranking, scores, len(selected)


# =========================================================
# RANKER 5: iBMA (Iterated Bayesian Model Averaging)
# Yeung, Bumgarner & Raftery (2005); aproximacion BIC de Raftery (1995);
# uno de los cuatro metodos de Chinn, Meunier & Stumpner (2023).
# No hay paquete Python mantenido (Chinn et al. usan R), pero la
# aproximacion BIC elimina la necesidad de MCMC y lo hace enumerable:
#   1. "Pecking order": candidatas ordenadas por R2 univariado con el target.
#   2. Ventana de w variables: se enumeran TODOS los 2^w modelos OLS posibles;
#      peso posterior de cada modelo ~ exp(-BIC/2).
#   3. PIP de cada variable = suma de pesos de los modelos que la incluyen.
#   4. Variables con PIP < pip_drop se descartan; la ventana se rellena con
#      las siguientes del pecking order; se itera hasta agotar la lista.
#   5. Ranking final por PIP (sobrevivientes primero, luego descartadas por
#      su ultimo PIP observado).
# w=10 -> 1,024 modelos por ventana: segundos, no minutos, con n~60.
# =========================================================
def rank_ibma(train, target, w=10, pip_drop=0.05, max_iter=200):
    dfc = train.dropna(subset=[target])
    y = dfc[target].values
    n = len(y)

    # --- 1. pecking order: R2 univariado ---------------------------------
    r2 = {}
    for var in train.columns.drop(target):
        x = dfc[var].values
        if np.nanstd(x) == 0:
            continue
        mask = ~np.isnan(x)
        if mask.sum() < 10:
            continue
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        r2[var] = r ** 2
    order = pd.Series(r2).sort_values(ascending=False).index.tolist()

    # --- 2-4. ventanas iteradas con enumeracion 2^w y pesos BIC ----------
    def window_pips(vars_in):
        Xw = dfc[vars_in].to_numpy()
        Xw = np.where(np.isnan(Xw), np.nanmean(Xw, axis=0), Xw)  # guardia
        k = len(vars_in)
        bics = np.empty(2 ** k)
        ones = np.ones((n, 1))
        for m in range(2 ** k):                       # cada subconjunto
            idx = [j for j in range(k) if (m >> j) & 1]
            X = np.hstack([ones, Xw[:, idx]]) if idx else ones
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            rss = ((y - X @ beta) ** 2).sum()
            bics[m] = n * np.log(max(rss, 1e-12) / n) + X.shape[1] * np.log(n)
        wgt = np.exp(-(bics - bics.min()) / 2)        # estabilizado
        wgt /= wgt.sum()
        pips = {}
        for j, v in enumerate(vars_in):
            inc = np.array([(m >> j) & 1 for m in range(2 ** k)], dtype=bool)
            pips[v] = wgt[inc].sum()
        return pips

    window = order[:w]
    queue = order[w:]
    last_pip = {}
    for _ in range(max_iter):
        pips = window_pips(window)
        last_pip.update(pips)
        keep = [v for v in window if pips[v] >= pip_drop]
        # garantia de avance: si nadie cae, cae la de menor PIP
        if len(keep) == len(window) and queue:
            keep.remove(min(window, key=lambda v: pips[v]))
        window = keep
        while len(window) < w and queue:
            window.append(queue.pop(0))
        if not queue and (len(window) == len(keep)):
            break

    final_pips = window_pips(window) if window else {}
    last_pip.update(final_pips)

    # --- 5. ranking final por PIP ----------------------------------------
    scores = pd.Series(last_pip).sort_values(ascending=False)
    return scores.index.tolist(), scores


def gen_lagged_data(metadata, data, last_date, lag):
    # only go up to the last date
    lagged_data = data.loc[data.date <= last_date, :].reset_index(drop=True)
    for col in lagged_data.columns[1:]:
        pub_lag = metadata.loc[metadata.ticket == col, "months_lag"].values[0] # publication lag of this particular variable
        # go back as far as needed for the pub_lag of the variable, then + the lag (so -2 for 2 months back), also -1 because 0 lag means in month, last month data available, not current month in
        #lagged_data.loc[(len(lagged_data) - pub_lag + lag - 1) :, col] = np.nan
        lagged_data.loc[(len(lagged_data) - pub_lag + lag) :, col] = np.nan

    return lagged_data

def flatten_data(data, target_variable, n_lags):
    flattened_data = data.loc[~pd.isna(data[target_variable]), :]
    orig_index = flattened_data.index
    for i in range(1, n_lags):
        lagged_indices = orig_index - i
        lagged_indices = lagged_indices[lagged_indices >= 0]
        tmp = data.loc[lagged_indices, :]
        tmp.date = tmp.date + pd.DateOffset(months=i)
        tmp = tmp.drop([target_variable], axis=1)
        tmp.columns = [j + "_" + str(i) if j != "date" else j for j in tmp.columns]
        flattened_data = flattened_data.merge(tmp, how="left", on="date")

    return flattened_data

def mean_fill_dataset(training, test):
    mean_dict = {}
    for col in training.columns[1:]:
        mean_dict[col] = np.nanmean(training[col])
    filled = test.copy()
    for col in training.columns[1:]:
        filled.loc[pd.isna(filled[col]), col] = mean_dict[col]
        
    return filled

def ffill_dataset(training, test):
    """Ragged-edge fill by carrying the last observation forward.

    The alternative to mean_fill_dataset(): where that one drops a
    training-sample mean into every gap, this one repeats whatever was last
    observed (then back-fills any leading NaN). For a series in growth rates the
    mean is the safer neutral guess; for a series with persistent level shifts
    the last observation carries more information. act8 picks between them with
    config.NWCST_FILL; act6 always uses the mean.
    """
    combined = pd.concat([training, test], ignore_index=True)
    combined = combined.ffill().bfill()
    return combined.iloc[len(training):].reset_index(drop=True)


def gdp_lag_features(frame, total_lags, y_true, pub_lag_months, mode="vintage", reading_date=None,
                     estimates=None, pseudo=None, target_variable=target_variable):
    """GDP-lag features built from the REAL prints, not from the frame's (filled) target column.

    frame      : monthly or quarterly frame with a "date" column (first day of the month).
    y_true     : the target series at its quarter-end stamps (data[target].dropna()); pseudo-prints, if any, are in it.
    mode "vintage": lag l = the l-th most recent quarter PUBLISHED at the reading date, where quarter d is published
                    when its value is known and d + pub_lag_months <= reading_date (a pseudo-print counts as published).
                    Training frames (reading_date None) use the V0 convention, two quarters back: lag l = y(t - l - 1).
    mode "chain"  : lag l = quarter t-l: its print when published, else the pipeline's own nowcast of t-l at its latest
                    vintage on or before the reading date (estimates = {(quarter_end, vintage): value}), else the most
                    recent published quarter. Training frames: lag l = y(t - l).
    Leading rows the lags cannot reach are filled with the column mean, as the older helpers did.
    Replaces lagged_target() in the model loop, whose blank-then-ffill left the last six quarters of every training
    window with one frozen value in both lag columns ."""
    if total_lags == 0:
        return frame
    y = pd.Series(y_true).dropna().copy(); y.index = pd.to_datetime(y.index)
    pseudo = {pd.Timestamp(k): float(v) for k, v in (pseudo or {}).items()}
    for k, v in pseudo.items():
        y.loc[k] = v
    y = y.sort_index()
    def _qend(d):                      # quarter-end stamp (first day of the quarter's last month) of any monthly date
        d = pd.Timestamp(d); m = ((d.month - 1) // 3 + 1) * 3; return pd.Timestamp(d.year, m, 1)
    rows = pd.to_datetime(frame["date"]); row_q = [_qend(d) for d in rows]
    qd = list(y.index); yv = [float(v) for v in y.values]
    _maxq = max(row_q) if len(row_q) else qd[-1]
    while qd[-1] < _maxq:                                   # extend the quarter axis over the frame (live rows): unknown values
        qd.append(qd[-1] + pd.DateOffset(months=3)); yv.append(np.nan)
    _minq = min(row_q) if len(row_q) else qd[0]
    while qd[0] > _minq:                                    # and before the first print (a frame that starts at the sample's
        qd.insert(0, qd[0] - pd.DateOffset(months=3)); yv.insert(0, np.nan)   # first quarter, whose growth rate does not exist)
    pos = {d: i for i, d in enumerate(qd)}
    rd = pd.Timestamp(reading_date) if reading_date is not None else None
    est = {(pd.Timestamp(k[0]), int(k[1])): float(v) for k, v in (estimates or {}).items()}
    def _printed(i):
        if i < 0 or i >= len(qd) or np.isnan(yv[i]): return False
        if rd is None or qd[i] in pseudo: return True
        return (qd[i] + pd.DateOffset(months=int(pub_lag_months))) <= rd
    for l in range(1, total_lags + 1):
        out = []
        for qe in row_q:
            i = pos[qe]; val = np.nan
            if rd is None:                                  # training convention
                j = i - l - (1 if mode == "vintage" else 0)
                val = yv[j] if j >= 0 else np.nan
            elif mode == "chain":
                j = i - l
                if j >= 0:
                    if _printed(j): val = yv[j]
                    else:
                        dj = qd[j]; v = (rd.year - dj.year) * 12 + (rd.month - dj.month)
                        cands = [vv for (q_, vv) in est if q_ == dj and vv <= v]
                        if cands: val = est[(dj, max(cands))]
                        else:
                            k = j - 1
                            while k >= 0 and not _printed(k): k -= 1
                            val = yv[k] if k >= 0 else np.nan
            else:                                           # vintage convention at a reading date
                k = i - 1; cnt = 0
                while k >= 0:
                    if _printed(k):
                        cnt += 1
                        if cnt == l: val = yv[k]; break
                    k -= 1
            out.append(val)
        col = pd.Series(out, index=frame.index, dtype=float)
        frame[f"{target_variable}_l{l}"] = col.fillna(col.mean())
    return frame


# called by perform_tab / forecast_table



def evaluate_selection(train, test, target, top_vars, factors=5):
    """
    FROZEN EVALUATOR. Scores ANY set of selected variables with exactly the
    same recipe as the benchmark: standardize -> PCA -> OLS -> out-of-sample
    RMSE. It is deliberately identical for every selection method, so any
    difference in RMSE is attributable to the SELECTOR alone, never to the
    model that scores it (Chinn, Meunier & Stumpner 2023: freeze the
    evaluation, vary the selection).

    - train / test: quarterly DataFrames, same columns
    - top_vars: list of selected variable names
    - factors: number of PCA components (same as the benchmark)

    Scaler and PCA are fitted on TRAIN ONLY and applied to test: without this
    the test window leaks into the selection and every method looks better
    than it is.
    """
    X_train_raw = train[list(top_vars)]
    X_test_raw  = test[list(top_vars)]
    y_train_all = train[target]
    y_test_all  = test[target]

    # Fit scaler on train only, transform both (no data leakage)
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_raw), index=X_train_raw.index
    ).dropna(axis=1)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw), index=X_test_raw.index
    )[X_train_scaled.columns]

    # Fit PCA on train only, transform both
    pca_local = PCA(n_components=factors)
    X_train_pca = pd.DataFrame(
        pca_local.fit_transform(X_train_scaled), index=X_train_raw.index,
        columns=[f"PC_{i}" for i in range(factors)]
    )
    X_test_pca = pd.DataFrame(
        pca_local.transform(X_test_scaled), index=X_test_raw.index,
        columns=[f"PC_{i}" for i in range(factors)]
    )

    # Drop quarters without observed GDP, in train and in test
    obs_mask      = y_train_all.notna()
    obs_mask_test = y_test_all.notna()

    final_model = LinearRegression()
    final_model.fit(X_train_pca[obs_mask], y_train_all[obs_mask])
    y_pred = final_model.predict(X_test_pca[obs_mask_test])

    return float(np.sqrt(np.mean((y_test_all[obs_mask_test].values - y_pred) ** 2)))



def vintage_col_names(lags):
    """Map vintage integers -> forecast_table column names, {lag: name}.

    The historical five-vintage grid keeps its legacy names so every stored
    output/{date}/ table and act9_results.ipynb keep loading unchanged; any
    other grid gets self-describing p-names ("p-1", "p+0", ...). Single source
    of truth for the vintage column schema -- act8's post-loop cells derive
    their column lists from this instead of hard-coding the five names.
    """
    legacy = {-2: "two_back", -1: "one_back", 0: "zero_back",
              1: "one_ahead", 2: "two_ahead"}
    lags = sorted(lags)
    if lags == [-2, -1, 0, 1, 2]:
        return {l: legacy[l] for l in lags}
    return {l: f"p{l:+d}" for l in lags}


def perform_tab( pred_values , model_name , specification , values , lags) :
    # table of RMSE by vintage
    table = pd.DataFrame(columns=[ "Vintage", "RMSE", "estimator", "spec" ])
    for lag in lags:
        tmp = pd.DataFrame({
            "Vintage" : lag,
            "RMSE" : np.sqrt(np.mean((np.array(values) - np.array(pred_values[lag])) ** 2)) ,
            "estimator": model_name ,
            "spec" : specification ,
        }, index=[0])
        table = pd.concat([table, tmp]).reset_index(drop=True)
    table.round(4)
    table.set_index('Vintage' , inplace=True)
    return table

def forecast_table( pred_values , model_name , specification, values  , dates ) :
    # plot of predictions vs actuals
    # One column per vintage, named by vintage_col_names() -- the historical
    # -2..+2 grid keeps its two_back..two_ahead names, so the on-disk schema
    # act9_results.ipynb reads is unchanged; other grids no longer KeyError.
    names = vintage_col_names(pred_values.keys())
    cols = { "actuals": values }
    for lag, name in names.items():
        cols[name] = pred_values[lag]
    cols["estimator"] = model_name
    cols["spec"] = specification
    result = pd.DataFrame(cols)
    result.index = pd.to_datetime(dates)
    return result

def fit_ols(
    ytrain ,
    xtrain ,
    target_variable = target_variable,
    sample_weight = None,
    ) :

    model = LinearRegression()
    return model.fit(xtrain, ytrain, sample_weight=sample_weight) , xtrain.columns

def predict_ols(
    model ,
    X ,
    train_vars ,
    date ,
    pred_dict ,
    l ,
    target_variable = target_variable ,
    ) :
    X = X[train_vars]
    
    pred = model.predict(X)[0]
    pred_dict[l].append(pred)
    return pred_dict

def oos_ols(
    model_fit ,
    new_data ,
    train_vars ,
    table ,
    new_dates,
    spec = "model" ,
    desired_date = None,
    ):

    new_data = new_data[train_vars]
    oos_forecast = model_fit.predict( new_data )
    oos_forecast = pd.DataFrame( {
        "date": new_dates,
        "nowcast": oos_forecast
    }).set_index('date')
    oos_forecast['estimator'] = 'OLS'
    oos_forecast['spec'] = spec
    
    table = pd.concat([ table , oos_forecast ], axis=0)
    return table

def fit_olsr(
    ytrain ,
    xtrain ,
    target_variable = target_variable,
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 20],
    sample_weight = None,
    scale = False,
    ) :
    # scale=True wraps RidgeCV in a StandardScaler pipeline. Ridge penalises the
    # coefficient vector, so without scaling a predictor measured in thousands is
    # penalised far less than one measured in percent. Off by default because
    # act8's stored runs were fit unscaled.
    if scale:
        model = Pipeline([('scaler', StandardScaler()),
                          ('ridge', RidgeCV(alphas=alphas))])
        if sample_weight is not None:
            return model.fit(xtrain, ytrain, ridge__sample_weight=sample_weight) , xtrain.columns
        return model.fit(xtrain, ytrain) , xtrain.columns

    model = RidgeCV( alphas = alphas )
    return model.fit(xtrain, ytrain, sample_weight=sample_weight) , xtrain.columns

def predict_olsr(
    model ,
    X ,
    train_vars ,
    date ,
    pred_dict ,
    l ,
    target_variable = target_variable,
    ) :
    X = X[train_vars]
    
    pred = model.predict(X)[0]
    pred_dict[l].append(pred)
    return pred_dict

def oos_olsr( 
    model_fit ,
    new_data ,
    train_vars ,
    table,
    new_dates,
    spec = "model" ,
    desired_date = None,
    ):
    new_data = new_data[train_vars]
    oos_forecast = model_fit.predict(new_data)
    
    oos_forecast = pd.DataFrame( {
        "date": new_dates,
        "nowcast": oos_forecast
    }).set_index('date')
    oos_forecast['estimator'] = 'OLSR'
    oos_forecast['spec'] = spec
    
    table = pd.concat([ table , oos_forecast ], axis=0)
    return table

def fit_enet(
    ytrain,
    xtrain ,
    target_variable = target_variable,
    params = {
        'alpha' : 1e-5 ,
        'l1_ratio' : 0.25 ,
    },
    sample_weight = None,
    scale = True,          # accepted for the worker's signature; the CV pipelines always scale
    ) :

    # alpha and l1_ratio chosen by time-series CV (a fixed alpha of 1e-5 would be unregularised)
    # least squares, which on p > n designs produced the -147-type readings of the V+1 cluster.
    model = Pipeline([('scaler', StandardScaler()),
                      ('model', ElasticNetCV(l1_ratio=[0.25, 0.5, 0.75], n_alphas=50,
                                             cv=TimeSeriesSplit(n_splits=4), max_iter=5000))])
    fit_kwargs = {'model__sample_weight': sample_weight} if sample_weight is not None else {}
    return model.fit(xtrain, ytrain, **fit_kwargs) , xtrain.columns


def predict_enet(
    model ,
    X ,
    train_vars ,
    date ,
    pred_dict ,
    l ,
    target_variable = target_variable,
    ) :
    X = X[train_vars]
    
    pred = model.predict(X)[0]
    pred_dict[l].append(pred)
    return pred_dict

def oos_enet( 
    model_fit,
    new_data ,
    train_vars ,
    table,
    new_dates,
    spec = "model" ,
    desired_date = None,
    ):
    new_data = new_data[train_vars]
    oos_forecast = model_fit.predict(new_data)
    
    oos_forecast = pd.DataFrame( {
        "date": new_dates,
        "nowcast": oos_forecast
    }).set_index('date')
    oos_forecast['estimator'] = 'ENET'
    oos_forecast['spec'] = spec
    
    table = pd.concat([ table , oos_forecast ], axis=0)
    return table

def fit_lasso(
    ytrain, 
    xtrain ,
    target_variable = target_variable,
    alpha = 1e-5,
    sample_weight = None,
    scale = True,          # accepted for the worker's signature; the CV pipelines always scale
    ) :

    # alpha chosen by time-series CV
    model = Pipeline([('scaler', StandardScaler()),
                      ('model', LassoCV(n_alphas=50, cv=TimeSeriesSplit(n_splits=4), max_iter=5000))])
    fit_kwargs = {'model__sample_weight': sample_weight} if sample_weight is not None else {}
    return model.fit(xtrain, ytrain, **fit_kwargs) , xtrain.columns


def predict_lasso(
    model ,
    X ,
    train_vars ,
    date ,
    pred_dict ,
    l ,
    target_variable = target_variable,
    ) :
    X = X[train_vars]
    
    pred = model.predict(X)[0]
    pred_dict[l].append(pred)
    return pred_dict

def oos_lasso( 
    model_fit,
    new_data ,
    train_vars ,
    table,
    new_dates,
    spec = "model" ,
    desired_date = None,
    ):
    new_data = new_data[train_vars]
    oos_forecast = model_fit.predict(new_data)
    
    oos_forecast = pd.DataFrame( {
        "date": new_dates,
        "nowcast": oos_forecast
    }).set_index('date')
    oos_forecast['estimator'] = 'LASSO'
    oos_forecast['spec'] = spec
    
    table = pd.concat([ table , oos_forecast ], axis=0)
    return table

def fit_dt(
    ytrain ,
    xtrain ,
    target_variable = target_variable,
    ModelN = 200 ,
    sample_weight = None,
    seed = None,
    min_leaf = 2,          # DT4 passes 4: >= 4 quarters per leaf blunts memorised shock leaves
    ) :
    # seed=None leaves random_state unset, so the ModelN members of the bag differ
    # from each other AND from the previous run -- that is what act8 has always
    # done. Pass an int and member i gets random_state=seed+i: still a diverse
    # bag, but the same bag every time, which is what makes a refit comparable.
    models = []
    for i in range(ModelN):
        model = DecisionTreeRegressor(criterion = "absolute_error",
                                      min_samples_split = 6,
                                      min_samples_leaf = min_leaf,
                                      random_state = None if seed is None else seed + i)

        model.fit(xtrain, ytrain, sample_weight=sample_weight)
        models.append(model)

    return models , xtrain.columns

def predict_dt(
    model ,
    X ,
    train_vars ,
    date ,
    pred_dict ,
    l ,
    target_variable = target_variable,
    ) :
    X = X[train_vars]
    
    preds = []
    for mod in model:
        prediction = mod.predict(X)[0]
        preds.append(prediction)
    
    pred_dict[l].append(np.nanmean(preds))
    return pred_dict

def oos_dt( 
    model_fit,
    new_data ,
    train_vars ,
    table,
    new_dates,
    spec = "model" ,
    desired_date = None,
    label = "DT",            # "DT4" when called for the 4-quarter tree (was always "DT", so pool_mask dropped DT4 from the live pool)
    ):
    
    new_data = new_data[train_vars]
    
    preds = []
    for mod in model_fit :
        prediction = mod.predict(new_data)
        preds.append(prediction)
    
    oos_forecast = pd.DataFrame(preds).T.mean(axis=1).to_frame(name="nowcast")
    oos_forecast.index = new_dates
    oos_forecast['estimator'] = label
    oos_forecast['spec'] = spec
    
    table = pd.concat([ table , oos_forecast ], axis=0)
    return table

def fit_rf(
    ytrain ,
    xtrain , 
    target_variable = target_variable,
    ModelN = 200 , 
    n_estimators = 130 ,
    sample_weight = None,
    seed = None,
    ) :

    models = []
    for i in range(ModelN):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion = "absolute_error",
            max_features=min(18, xtrain.shape[1]),   # cap: a trimmed spec can have <18 predictors
            min_samples_split=4,
            min_samples_leaf=2,
            random_state = None if seed is None else seed + i
            )

        model.fit(xtrain, ytrain, sample_weight=sample_weight)
        models.append(model)

    return models , xtrain.columns

def predict_rf(
    model ,
    X ,
    train_vars ,
    date ,
    pred_dict ,
    l ,
    target_variable = target_variable,
    ) :
    X = X[train_vars]
    
    preds = []
    for mod in model:
        prediction = mod.predict(X)[0]
        preds.append(prediction)
    
    pred_dict[l].append(np.nanmean(preds))
    return pred_dict

def oos_rf( 
    model_fit,
    new_data ,
    train_vars ,
    table,
    new_dates,
    spec = "model" ,
    desired_date = None,
    ):

    new_data = new_data[train_vars]
    
    preds = []
    for mod in model_fit :
        prediction = mod.predict(new_data)
        preds.append(prediction)
    
    oos_forecast = pd.DataFrame(preds).T.mean(axis=1).to_frame(name="nowcast")
    oos_forecast.index = new_dates
    oos_forecast['estimator'] = 'RF'
    oos_forecast['spec'] = spec
    
    table = pd.concat([ table , oos_forecast ], axis=0)
    return table

def fit_gbt(
    ytrain, 
    xtrain ,
    target_variable = target_variable,
    ModelN = 200 , 
    n_estimators = 100 ,
    learning = 0.15 ,
    sample_weight = None,
    seed = None,
    ) :

    models = []
    for i in range(ModelN):
        model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=learning,
                    loss='absolute_error',
                    min_samples_split=6,
                    min_samples_leaf=3,
                    random_state = None if seed is None else seed + i
                )

        model.fit(xtrain, ytrain, sample_weight=sample_weight)
        models.append(model)

    return models , xtrain.columns

def predict_gbt(
    model ,
    X ,
    train_vars ,
    date ,
    pred_dict ,
    l ,
    target_variable = target_variable,
    ) :
    X = X[train_vars]
    
    preds = []
    for mod in model:
        prediction = mod.predict(X)[0]
        preds.append(prediction)
    
    pred_dict[l].append(np.nanmean(preds))
    return pred_dict

def oos_gbt( 
    model_fit,
    new_data ,
    train_vars ,
    table ,
    new_dates,
    spec = "model" ,
    desired_date = None,
    ):

    new_data = new_data[train_vars]
    
    preds = []
    for mod in model_fit:
        prediction = mod.predict(new_data)
        preds.append(prediction)
    
    oos_forecast = pd.DataFrame(preds).T.mean(axis=1).to_frame(name="nowcast")
    oos_forecast.index = new_dates
    oos_forecast['estimator'] = 'GBT'
    oos_forecast['spec'] = spec
    
    table = pd.concat([ table , oos_forecast ], axis=0)
    return table

def fit_lstm(
    ttrain ,
    gdp_lags = 0,
    target_variable = target_variable,
    n_models = None ,
    train_episodes = None ,
    seed = None ,
    params = {
        "n_timesteps": 12,
        "fill_na_func": np.nanmean,
        "fill_ragged_edges_func": np.nanmean,
        "n_models": 100,
        "train_episodes": 100,
        "batch_size": 50,
        "decay": 0.98,
        "n_hidden": 10,
        "n_layers": 1,
        "dropout": 0.0,
        "criterion": torch.nn.MSELoss(),
        "optimizer": torch.optim.Adam,
        "optimizer_parameters": {"lr": 1e-2, "weight_decay": 0.0}
    }
    ) :

    #train = lagged_target(ttrain, gdp_lags)

    # Seed torch + numpy so the network's initialisation is reproducible inside
    # each joblib worker. Wrapped: a torch build without CUDA raises here.
    if seed is not None:
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        np.random.seed(seed)

    # Per-run overrides from config.py (ITER_LSTM / EPOCH_LSTM). Copy first so we
    # never mutate the shared default `params` dict across calls.
    params = {**params}
    if n_models is not None:
        params["n_models"] = n_models
    if train_episodes is not None:
        params["train_episodes"] = train_episodes

    model = LSTM(
        data = ttrain,
        target_variable = target_variable ,
        **params
        )
    model.train(quiet=True)
    
    return model , ttrain.drop(["date", target_variable], axis=1).columns

def predict_lstm(
    model ,
    X ,
    train_vars ,
    date ,
    pred_dict ,
    l ,
    gdp_lags = 0 ,
    target_variable = target_variable,
    ) :
    
    #X = lagged_target(X, gdp_lags)
    #X = X[train_vars]
    
    pred = model.predict(X).loc[lambda x: x.date == date, "predictions"].values[0]
    
    pred_dict[l].append(pred)
    return pred_dict

def oos_lstm( 
    model_fit,
    new_data ,
    train_vars ,
    table,
    new_dates,
    gdp_lags = 0 ,
    spec = "model" ,
    desired_date = None,
    ):
    
    #new_data = lagged_target(new_data, gdp_lags)
    new_data = new_data[new_data['date'] <= desired_date ]
    
    new_dates = new_data['date']
    
    oos_forecast = model_fit.predict(new_data)[['date', 'predictions']].set_index('date')
    
    oos_forecast = oos_forecast[oos_forecast.index <= desired_date]
    oos_forecast = oos_forecast[oos_forecast.index > new_data[['date',target_variable]].dropna().date.max()]
    oos_forecast = oos_forecast.rename(columns = {'predictions' : 'nowcast'})
    oos_forecast = oos_forecast[oos_forecast.index.month.astype(int).isin([3,6,9,12])]
    oos_forecast['estimator'] = 'LSTM'
    oos_forecast['spec'] = spec
    
    table = pd.concat([ table , oos_forecast ], axis=0)
    return table
