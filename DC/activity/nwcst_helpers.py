"""
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

def estimate_factors(X, r_max=10, criterion='IC1'):
    '''
    Information criteria to PCA.
    Bai and Ng (2002)
    inputs:
    - X: df
    - r_max: max number of factors (should be less than T or N)
    - criterion: conservative criteria is IC3
    '''
    
    T, N = X.shape
    r_max = min(r_max, T - 1)  # enforce theoretical max
    IC_vals = []
    
    for r in range(r_max + 1):
        if r == 0:
            sigma2_r = np.mean(X.values ** 2)
        else:
            pca = PCA(n_components=r)
            X_scaled = scaler.fit_transform(X)
            F_hat = pca.fit_transform(X_scaled)
            Lambda_hat = X_scaled.T @ F_hat / T
            X_hat = F_hat @ Lambda_hat.T
            sigma2_r = np.mean((X_scaled - X_hat).values ** 2)
    
        if criterion == 'IC1':
            penalty = r * (N + T) / (N * T) * np.log(N * T / (N + T))
        elif criterion == 'IC2':
            penalty = r * (N + T) / (N * T) * np.log(min(N, T))
        elif criterion == 'IC3':
            penalty = r * np.log(min(N, T)) / min(N, T)
        else:
            raise ValueError("Invalid criterion")
    
        IC_vals.append(np.log(sigma2_r) + penalty)

    return np.argmin(IC_vals)



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

    Chinn, M. D., Meunier, B., & Stumpner, S. (2023). Nowcasting world trade with machine learning: a three-step approach (No. w31419). National Bureau of Economic Research.
    """
    df0 = deepcopy(df)
    
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


def deduplicate_correlated(df, threshold=0.85, target=target_variable,
                           metadata=None, meta_col="ticket"):
    """Drop one variable from each highly-correlated pair.
    Priority: (1) keep aggregate tickets (numeric suffix % 100 == 0);
              (2) keep the variable with more non-null observations.

    Sub-components of the same category (e.g. BOP financial-account
    sub-accounts) are nearly collinear -- a selector picks several of them
    without adding information. `metadata` and `meta_col` are parameters here
    (the notebook copies of this function closed over notebook globals, which
    a module cannot do).
    """
    if metadata is None:
        raise ValueError("deduplicate_correlated() needs the metadata frame")

    def _is_agg(ticket):
        nums = re.findall(r'\d+', str(ticket))
        return bool(nums) and int(nums[-1]) % 100 == 0

    agg_set = set(metadata.loc[metadata[meta_col].apply(_is_agg), meta_col])

    X   = df.drop(columns=[target])
    obs = X.notna().sum()
    corr  = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        peers = upper.index[upper[col] > threshold].tolist()
        for peer in peers:
            if peer in to_drop:
                continue
            col_agg  = col  in agg_set
            peer_agg = peer in agg_set
            if col_agg and not peer_agg:
                to_drop.add(peer)
            elif peer_agg and not col_agg:
                to_drop.add(col)
                break
            elif obs[col] >= obs[peer]:
                to_drop.add(peer)
            else:
                to_drop.add(col)
                break

    kept = [c for c in df.columns if c not in to_drop]
    print(f"Removed {len(df.columns) - len(kept)} correlated variables "
          f"(|r| > {threshold}). {len(kept)} remain.")
    return df[kept]


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

def lagged_target(data, total_lags, metadata, target_variable=target_variable) :
    if total_lags == 0 :
        return data
    else :
        for l in range(1,total_lags+1) :
            data[f"{target_variable}_l{l}"] = data[target_variable]
            data.loc[data.index[-metadata[metadata['ticket']==target_variable]['months_lag'].values[0]:], f"{target_variable}_l{l}"] = np.nan
            data[f"{target_variable}_l{l}"] = data[f"{target_variable}_l{l}"].ffill()
            data[f"{target_variable}_l{l}"] = data[f"{target_variable}_l{l}"].shift(l)
            data[f"{target_variable}_l{l}"] = data[f"{target_variable}_l{l}"].fillna(data[f"{target_variable}_l{l}"].mean())
        return data


def lagged_target_q(data, total_lags, metadata, target_variable=target_variable):
    """Autoregressive GDP lags, QUARTERLY-INDEXED variant (act6_selection).

    Deliberate duplicate of lagged_target() above. The two differ, and both are
    live -- do not merge them:

      lagged_target()    blanks the last `months_lag` ROWS and always .shift(1).
                         Used by act8_nwcst's model loop; its numbers are baked
                         into every output/<date>/ run.
      lagged_target_q()  converts the publication lag to quarters,
                         max(1, months_lag // 3), and shifts by l. Correct when
                         the frame has already been collapsed to quarter-end
                         rows, which is what act6's walkthrough does.
    """
    if total_lags == 0:
        return data
    else:
        for l in range(1, total_lags + 1):
            data[f"{target_variable}_l{l}"] = data[target_variable]
            pub_lag_q = max(1, metadata[metadata['ticket'] == target_variable]['months_lag'].values[0] // 3)
            # // is integer (floor) division; converts monthly publication lag to quarterly
            data.loc[data.index[-pub_lag_q:], f"{target_variable}_l{l}"] = np.nan
            data[f"{target_variable}_l{l}"] = data[f"{target_variable}_l{l}"].ffill()
            data[f"{target_variable}_l{l}"] = data[f"{target_variable}_l{l}"].shift(l)
            data[f"{target_variable}_l{l}"] = data[f"{target_variable}_l{l}"].fillna(data[f"{target_variable}_l{l}"].mean())
        return data


def gen_aligned_data(metadata, data, last_date, lag, target_variable=target_variable):
    """Vintage dataset by ALIGNMENT: shift each series by its effective delay.

    Companion (and act8 successor) of gen_lagged_data() above. Where that one
    blanks the ragged edge and leaves the fill to the caller, this one replaces
    each series with its own publication-lag-shifted copy,

        d_j = max(0, months_lag_j - lag)        # effective delay, months
        x~_{j,t} = x_{j, t - d_j}               # latest value published at t

    so EVERY row -- training rows and the prediction row alike -- carries only
    what was actually available `lag` months around its reference date. Applied
    to train and test from the same frame, this is what makes the refit
    coefficients genuinely vintage-specific: the model trains on the same
    staleness it predicts with, instead of training on final data and
    predicting on mean-imputed constants.

    `date` and the target are never shifted (the target is y, and
    flatten_data() keys its row filter off it). d is floored at 0 -- a lag
    larger than the publication delay never turns into a lead. The only NaNs
    introduced are the first d_j rows of each column.
    """
    aligned = data.loc[data.date <= last_date, :].reset_index(drop=True)
    for col in aligned.columns:
        if col in ("date", target_variable):
            continue
        pub_lag = metadata.loc[metadata.ticket == col, "months_lag"].values[0]
        d = max(0, int(pub_lag) - lag)
        if d:
            aligned[col] = aligned[col].shift(d)
    return aligned


def lagged_target_v(data, total_lags, metadata, lag, target_variable=target_variable):
    """Autoregressive GDP lags, VINTAGE-AWARE variant (act8_nwcst).

    Third sibling of lagged_target() / lagged_target_q() above -- same deal,
    deliberate near-duplicate, do not merge. This one is for the refit-per-
    vintage act8 loop: the number of quarters to reach the most recent
    PUBLISHED GDP value depends on the vintage,

        d = max(0, months_lag_y - lag)          # effective delay, months
        q = max(1, ceil(d / 3))                 # quarters back, never current
        AR term l = y.shift(q + l - 1)          # on QUARTERLY rows

    Unlike lagged_target(), no blank-then-ffill: the shift is taken directly on
    the (real, unfilled at quarter rows) target, so an unpublished value is
    never carried forward into a feature. Leading rows that the shift cannot
    reach are filled with the column mean, matching the fill style of the
    other two variants. Expects a frame already collapsed to quarterly rows.
    """
    if total_lags == 0:
        return data
    pub_lag = metadata.loc[metadata.ticket == target_variable, "months_lag"].values[0]
    d = max(0, int(pub_lag) - lag)
    q = max(1, -(-d // 3))                       # ceil(d/3), floored at 1
    for l in range(1, total_lags + 1):
        col = data[target_variable].shift(q + l - 1)
        data[f"{target_variable}_l{l}"] = col.fillna(col.mean())
    return data


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


def fit_ols_flat(ttrain, metadata, gdp_lags=0, flat_lags=3,
                 target_variable=target_variable):
    """OLS that does its own reshaping (act6_selection walkthrough).

    Unlike fit_ols(ytrain, xtrain), which expects matrices that are already
    flattened and lagged, this one takes the raw rolling training frame and
    performs the whole recipe: flatten the within-quarter months into extra
    columns, keep quarter-end rows only, append GDP's own lags, then fit.

    Returns (fitted model, the column names it was trained on).
    """
    transformed_train = flatten_data(ttrain, target_variable, flat_lags)
    # keep only quarter-end months (March=3, June=6, Sep=9, Dec=12) -- GDP is quarterly
    transformed_train = (transformed_train
                         .loc[transformed_train.date.dt.month.astype(int).isin([3, 6, 9, 12]), :]
                         .fillna(transformed_train.mean(numeric_only=True))
                         .dropna(axis=1)
                         .reset_index(drop=True))
    transformed_train = lagged_target_q(transformed_train, gdp_lags, metadata,
                                        target_variable=target_variable)

    model = LinearRegression()
    x = transformed_train.drop(["date", target_variable], axis=1)
    y = transformed_train[target_variable]
    return model.fit(x, y), x.columns


def predict_ols_flat(model, X, train_vars, date, pred_dict, l, metadata,
                     flat_lags=3, gdp_lags=0,
                     target_variable=target_variable):
    """Companion of fit_ols_flat(): applies the SAME reshaping to the test
    frame, picks the single row for `date`, and appends the nowcast to
    pred_dict[l]. predict_ols() above assumes X arrives already reshaped.
    """
    Xi = flatten_data(X, target_variable, flat_lags)
    Xi = lagged_target_q(Xi, gdp_lags, metadata, target_variable=target_variable)

    # Select only the single row matching the target date
    Xi = Xi.loc[Xi['date'] == date, :].drop(["date", target_variable], axis=1)
    Xi = Xi[train_vars]      # reorder/subset columns to match the training set

    pred = model.predict(Xi)[0]
    pred_dict[l].append(pred)
    return pred_dict


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
    scale = False,
    ) :

    if scale:
        model = Pipeline([('scaler', StandardScaler()),
                          ('model', ElasticNet(alpha=params['alpha'],
                                               l1_ratio=params['l1_ratio']))])
        fit_kwargs = {'model__sample_weight': sample_weight} if sample_weight is not None else {}
        return model.fit(xtrain, ytrain, **fit_kwargs) , xtrain.columns

    model = ElasticNet(alpha = params['alpha'] , l1_ratio = params['l1_ratio'] )
    return model.fit(xtrain, ytrain, sample_weight=sample_weight) , xtrain.columns

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
    scale = False,
    ) :

    if scale:
        model = Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=alpha))])
        fit_kwargs = {'model__sample_weight': sample_weight} if sample_weight is not None else {}
        return model.fit(xtrain, ytrain, **fit_kwargs) , xtrain.columns

    model = Lasso( alpha = alpha )

    return model.fit(xtrain, ytrain, sample_weight=sample_weight) , xtrain.columns

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
    ) :
    # seed=None leaves random_state unset, so the ModelN members of the bag differ
    # from each other AND from the previous run -- that is what act8 has always
    # done. Pass an int and member i gets random_state=seed+i: still a diverse
    # bag, but the same bag every time, which is what makes a refit comparable.
    models = []
    for i in range(ModelN):
        model = DecisionTreeRegressor(criterion = "absolute_error",
                                      min_samples_split = 6,
                                      min_samples_leaf = 2,
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
    ):
    
    new_data = new_data[train_vars]
    
    preds = []
    for mod in model_fit :
        prediction = mod.predict(new_data)
        preds.append(prediction)
    
    oos_forecast = pd.DataFrame(preds).T.mean(axis=1).to_frame(name="nowcast")
    oos_forecast.index = new_dates
    oos_forecast['estimator'] = 'DT'
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
