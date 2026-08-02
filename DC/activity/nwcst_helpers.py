"""
nwcst_helpers.py  —  nowcasting model library (factors, LARS ranking, data
shaping, and fit/predict/oos for OLS, OLSR, ENET, LASSO, DT, RF, GBT, LSTM).

WORKSHOP DC COPY. Sibling of ntl_helpers.py; imported by act8_nwcst.ipynb so the
notebook stays a thin orchestrator. Built from _baseline/nwcst_helpers.py, with
two deliberate departures from it:

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

NOTE on `desired_date`: in _baseline the eight oos_* functions declare
`desired_date=desired_date`, a name that does not exist at module level, so that
module cannot be imported at all. Defining the missing global is NOT the fix:
`from nwcst_helpers import *` would then rebind the notebook's own desired_date
to None and every OOS block would be silently skipped. The default is None here.
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

    # --- Evaluation: mirror the benchmark OLS setup exactly ---
    df_train = df[  [target] + top_vars]
    df_test  = test[[target] + top_vars]

    X_train_raw = df_train.drop(target, axis=1)
    X_test_raw  = df_test.drop(target, axis=1)
    y_train_all = df_train[target]
    y_test_all  = df_test[target]

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

    # Drop NaN GDP rows from training (same as benchmark obs_mask)
    obs_mask = y_train_all.notna()
    X_train_pca = X_train_pca[obs_mask]
    y_train_fit = y_train_all[obs_mask]

    # Drop NaN GDP rows from test when computing RMSE
    obs_mask_test = y_test_all.notna()
    X_test_eval  = X_test_pca[obs_mask_test]
    y_test_eval  = y_test_all[obs_mask_test]

    final_model = LinearRegression()
    final_model.fit(X_train_pca, y_train_fit)
    y_pred = final_model.predict(X_test_eval)

    rmse = float(np.sqrt(np.mean((y_test_eval.values - y_pred) ** 2)))
    return top_vars, rmse

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

def lagged_target(data, total_lags, metadata, target_variable=target_variable) :
    if total_lags == 0 :
        return data
    else :
        for l in range(1,total_lags+1) :
            data[f"{target_variable}_l{l}"] = data[target_variable]
            data.loc[data.index[-metadata[metadata['ticket']==target_variable]['months_lag'].values[0]:], f"{target_variable}_l{l}"] = np.nan
            data[f"{target_variable}_l{l}"] = data[f"{target_variable}_l{l}"].ffill()
            data[f"{target_variable}_l{l}"] = data[f"{target_variable}_l{l}"].shift(1)
            data[f"{target_variable}_l{l}"] = data[f"{target_variable}_l{l}"].fillna(data[f"{target_variable}_l{l}"].mean())
        return data

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
    result = pd.DataFrame(
        {
            "actuals": values ,
            "two_back": pred_values[-2] ,
            "one_back": pred_values[-1] ,
            "zero_back": pred_values[0] ,
            "one_ahead": pred_values[1] ,
            "two_ahead": pred_values[2] ,
            "estimator": model_name ,
            "spec": specification ,
        }
    )
    result.index = pd.to_datetime(dates)
    return result

def fit_ols(
    ytrain ,
    xtrain ,
    target_variable = target_variable,
    ) :
    
    model = LinearRegression()
    return model.fit(xtrain, ytrain) , xtrain.columns

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
    ) :
    
    model = RidgeCV( alphas = alphas )
    return model.fit(xtrain, ytrain) , xtrain.columns

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
    }
    ) :

    model = ElasticNet(alpha = params['alpha'] , l1_ratio = params['l1_ratio'] )
    return model.fit(xtrain, ytrain) , xtrain.columns

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
    ) :
    
    model = Lasso( alpha = alpha )
    
    return model.fit(xtrain, ytrain) , xtrain.columns

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
    ) :
    
    models = []
    for i in range(ModelN):
        model = DecisionTreeRegressor(criterion = "absolute_error", 
                                      min_samples_split = 6, 
                                      min_samples_leaf = 2)
        
        model.fit(xtrain, ytrain)
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
    ) :

    models = []
    for i in range(ModelN):
        model = RandomForestRegressor(
            n_estimators=n_estimators,  
            criterion = "absolute_error", 
            max_features=min(18, xtrain.shape[1]),   # cap: a trimmed spec can have <18 predictors
            min_samples_split=4, 
            min_samples_leaf=2
            )
        
        model.fit(xtrain, ytrain)
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
    ) :
        
    models = []
    for i in range(ModelN):
        model = GradientBoostingRegressor(
                    n_estimators=100, 
                    learning_rate=learning, 
                    loss='absolute_error', 
                    min_samples_split=6, 
                    min_samples_leaf=3
                )
        
        model.fit(xtrain, ytrain)
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
