# ============================================================
# GDP Nowcasting — Training Exercise
# MIDAS · MF-BVAR · DFM — Barbados
# IDB Caribbean Country Department
# ============================================================
# Run this script section by section, following along with
# the presentation slides and the HTML notebook.
# Each section header matches a section in the notebook.
# ============================================================


# ---- 0. Libraries ----------------------------------------------------------

#install.packages(c("Rmisc, nowcastDFM, imputeTS, midasr, mfbvar"))

library(Rmisc)        # CI(): confidence intervals for the BVAR prior
library(nowcastDFM)   # dfm(), predict_dfm(): Dynamic Factor Model
library(tidyverse)    # data manipulation and plotting
library(imputeTS)     # na_mean(): fill NAs with column means
library(midasr)       # midas_r(), mls(): MIDAS regression
library(mfbvar)       # set_prior(), estimate_mfbvar(): Bayesian VAR
library(lubridate)    # date arithmetic


# ---- 1. Parameters ---------------------------------------------------------

PROJECT <- "C:/Users/agmaz/Desktop/Nwcst Training R/Barbados"  # <- update if needed
PATH_DATA   <- PROJECT
PATH_OUTPUT <- PROJECT

target_variable  <- "RGDP0000"
lags             <- seq(-2, 2)           # vintages: -2 (earliest) to +2 (latest)
train_start_date <- as.Date("2010-04-01")
test_start_date  <- as.Date("2021-09-01")
test_end_date    <- as.Date("2025-12-01")

XVars <- c("RTOU1000",   # stayover arrivals   — pub lag: 5 months
           "TREX0000",   # re-exports          — pub lag: 5 months
           "UGTR0000",   # Google Trends       — pub lag: 0 months
           "UNTL0000")   # nighttime lights    — pub lag: 0 months


# ---- 2. Data ---------------------------------------------------------------

data_raw     <- read.csv(file.path(PATH_DATA, "data_processed_BB.csv"))
data_raw     <- data_raw[-c(nrow(data_raw), nrow(data_raw) - 1), ]  # drop incomplete quarters
metadata_raw <- read.csv(file.path(PATH_DATA, "meta_processed_BB.csv"))

data <- data_raw %>%
  mutate(date = as.Date(date)) %>%
  select(date, all_of(c(target_variable, XVars)))

metadata <- metadata_raw %>%
  dplyr::filter(ticket %in% colnames(data))

q_vars <- metadata %>%
  dplyr::filter(freq == "quarterly", ticket != target_variable) %>%
  pull(ticket)

# Inspect the data structure
str(data)
metadata %>% select(ticket, freq, months_lag)


# ---- 3. Shared infrastructure ----------------------------------------------

# gen_lagged_data(): reconstruct the dataset as it looked at any past date.
# For each series, the last `months_lag` observations are set to NA,
# shifted by the vintage `lag`. Identical across all three models.

gen_lagged_data <- function(metadata, data, last_date, lag) {
  lagged_data <- data %>% dplyr::filter(date <= last_date)

  for (col in colnames(lagged_data)[-1]) {
    pub_lag   <- metadata %>% dplyr::filter(ticket == col) %>% pull(months_lag)
    condition <- nrow(lagged_data) - pub_lag + lag  # first row to set as NA
    if (condition <= nrow(lagged_data)) {
      lagged_data[condition:nrow(lagged_data), col] <- NA
    }
  }
  lagged_data %>% dplyr::filter(!is.na(date))
}

# Test window and prediction containers
test       <- data %>% dplyr::filter(date >= train_start_date & date <= test_end_date)
test_dates <- seq(test_start_date, test_end_date, by = "3 months")
actuals    <- test %>% dplyr::filter(date %in% test_dates) %>% pull(target_variable)

pred_midas <- data.frame(date = test_dates)
pred_bvar  <- data.frame(date = test_dates)
pred_dfm   <- data.frame(date = test_dates)
for (lag in lags) {
  pred_midas[, as.character(lag)] <- NA
  pred_bvar[,  as.character(lag)] <- NA
  pred_dfm[,   as.character(lag)] <- NA
}

# See the vintage effect: same data, two different points in time
tail(gen_lagged_data(metadata, test, test_dates[1], -2), 6)  # lag -2
tail(gen_lagged_data(metadata, test, test_dates[1],  2), 6)  # lag +2


# ============================================================
# MODEL 1: MIDAS
# ============================================================
# One bivariate regression per indicator.
# mls() bridges the frequency gap; nealmon keeps the lag
# structure parsimonious. Predictions combined via
# inverse in-sample RMSE weights.
# NA handling: na_mean() fills everything before estimation.
# ============================================================

for (i in seq_along(test_dates)) {
  cat("MIDAS | Test date:", as.character(test_dates[i]), "\n")

  # Training slice: data up to one quarter before the test date, all NAs filled
  train <- test %>%
    dplyr::filter(date <= test_dates[i] %m-% months(3)) %>%
    na_mean()

  # Quarterly GDP rows only (substr pulls month digits from "YYYY-MM-DD")
  y <- train[substr(train$date, 6, 7) %in% c("03","06","09","12"), target_variable]

  # One MIDAS model per indicator
  models <- list()
  for (col in XVars) {
    if (col %in% q_vars) {
      x <- train[substr(train$date, 6, 7) %in% c("03","06","09","12"), col]
      models[[col]] <- midas_r(y ~ mls(x, 0:1, 1, nealmon), start = list(x = c(1, -0.5)))
    } else {
      x <- train[, col]
      models[[col]] <- midas_r(y ~ mls(x, 0:3, 3, nealmon), start = list(x = c(1, -0.5)))
    }
  }

  # Inverse-RMSE weights: worst in-sample fit gets weight ~0
  weight <- sapply(XVars, function(col) {
    sqrt(mean((models[[col]]$fitted.values - y[2:length(y)])^2))
  })
  adj    <- abs(weight - max(weight))   # distance from worst
  weight <- adj / sum(adj)              # normalize to sum to 1

  # Predict on each vintage
  for (lag in lags) {
    lagged_data <- gen_lagged_data(metadata, test, test_dates[i], lag) %>% data.frame()
    lagged_data[lagged_data$date == test_dates[i], target_variable] <- NA  # hide actual
    lagged_data <- na_mean(lagged_data)  # fill before prediction

    preds <- sapply(XVars, function(col) {
      x <- if (col %in% q_vars) {
        lagged_data[substr(lagged_data$date, 6, 7) %in% c("03","06","09","12"), col]
      } else { lagged_data[, col] }
      p <- forecast(models[[col]], newdata = list(x = x))$mean
      p[length(p)]  # last element = current quarter forecast
    })

    pred_midas[pred_midas$date == test_dates[i], as.character(lag)] <-
      weighted.mean(preds, weight)
  }
}

# MIDAS performance
perf_midas <- data.frame()
for (lag in lags) {
  perf_midas <- rbind(perf_midas,
    data.frame(Vintage = lag,
               RMSE    = sqrt(mean((actuals - pred_midas[[as.character(lag)]])^2)),
               Model   = "MIDAS"))
}
print(perf_midas)


# ============================================================
# MODEL 2: MF-BVAR
# ============================================================
# One joint Bayesian VAR over all series.
# Minnesota prior + steady-state prior via MCMC (200 draws).
# NA handling: interior gaps filled; ragged edge preserved —
# mfbvar reads staggered trailing NAs natively.
# ============================================================

# Helper: fill interior NAs up to each series' last observation;
# leave the trailing NA run (ragged edge) intact for mfbvar
fill_keep_ragged_edge <- function(df) {
  for (col in setdiff(colnames(df), "date")) {
    x <- df[[col]]
    if (any(!is.na(x))) {
      last_obs      <- max(which(!is.na(x)))    # index of last real observation
      x[1:last_obs] <- na_mean(x[1:last_obs])   # fill interior gaps only
      df[[col]]     <- x                        # trailing NAs stay untouched
    }
  }
  df
}

# Helper: convert data frame to a named list of ts objects.
# mfbvar requires: monthly → freq 12, quarterly → freq 4,
# quarterly variables LAST in the list.
make_mf_list <- function(df, metadata, target_variable) {
  mf       <- list()
  q_months <- c("03", "06", "09", "12")

  for (col in setdiff(colnames(df), "date")) {
    freq_lab <- metadata %>% dplyr::filter(ticket == col) %>% pull(freq)
    if (freq_lab == "quarterly") {
      sub       <- df %>% dplyr::filter(substr(date, 6, 7) %in% q_months)
      first     <- sub$date[1]
      mf[[col]] <- ts(sub[[col]], start = c(year(first), quarter(first)), frequency = 4)
    } else {
      first     <- df$date[1]
      mf[[col]] <- ts(df[[col]], start = c(year(first), month(first)), frequency = 12)
    }
  }

  # Enforce quarterly-last ordering (hard requirement of mfbvar)
  q_cols <- metadata %>%
    dplyr::filter(freq == "quarterly", ticket %in% names(mf)) %>%
    pull(ticket)
  mf[c(setdiff(names(mf), q_cols), q_cols)]
}

# Helper: build priors and run the MCMC estimator
estimate_bvar <- function(mf_list, target_variable) {

  # Minnesota prior: shrinks VAR coefficients toward a random walk
  prior <- set_prior(Y = mf_list, n_lags = 3, n_reps = 200,
                     block_exo = c(target_variable))

  # Steady-state prior: anchors long-run means to historical 95% CI
  # na.omit() prevents ragged-edge NAs from producing NA prior bounds
  c_interval      <- t(sapply(mf_list,
                    function(s) CI(na.omit(as.numeric(s)), ci = 0.95)))
  prior_intervals <- c_interval[, c("upper", "lower")]
  moments         <- interval_to_moments(prior_intervals)

  prior <- update_prior(prior, d = "intercept",
                        prior_psi_mean  = moments$prior_psi_mean,
                        prior_psi_Omega = moments$prior_psi_Omega)
  prior <- update_prior(prior, n_fcst = 12)  # 12-month forecast horizon

  # Estimate: Minnesota prior, inverse-Wishart error covariance
  estimate_mfbvar(prior, prior = "minn", variance = "iw")
}

set.seed(2025)  # MCMC is stochastic — fix seed for reproducibility

for (i in seq_along(test_dates)) {
  current_date <- test_dates[i]
  cat("BVAR  | Test date:", as.character(current_date), "\n")

  for (lag in lags) {
    lagged_data <- gen_lagged_data(metadata, test, current_date, lag) %>% data.frame()
    lagged_data[lagged_data$date == current_date, target_variable] <- NA

    lagged_data <- fill_keep_ragged_edge(lagged_data)  # fill interior, preserve tail
    lagged_data <- lagged_data[                        # drop rows that are all NA
      rowSums(is.na(lagged_data)) != ncol(lagged_data) - 1, ]

    mf_test <- make_mf_list(lagged_data, metadata, target_variable)
    model   <- estimate_bvar(mf_test, target_variable)

    # Average MCMC draws for the target quarter
    prediction <- predict(model, pred_bands = NULL) %>%
      dplyr::filter(variable == target_variable, fcst_date == current_date) %>%
      pull(fcst) %>% mean()

    pred_bvar[pred_bvar$date == current_date, as.character(lag)] <- prediction
  }
}

# BVAR performance
perf_bvar <- data.frame()
for (lag in lags) {
  perf_bvar <- rbind(perf_bvar,
    data.frame(Vintage = lag,
               RMSE    = sqrt(mean((actuals - pred_bvar[[as.character(lag)]])^2)),
               Model   = "BVAR"))
}
print(perf_bvar)


# ============================================================
# MODEL 3: DFM
# ============================================================
# All series driven by latent factors estimated via EM + Kalman.
# NA handling: none — the Kalman filter handles everything.
# Estimated ONCE; only the input vintage changes per prediction.
# ============================================================

# One global factor shared by all series (one row per series)
blocks <- data.frame(block_1 = rep(1, ncol(data) - 1))

# Estimate on data up to one quarter before the first test date
train_dfm  <- test %>% dplyr::filter(date <= test_dates[1] %m-% months(3))
output_dfm <- dfm(data = train_dfm, blocks = blocks, max_iter = 300, p = 2)

for (i in seq_along(test_dates)) {
  current_date <- test_dates[i]
  cat("DFM   | Test date:", as.character(current_date), "\n")

  for (lag in lags) {
    lagged_data <- gen_lagged_data(metadata, test, current_date, lag) %>% data.frame()
    lagged_data[lagged_data$date == current_date, target_variable] <- NA

    # NAs passed in as-is — Kalman filter handles them
    prediction <- predict_dfm(lagged_data, output_dfm) %>%
      dplyr::filter(date == current_date) %>%
      pull(!!target_variable)

    pred_dfm[pred_dfm$date == current_date, as.character(lag)] <- prediction
  }
}

# DFM performance
perf_dfm <- data.frame()
for (lag in lags) {
  perf_dfm <- rbind(perf_dfm,
    data.frame(Vintage = lag,
               RMSE    = sqrt(mean((actuals - pred_dfm[[as.character(lag)]])^2)),
               Model   = "DFM"))
}
print(perf_dfm)


# ============================================================
# COMPARISON
# ============================================================

all_perf <- rbind(perf_midas, perf_bvar, perf_dfm)

# RMSE table with best model per vintage
all_perf %>%
  pivot_wider(names_from = Model, values_from = RMSE) %>%
  mutate(Best = case_when(
    pmin(MIDAS, BVAR, DFM) == MIDAS ~ "MIDAS",
    pmin(MIDAS, BVAR, DFM) == BVAR  ~ "BVAR",
    TRUE                             ~ "DFM")) %>%
  arrange(Vintage) %>%
  print()

# RMSE by vintage chart
ggplot(all_perf, aes(x = Vintage, y = RMSE, color = Model)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_x_continuous(breaks = -2:2,
    labels = c("-2\n(earliest)", "-1", "0", "+1", "+2\n(latest)")) +
  scale_color_manual(values = c("MIDAS" = "#0D3B7A",
                                "BVAR"  = "#00A3A6",
                                "DFM"   = "#C8102E")) +
  labs(title    = "RMSE by vintage — MIDAS vs BVAR vs DFM",
       subtitle = "Barbados, 2021Q3–2025Q4",
       x = "Vintage", y = "RMSE", color = NULL) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

# Actuals vs nowcasts at vintage 0
data.frame(
  date   = test_dates,
  Actual = actuals,
  MIDAS  = pred_midas[["0"]],
  BVAR   = pred_bvar[["0"]],
  DFM    = pred_dfm[["0"]]
) %>%
  pivot_longer(-date, names_to = "series", values_to = "value") %>%
  mutate(series = factor(series, levels = c("Actual", "MIDAS", "BVAR", "DFM"))) %>%
  ggplot(aes(x = date, y = value, color = series, linetype = series)) +
  geom_line(linewidth = 1.1) +
  geom_point(data = ~dplyr::filter(., series == "Actual"), size = 2.5) +
  scale_color_manual(values = c("Actual" = "black", "MIDAS" = "#0D3B7A",
                                "BVAR" = "#00A3A6", "DFM" = "#C8102E")) +
  scale_linetype_manual(values = c("Actual" = "solid", "MIDAS" = "dashed",
                                   "BVAR" = "dashed", "DFM" = "dashed")) +
  labs(title    = "Actuals vs nowcasts — vintage 0",
       subtitle = "Barbados real GDP growth",
       x = NULL, y = "% growth", color = NULL, linetype = NULL) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")
