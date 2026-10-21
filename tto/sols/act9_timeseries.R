# =============================================================================
# Activity 9 -- Classical mixed-frequency models: DFM, MIDAS, MF-BVAR, MIDAS-ML (R)
# GDP Growth Nowcasting Workshop -- IDB, Port of Spain
# Tuesday · Block 4 · about 20-30 minutes (the whole script runs in about a minute)
#
# Packages: tidyverse, nowcastDFM, midasr, mfbvar, midasml, imputeTS
#
# What you do:
#   Blocks 1-2 squeezed every monthly indicator into a quarterly average. The models of
#   the theory block ("Classical Macro-Econometric Nowcasting Models") use the MONTHLY
#   data directly. You estimate them on the same train window (to 2023Q1), nowcast the
#   same 12 test quarters (2023Q2-2026Q1), and add their test RMSE to the shared table:
#     - DFM      one latent factor behind all 15 indicators, Kalman filter (nowcastDFM)
#     - MIDAS    GDP on the monthly lags of one indicator, nealmon lag polynomial (midasr)
#     - MF-BVAR  a small monthly VAR with GDP as a latent monthly series (mfbvar)
#     - MIDAS-ML sparse-group LASSO over all indicators and their lags (midasml)
#   Section marked YOUR TURN: the MIDAS indicators and number of lags.
#   Solution version: the YOUR TURN holds the worked choice, with the reason.
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# install.packages(c("tidyverse", "nowcastDFM", "midasr", "midasml", "imputeTS"))
# mfbvar: install.packages("mfbvar"), or devtools::install_github("ankargren/mfbvar")
#         (needs Rtools on Windows; see install/act0_env_check_R.Rmd)

library(tidyverse)   # readr, dplyr, ggplot2 ...
library(midasr)      # midas_r(), mls(), nealmon()
# nowcastDFM, mfbvar, midasml and imputeTS are called with pkg::function() and not
# attached: nowcastDFM attaches the 'matlab' and 'signal' packages, which mask base
# functions such as sum() and dplyr::filter().

# Work from the folder that holds this script, so that ../data/ resolves.
# In VS Code / RStudio: Session -> Set Working Directory -> To Source File Location.
getwd()

set.seed(42)

# --- Helpers (the same in act6, act7 and act9) --------------------------------

rmse <- function(observed, predicted) sqrt(mean((observed - predicted)^2))

results_file <- "../output/workshop_results.csv"

# Add (or replace) one row of the workshop results table, shared with the Python notebooks.
# A row with the same activity + language + model is replaced; all other rows are kept.
save_result <- function(block, activity, model, rmse_test, notes = "") {
  dir.create(dirname(results_file), showWarnings = FALSE, recursive = TRUE)
  row <- data.frame(block = block, activity = activity, language = "R", model = model,
                    rmse_test = round(rmse_test, 4), notes = notes)
  if (file.exists(results_file)) {
    old <- read.csv(results_file, stringsAsFactors = FALSE)
    old <- old[!(old$activity == activity & old$language == "R" & old$model == model),
               names(row)]
    row <- rbind(old, row)
  }
  write.csv(row, results_file, row.names = FALSE)
  invisible(row)
}

# Everything saved so far (both languages), best first
show_results <- function() {
  if (!file.exists(results_file)) return(cat("No results saved yet.\n"))
  r <- read.csv(results_file, stringsAsFactors = FALSE)
  r[order(r$rmse_test), ]
}

# nowcastDFM prints a line every 20 EM iterations; this keeps the console readable
quietly <- function(expr) {
  out <- NULL
  invisible(capture.output(out <- suppressMessages(suppressWarnings(expr))))
  out
}

# =============================================================================
# 1. DATA: MONTHLY INDICATORS, QUARTERLY GDP, RAGGED EDGE
# =============================================================================
# blocks_monthly.csv: the same 15 predictors as Blocks 1-2, monthly % growth (seasonally
# adjusted), 2013-01 to 2026-10. RGDP0000 (real GDP growth, q/q %) appears only on the
# last month of each quarter. blocks_quarterly.csv gives the train/test split.

m    <- read_csv("../data/blocks_monthly.csv", show_col_types = FALSE)
q    <- read_csv("../data/blocks_quarterly.csv", show_col_types = FALSE)
meta <- read_csv("../data/blocks_meta.csv", show_col_types = FALSE)

predictors <- meta$ticker
labels     <- setNames(meta$label, meta$ticker)

train_end <- max(q$date[q$sample == "train"])   # 2023-03-01
test_q    <- q$date[q$sample == "test"]         # 12 quarter-end months
y_train   <- q$RGDP0000[q$sample == "train"]
y_test    <- q$RGDP0000[q$sample == "test"]

# The ragged edge: each indicator stops in a different month. This is the problem the
# theory block opened with: GDP is out to 2026Q1 only, the indicators run further.
m %>%
  pivot_longer(-date, names_to = "ticker") %>%
  filter(!is.na(value)) %>%
  group_by(ticker) %>%
  summarise(last_month = max(date)) %>%
  mutate(label = labels[ticker]) %>%
  arrange(last_month)

# The months the classical models see for one quarter: GDP observed every third month
m %>% filter(date >= as.Date("2023-01-01"), date <= as.Date("2023-06-01")) %>%
  select(date, RGDP0000, RENE5005, UNTL0003, MCRD5000)

# Benchmark every model must beat: predict the train-sample mean every quarter
rmse_naive <- rmse(y_test, mean(y_train))
cat(sprintf("Naive (train mean) test RMSE: %.4f\n", rmse_naive))
save_result(4, "act9", "Mean (train)", rmse_naive, "naive benchmark: mean GDP growth 2013Q1-2023Q1")

# =============================================================================
# 2. DYNAMIC FACTOR MODEL (nowcastDFM)
# =============================================================================
# Measurement:  x_t = Lambda f_t + e_t     (every indicator loads on the common factor)
# State:        f_t = A_1 f_{t-1} + u_t    (the factor follows a VAR(p))
# The whole system runs at MONTHLY frequency; quarterly GDP is a monthly latent series
# observed every third month, so the Kalman filter fills the missing months and the
# ragged edge. Estimation: PCA start, then EM iterations (dfm() does it all).
#
#   data      date column first, then the series; nowcastDFM detects that RGDP0000 is
#             quarterly because it is only observed in Mar/Jun/Sep/Dec
#   blocks    default: one global factor (what the Barbados script uses; T is small)
#   p         lags of the factor VAR

dfm_vars <- predictors   # all 15 indicators

panel <- m %>% select(date, RGDP0000, all_of(dfm_vars)) %>% as.data.frame()
panel_train <- panel %>% filter(date <= train_end)

t0 <- Sys.time()
dfm_fit <- quietly(nowcastDFM::dfm(panel_train, p = 1, max_iter = 1500, threshold = 1e-5))
cat(sprintf("DFM estimated in %.0f s: %d EM iterations, converged = %s\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs")),
            dfm_fit$num_iter, dfm_fit$convergence == 1))

# Loadings on the global factor (standardized data). Same sign = moves with the cycle.
# GDP is quarterly, so it loads on the factor and its lags with weights 1-2-3-2-1;
# the first one is shown.
# (dfm() puts the monthly series first and the quarterly one last, naming it "quarterlies")
dfm_series <- sub("^quarterlies$", "RGDP0000", dfm_fit$model)
tibble(series = dfm_series, label = c(labels, RGDP0000 = "Real GDP growth")[dfm_series],
       loading = round(dfm_fit$C[, 1], 3)) %>%
  arrange(desc(abs(loading)))

# The estimated factor against GDP growth (both standardized)
tibble(date = panel_train$date, factor = dfm_fit$Z[, 1],
       gdp = (panel_train$RGDP0000 - mean(y_train)) / sd(y_train)) %>%
  ggplot(aes(date)) +
  geom_line(aes(y = factor / sd(factor), color = "Common factor (monthly)")) +
  geom_point(aes(y = gdp, color = "Real GDP growth (quarterly)"), na.rm = TRUE) +
  scale_color_manual(values = c("#2980B9", "black")) +
  labs(title = "DFM: the common factor behind the 15 indicators, train sample",
       x = NULL, y = "Standardized", color = NULL) +
  theme_minimal()

# --- Pseudo-real-time nowcasts over the test quarters ----------------------------
# Parameters stay fixed at the train estimates (like every model in Blocks 1-2). For each
# test quarter we take the data available up to the quarter's last month, hide that
# quarter's GDP, and let the Kalman filter (predict_dfm) produce the nowcast.
# Simplification: every indicator is assumed published by the end of the quarter; a
# full real-time exercise would also cut each series at its own release lag.

dfm_nowcast <- function(d) {
  vintage <- panel %>% filter(date <= d)
  vintage$RGDP0000[vintage$date == d] <- NA
  filled  <- quietly(nowcastDFM::predict_dfm(vintage, dfm_fit))
  filled$RGDP0000[filled$date == d]
}
pred_dfm <- sapply(test_q, dfm_nowcast)
rmse_dfm <- rmse(y_test, pred_dfm)
cat(sprintf("DFM test RMSE: %.4f   (naive: %.4f)\n", rmse_dfm, rmse_naive))

save_result(4, "act9", "DFM", rmse_dfm,
            "nowcastDFM, 1 global factor, p=1, 15 indicators, fixed on train, Kalman nowcast")

# --- The live nowcast: the ragged edge today --------------------------------------
# With ALL the data to October 2026 (GDP is out to 2026Q1 only), the same model nowcasts
# 2026Q2 and 2026Q3 and forecasts 2026Q4. Missing months are filled by the Kalman filter.
live <- quietly(nowcastDFM::predict_dfm(panel, dfm_fit, months_ahead = 3))
live %>%
  filter(date %in% as.Date(c("2026-06-01", "2026-09-01", "2026-12-01"))) %>%
  transmute(quarter = paste0(format(date, "%Y"), "Q", (as.integer(format(date, "%m")) + 2) %/% 3),
            dfm_nowcast = round(RGDP0000, 2))

# TRY IT (optional): p = 2 in dfm(), or a smaller panel in dfm_vars (e.g. energy and
# credit only). Re-run this section and compare the test RMSE. Each fit takes ~5-20 s.

# =============================================================================
# 3. MIDAS (midasr)
# =============================================================================
# Regress quarterly GDP directly on the MONTHLY lags of one indicator, with the lag
# weights tied to a smooth polynomial (two parameters instead of one per lag):
#   y_q = b0 + b1 * sum_l w(l; theta) x_{3q - l} + e_q
#   mls(x, 0:5, 3, nealmon)   lags 0..5 months back from the quarter's last month
#                             (6 monthly lags: this quarter and the previous one),
#                             frequency ratio 3, exponential Almon weights
#   start = list(x = c(1, -0.5))   nealmon starting values: scale 1, gentle decay
# Estimation is non-linear least squares, ONCE on the train window (fixed parameters).
# Each test quarter is nowcast with its own three months of data (complete quarter).

# Monthly data to the last test quarter. MIDAS needs a complete monthly series, so the
# few interior gaps (night lights in 2017, 2023, 2024) are linearly interpolated.
mx <- m %>% filter(date <= max(test_q)) %>%
  mutate(across(all_of(predictors), ~ imputeTS::na_interpolation(.x)))
n_train <- length(y_train)
first_test_month <- 3 * n_train + 1   # monthly row where the test sample starts

midas_one <- function(ticker, n_lags) {
  x <- mx[[ticker]]
  dat <- list(y = y_train, x = x[1:(3 * n_train)])
  fit <- midas_r(y ~ mls(x, 0:(n_lags - 1), 3, nealmon), data = dat,
                 start = list(x = c(1, -0.5)))
  pred <- forecast(fit, newdata = list(x = x[first_test_month:length(x)]))$mean
  list(fit = fit, pred = as.numeric(pred))
}

# --- One indicator in detail: LNG production, 6 monthly lags ----------------------
lng <- midas_one("RENE5005", 6)
summary(lng$fit)
cat(sprintf("MIDAS (LNG, 6 lags) test RMSE: %.4f\n", rmse(y_test, lng$pred)))

# The implied weight on each monthly lag (nealmon: scale x exp(theta * lag), normalized)
theta_hat <- coef(lng$fit)[c("x1", "x2")]
tibble(lag = 0:5, coefficient = nealmon(theta_hat, 6)) %>%
  ggplot(aes(lag, coefficient)) +
  geom_col(fill = "#2980B9") +
  scale_x_continuous(breaks = 0:5, labels = c("0\n(last month\nof quarter)", 1:5)) +
  labs(title = "MIDAS, LNG production: coefficient on each monthly lag (nealmon)",
       x = "Months back from the quarter's last month", y = "Coefficient") +
  theme_minimal()

# --- Exploring indicators and lags -------------------------------------------------
# Three candidates with an economic story for T&T: LNG production (energy output),
# night lights over oil & gas sites (satellite activity), private sector credit.
# Train RMSE is in-sample; test RMSE is shown to learn from, not to pick on.
exploration_midas <- expand_grid(ticker = c("RENE5005", "UNTL0003", "MCRD5000"),
                                 n_lags = c(3, 6)) %>%
  mutate(res = map2(ticker, n_lags, midas_one),
         label      = labels[ticker],
         theta      = map_dbl(res, ~ round(coef(.x$fit)[["x2"]], 2)),
         train_rmse = map_dbl(res, ~ round(sqrt(mean(residuals(.x$fit)^2)), 4)),
         test_rmse  = map_dbl(res, ~ round(rmse(y_test, .x$pred), 4))) %>%
  select(-res)
exploration_midas
# theta < 0: weights decay with the lag (recent months matter most). A large positive
# theta puts all the weight on the oldest lag: usually a sign NLS wandered to an odd
# optimum (the theory block: "NLS can converge to a local minimum").

# ---- YOUR TURN: fix the final MIDAS ------------------------------------------
# Choose the indicator(s) and the number of monthly lags. With several indicators, one
# MIDAS is fitted per indicator and the nowcasts are averaged (as in the Barbados script).
# Worked choice: all three indicators, 6 lags each, averaged. No single indicator is
# reliable in 41 quarters, and averaging single-indicator MIDAS nowcasts is the standard
# hedge (slide 27). LNG alone with 3 lags has a lower test RMSE, but choosing the best
# cell of the test column would be fitting the 12 test quarters.
midas_vars <- c("RENE5005", "UNTL0003", "MCRD5000")
n_lags     <- 6
# ------------------------------------------------------------------------------

midas_preds <- sapply(midas_vars, function(v) midas_one(v, n_lags)$pred)
pred_midas  <- rowMeans(matrix(midas_preds, nrow = length(test_q)))
rmse_midas  <- rmse(y_test, pred_midas)
cat(sprintf("MIDAS test RMSE: %.4f   (naive: %.4f)\n", rmse_midas, rmse_naive))

save_result(4, "act9", "MIDAS", rmse_midas,
            sprintf("midasr nealmon, %d monthly lags, avg of %s, fixed on train",
                    n_lags, paste(midas_vars, collapse = "+")))

# =============================================================================
# 4. MF-BVAR (mfbvar) -- about 15 seconds
# =============================================================================
# A VAR on the monthly indicators AND GDP, written at monthly frequency, with GDP an
# unobserved monthly series whose quarterly average is observed. A Minnesota prior
# (lambda1 = overall tightness) shrinks the many VAR coefficients; a Gibbs sampler draws
# the parameters and the missing months.
# Unlike the DFM and MIDAS above, mfbvar has no "filter new data with old parameters"
# step, so the model is RE-ESTIMATED for each test quarter on all data up to that
# quarter's last month (expanding window; GDP known up to the previous quarter). It also
# uses GDP's own past, which the other models do not.

bvar_vars <- c("RENE5005", "UNTL0003", "MCRD5000")   # K = 4 with GDP: keep it small

bvar_nowcast <- function(d) {
  v <- mx %>% filter(date <= d)
  Y <- lapply(bvar_vars, function(s) ts(v[[s]], start = c(2013, 1), frequency = 12))
  names(Y) <- bvar_vars
  Y$GDP <- ts(q$RGDP0000[q$date < d], start = c(2013, 1), frequency = 4)  # quarterly last
  prior <- mfbvar::set_prior(Y = Y, n_lags = 4, n_reps = 1000, n_burnin = 500,
                             n_fcst = 1, lambda1 = 0.2)
  fit   <- mfbvar::estimate_mfbvar(prior, prior = "minn", variance = "iw")
  draws <- predict(fit, pred_bands = NULL)
  mean(draws$fcst[draws$variable == "GDP" & draws$fcst_date == d])
}

set.seed(42)
t0 <- Sys.time()
pred_bvar <- sapply(test_q, bvar_nowcast)
rmse_bvar <- rmse(y_test, pred_bvar)
cat(sprintf("MF-BVAR test RMSE: %.4f   (%.0f s)\n", rmse_bvar,
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))

save_result(4, "act9", "MF-BVAR", rmse_bvar,
            "mfbvar Minnesota lambda1=0.2, 4 lags, LNG+NTL O&G+credit+GDP, re-estimated each quarter")

# =============================================================================
# 5. MIDAS-ML (midasml) -- sparse-group LASSO over all indicators and lags
# =============================================================================
# 15 indicators x 6 monthly lags = 90 regressors for 40 quarters. Each indicator's lags
# are compressed with a Legendre polynomial of degree 2 (3 columns per indicator, the
# MIDAS idea), and sparse-group LASSO drops whole indicators (group penalty) and single
# columns inside the survivors (LASSO penalty). gamma = 0.5 weighs the two; lambda by CV
# with the same contiguous folds as Block 1. Fixed estimation on the train window.

n_lags_ml <- 6
W <- midasml::lb(2, a = 0, b = 1, jmax = n_lags_ml)        # 6 lags -> 3 Legendre columns
X_ml <- do.call(cbind, lapply(predictors, function(v)
  mls(mx[[v]], 0:(n_lags_ml - 1), 3) %*% W))
gindex <- rep(seq_along(predictors), each = ncol(W))
y_all  <- q$RGDP0000[q$date <= max(test_q)]

rows_train <- which(seq_along(y_all) <= n_train & complete.cases(X_ml))  # 2013Q1 has no lags
rows_test  <- (n_train + 1):length(y_all)
foldid_ml  <- cut(seq_along(rows_train), 5, labels = FALSE)

set.seed(42)
sgl <- midasml::cv.sglfit(x = X_ml[rows_train, ], y = y_all[rows_train], gamma = 0.5,
                          gindex = gindex, foldid = foldid_ml, nfolds = 5)
pred_ml <- as.numeric(predict(sgl, newx = X_ml[rows_test, ], s = "lam.min"))
rmse_ml <- rmse(y_test, pred_ml)

beta_ml <- sgl$cv.fit$lam.min$beta
kept <- unique(gindex[abs(as.numeric(beta_ml)) > 1e-10])
cat(sprintf("MIDAS-ML test RMSE: %.4f   indicators kept: %s\n", rmse_ml,
            if (length(kept)) paste(labels[predictors[kept]], collapse = ", ") else "none"))

save_result(4, "act9", "MIDAS-ML", rmse_ml,
            "midasml sg-LASSO gamma=0.5, 15 indicators x 6 lags, Legendre deg 2, contiguous 5-fold CV")

# =============================================================================
# 6. OBSERVED VS NOWCASTS ON THE TEST QUARTERS
# =============================================================================

tibble(date = test_q, Observed = y_test, DFM = pred_dfm, MIDAS = pred_midas,
       `MF-BVAR` = pred_bvar, `MIDAS-ML` = pred_ml) %>%
  pivot_longer(-date, names_to = "series", values_to = "growth") %>%
  ggplot(aes(date, growth, color = series, linewidth = series == "Observed")) +
  geom_hline(yintercept = 0, color = "grey50") +
  geom_line() + geom_point() +
  scale_color_manual(values = c(Observed = "black", DFM = "#2980B9", MIDAS = "#C0392B",
                                `MF-BVAR` = "#27AE60", `MIDAS-ML` = "#E67E22")) +
  scale_linewidth_manual(values = c(0.7, 1.3), guide = "none") +
  labs(title = "Classical mixed-frequency nowcasts on the test quarters",
       x = NULL, y = "Real GDP growth, q/q %", color = NULL) +
  theme_minimal()

# =============================================================================
# 7. RESULTS TABLE (shared with the Python notebooks)
# =============================================================================
# Compare with Blocks 1-2. Keep in mind how each model used the data: DFM, MIDAS and
# MIDAS-ML were estimated once on the train window; MF-BVAR was re-estimated every quarter
# and also uses GDP's own lags.

show_results()
