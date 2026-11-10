# =============================================================================
# 5_timeseries_models.R: classical mixed-frequency nowcasts (DFM, MIDAS, MF-BVAR, MIDAS-ML)
# Trinidad and Tobago GDP nowcasting pipeline, optional step 5
#
# Reads the pipeline's own panel (data/data.csv and data/meta.csv, rebuilt by
# 2_data_build.ipynb) and estimates the four models of the time-series theory block on
# the same sample as the Python nowcast (config.py: training from 2013-Q1, test from
# 2023-Q2). It is a benchmark run beside the pipeline, not part of its pooled nowcast.
#
#   DFM       one latent monthly factor behind the indicators, Kalman filter (nowcastDFM)
#   MIDAS     GDP on the monthly lags of one indicator, nealmon weights; one model per
#             indicator, nowcasts averaged (midasr)
#   MF-BVAR   a small monthly VAR with GDP as a latent monthly series, Minnesota prior,
#             re-estimated every test quarter (mfbvar)
#   MIDAS-ML  sparse-group LASSO over every indicator and its monthly lags (midasml)
#
# Run from code/ (so that ../data/ and ../output/ resolve):
#   Rscript 5_timeseries_models.R
# or open it in RStudio / VS Code, set the working directory to this file's folder and
# source it. About two minutes; MF-BVAR is the slow part.
#
# Packages: tidyverse, imputeTS, midasr, nowcastDFM (GitHub: dhopp1/nowcastDFM), mfbvar
# (needs Rtools on Windows); midasml is optional (MIDAS-ML is skipped without it).
# See install/act0_env_check_R.Rmd.
#
# Writes output/<YYYYMMDD>_tto_timeseries/:
#   <date> 5_ts Tab nowcasts_test.csv   observed GDP growth and each model's nowcast, test quarters
#   <date> 5_ts Tab rmse.csv            test RMSE of every model and of the training mean
#   <date> 5_ts Tab nowcasts_live.csv   DFM nowcasts for the quarters after the last GDP print
#   <date> 5_ts Fig nowcasts_test.png   observed against the four nowcasts
# =============================================================================

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(midasr))
# nowcastDFM, mfbvar, midasml and imputeTS are called with pkg::function() and not attached:
# nowcastDFM attaches 'matlab' and 'signal', which mask base functions such as sum().

# =============================================================================
# SETTINGS
# =============================================================================
DATA_CSV   <- "../data/data.csv"
META_CSV   <- "../data/meta.csv"
TARGET     <- "RGDP0000"        # real GDP, quarterly, not seasonally adjusted at source
TRAIN_START <- as.Date("2013-03-01")   # first quarter with a growth rate (as config.py)
TEST_START  <- as.Date("2023-06-01")   # first test quarter (as config.py, production run)

# Indicators, chosen for the economic story (the same 15 as the workshop blocks). Any
# ticker in data.csv can be used. MIDAS, MF-BVAR and MIDAS-ML need series that start by
# January of the TRAIN_START year; the DFM accepts any.
DFM_VARS <- c("RENE5005", "XBRT0001", "UNTL0003", "UNTL0000", "RCEM1100", "XUST0001",
              "XUST0002", "MCRD5000", "MCRD3000", "MDEP1000", "MAGG1000", "MRAT1001",
              "FTFM1100", "UGTR0000", "UGTR1026")
MIDAS_VARS  <- c("RENE5005", "UNTL0003", "MCRD5000")   # LNG, night lights over oil and gas, private credit
MIDAS_LAGS  <- 6                                       # monthly lags: this quarter and the previous one
BVAR_VARS   <- c("RENE5005", "UNTL0003", "MCRD5000")   # keep the VAR small: K = 4 with GDP
BVAR_LAGS   <- 4
BVAR_LAMBDA1 <- 0.2                                    # Minnesota overall tightness
ML_VARS     <- DFM_VARS
ML_LAGS     <- 6
DFM_FACTOR_LAGS <- 1

# Nominal series are not deflated: the CPI starts in September 2015, so deflating would cut
# the sample to 2015-Q4. Monthly inflation is about 0.1 to 0.4 percent, so nominal growth is
# a close stand-in. Set DEFLATE <- TRUE to deflate by RCPI0000 anyway.
DEFLATE <- FALSE
CPI     <- "RCPI0000"

DATE_STR <- format(Sys.Date(), "%Y%m%d")
OUT_DIR  <- file.path("..", "output", paste0(DATE_STR, "_tto_timeseries"))
set.seed(42)

# =============================================================================
# 1. DATA: SEASONAL ADJUSTMENT AND GROWTH RATES
# =============================================================================
# The panel holds levels: monthly series on every month, quarterly ones (GDP) on the last
# month of each quarter. Each series that is not seasonally adjusted at source (meta.csv
# sa = 0) is adjusted with the classical additive decomposition at its own frequency (the
# pipeline's method when X-13 is not installed, and the one behind the workshop's block
# datasets), then turned into a growth rate: quarter on quarter for GDP, month on month for
# the indicators.

raw  <- read_csv(DATA_CSV, show_col_types = FALSE) %>% mutate(date = as.Date(date))
meta <- read_csv(META_CSV, show_col_types = FALSE)
all_vars <- unique(c(DFM_VARS, MIDAS_VARS, BVAR_VARS, ML_VARS))
missing  <- setdiff(c(TARGET, all_vars, if (DEFLATE) CPI), names(raw))
if (length(missing)) stop("not in data.csv: ", paste(missing, collapse = ", "))
info   <- meta %>% filter(ticket %in% all_vars) %>% select(ticket, description)
labels <- setNames(info$description, info$ticket)

remove_seasonal <- function(x, freq) {
  # The classical additive decomposition exactly as the Python pipeline runs it (statsmodels
  # seasonal_decompose with extrapolate_trend = "freq"): centred moving-average trend, its
  # missing ends extended by a straight line fitted on the nearest `freq` trend values,
  # seasonal factor = mean detrended value at each position in the cycle, centred on zero.
  # Like the pipeline, it works on the observed values only (gaps dropped, not filled).
  keep <- which(!is.na(x)); s <- x[keep]; n <- length(s)
  if (n < 2 * freq) return(x)
  w <- if (freq %% 2 == 0) c(0.5, rep(1, freq - 1), 0.5) / freq else rep(1, freq) / freq
  trend <- as.numeric(stats::filter(s, w, sides = 2))
  ok <- which(!is.na(trend)); front <- min(ok); back <- max(ok)
  i1 <- front:(min(front + freq, back) - 1); b1 <- coef(lm(trend[i1] ~ i1))
  if (front > 1) trend[1:(front - 1)] <- b1[1] + b1[2] * (1:(front - 1))
  i2 <- max(front, back - freq):(back - 1); b2 <- coef(lm(trend[i2] ~ i2))
  if (back < n) trend[(back + 1):n] <- b2[1] + b2[2] * ((back + 1):n)
  pos <- (seq_len(n) - 1) %% freq
  avg <- tapply(s - trend, pos, mean, na.rm = TRUE); avg <- avg - mean(avg)
  x[keep] <- s - as.numeric(avg[as.character(pos)])
  x
}
is_sa <- function(t) isTRUE(meta$sa[meta$ticket == t] == 1)

panel <- raw %>% select(date, all_of(c(TARGET, all_vars, if (DEFLATE) CPI)))
if (DEFLATE) {
  nominal <- intersect(all_vars, meta$ticket[meta$nominal == 1])
  panel <- panel %>% mutate(across(all_of(nominal), ~ .x / .data[[CPI]] * 100))
}
for (v in all_vars) if (!is_sa(v)) panel[[v]] <- remove_seasonal(panel[[v]], 12)

gdp <- panel %>% filter(!is.na(.data[[TARGET]])) %>% select(date, level = all_of(TARGET))
if (!is_sa(TARGET)) gdp$level <- remove_seasonal(gdp$level, 4)
gdp <- gdp %>% mutate(growth = 100 * (level / lag(level) - 1))

pct <- function(x) { g <- 100 * (x / lag(x) - 1); g[!is.finite(g)] <- NA; g }
m <- panel %>%
  transmute(date, across(all_of(all_vars), pct)) %>%
  left_join(gdp %>% select(date, !!TARGET := growth), by = "date") %>%
  relocate(all_of(TARGET), .after = date) %>%
  filter(date >= seq(TRAIN_START, by = "-2 months", length.out = 2)[2])
last_month <- max(m$date[rowSums(!is.na(m[all_vars])) > 0])
m <- m %>% filter(date <= last_month)

q <- m %>% filter(!is.na(.data[[TARGET]]), date >= TRAIN_START) %>%
  transmute(date, y = .data[[TARGET]], sample = if_else(date >= TEST_START, "test", "train"))
train_end <- max(q$date[q$sample == "train"])
test_q    <- q$date[q$sample == "test"]
y_train   <- q$y[q$sample == "train"]
y_test    <- q$y[q$sample == "test"]
n_train   <- length(y_train)
cat(sprintf("Monthly panel %s to %s | GDP growth %s to %s | train %d quarters, test %d (%s to %s)\n",
            min(m$date), last_month, min(q$date), max(q$date), n_train, length(test_q),
            min(test_q), max(test_q)))

# Series the MIDAS-type models use must cover the whole monthly span of the test window
mx <- m %>% filter(date <= max(test_q))
short <- unique(c(MIDAS_VARS, BVAR_VARS, ML_VARS))
short <- short[sapply(short, function(v) is.na(mx[[v]][2]) || all(is.na(tail(mx[[v]], 3))))]
if (length(short)) stop("these series do not cover ", min(mx$date), " to ", max(mx$date),
                        " (drop them from MIDAS_VARS / BVAR_VARS / ML_VARS): ", paste(short, collapse = ", "))
mx <- mx %>% mutate(across(all_of(all_vars), ~ imputeTS::na_interpolation(.x)))
first_test_month <- 3 * n_train + 1

rmse <- function(observed, predicted) sqrt(mean((observed - predicted)^2))
quietly <- function(expr) { out <- NULL; invisible(capture.output(out <- suppressMessages(suppressWarnings(expr)))); out }
rmse_naive <- rmse(y_test, mean(y_train))

# =============================================================================
# 2. DYNAMIC FACTOR MODEL (nowcastDFM)
# =============================================================================
# x_t = Lambda f_t + e_t, f_t = A f_{t-1} + u_t, at monthly frequency; GDP is a latent
# monthly series observed every third month. Estimated once on the training window (PCA
# start, then EM); each test quarter is nowcast by the Kalman filter on the data up to the
# quarter's last month with that quarter's GDP hidden. Simplification: every indicator is
# taken as published by the end of the quarter (no release-lag cut).

dfm_panel <- m %>% select(date, all_of(TARGET), all_of(DFM_VARS)) %>% as.data.frame()
t0 <- Sys.time()
dfm_fit <- quietly(nowcastDFM::dfm(dfm_panel %>% filter(date <= train_end), p = DFM_FACTOR_LAGS,
                                   max_iter = 1500, threshold = 1e-5))
cat(sprintf("DFM: %d EM iterations, converged = %s, %.0f s\n", dfm_fit$num_iter,
            dfm_fit$convergence == 1, as.numeric(difftime(Sys.time(), t0, units = "secs"))))

dfm_nowcast <- function(d) {
  vintage <- dfm_panel %>% filter(date <= d)
  vintage[[TARGET]][vintage$date == d] <- NA
  filled <- quietly(nowcastDFM::predict_dfm(vintage, dfm_fit))
  filled[[TARGET]][filled$date == d]
}
pred_dfm <- sapply(test_q, dfm_nowcast)

# Live: every month of data to the end of the panel; the quarters after the last print
live <- quietly(nowcastDFM::predict_dfm(dfm_panel, dfm_fit, months_ahead = 3))
live_q <- live %>%
  filter(date > max(q$date), as.integer(format(date, "%m")) %% 3 == 0) %>%
  transmute(quarter = paste0(format(date, "%Y"), "Q", as.integer(format(date, "%m")) %/% 3),
            date, dfm_nowcast = round(.data[[TARGET]], 3))

# =============================================================================
# 3. MIDAS (midasr), one model per indicator, nowcasts averaged
# =============================================================================
# y_q = b0 + b1 * sum_l w(l; theta) x_{3q-l} + e_q, exponential Almon (nealmon) weights,
# nonlinear least squares once on the training window.

midas_one <- function(ticker, n_lags) {
  x <- mx[[ticker]]
  fit <- midas_r(y ~ mls(x, 0:(n_lags - 1), 3, nealmon),
                 data = list(y = y_train, x = x[1:(3 * n_train)]), start = list(x = c(1, -0.5)))
  as.numeric(forecast(fit, newdata = list(x = x[first_test_month:length(x)]))$mean)
}
midas_by_var <- sapply(MIDAS_VARS, midas_one, n_lags = MIDAS_LAGS)
pred_midas   <- rowMeans(matrix(midas_by_var, nrow = length(test_q)))

# =============================================================================
# 4. MF-BVAR (mfbvar), re-estimated every test quarter
# =============================================================================
# A monthly VAR on the indicators and GDP (latent monthly, its quarterly average observed),
# Minnesota prior, Gibbs sampler. No filtering step with fixed parameters exists, so each
# test quarter is a new estimation on the data to that quarter's last month (expanding
# window, GDP known to the previous quarter). Uses GDP's own past.

start_ym <- as.integer(format(min(mx$date), c("%Y", "%m")))
bvar_nowcast <- function(d) {
  v <- mx %>% filter(date <= d)
  Y <- lapply(BVAR_VARS, function(s) ts(v[[s]], start = start_ym, frequency = 12))
  names(Y) <- BVAR_VARS
  Y$GDP <- ts(q$y[q$date < d], start = c(start_ym[1], (start_ym[2] + 2) %/% 3), frequency = 4)
  prior <- mfbvar::set_prior(Y = Y, n_lags = BVAR_LAGS, n_reps = 1000, n_burnin = 500,
                             n_fcst = 1, lambda1 = BVAR_LAMBDA1)
  fit   <- quietly(mfbvar::estimate_mfbvar(prior, prior = "minn", variance = "iw"))
  draws <- predict(fit, pred_bands = NULL)
  mean(draws$fcst[draws$variable == "GDP" & draws$fcst_date == d])
}
t0 <- Sys.time()
pred_bvar <- sapply(test_q, bvar_nowcast)
cat(sprintf("MF-BVAR: %d estimations, %.0f s\n", length(test_q),
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))

# =============================================================================
# 5. MIDAS-ML (midasml): sparse-group LASSO
# =============================================================================
# Each indicator's monthly lags are compressed by a degree-2 Legendre polynomial (3 columns
# per indicator); sparse-group LASSO drops whole indicators and single columns inside the
# survivors (gamma = 0.5 weighs the two); lambda by 5-fold contiguous cross-validation.

# midasml is not in the R environment check; without it this model is skipped.
#   install.packages("midasml")
pred_ml <- rep(NA_real_, length(test_q)); kept <- integer(0)
if (requireNamespace("midasml", quietly = TRUE)) {
  W <- midasml::lb(2, a = 0, b = 1, jmax = ML_LAGS)
  X_ml <- do.call(cbind, lapply(ML_VARS, function(v) mls(mx[[v]], 0:(ML_LAGS - 1), 3) %*% W))
  gindex <- rep(seq_along(ML_VARS), each = ncol(W))
  y_all  <- q$y[q$date <= max(test_q)]
  rows_train <- which(seq_along(y_all) <= n_train & complete.cases(X_ml))
  rows_test  <- (n_train + 1):length(y_all)
  sgl <- midasml::cv.sglfit(x = X_ml[rows_train, ], y = y_all[rows_train], gamma = 0.5, gindex = gindex,
                            foldid = cut(seq_along(rows_train), 5, labels = FALSE), nfolds = 5)
  pred_ml <- as.numeric(predict(sgl, newx = X_ml[rows_test, ], s = "lam.min"))
  kept <- unique(gindex[abs(as.numeric(sgl$cv.fit$lam.min$beta)) > 1e-10])
} else cat("MIDAS-ML skipped: package midasml not installed (install.packages(\"midasml\"))
")

# =============================================================================
# 6. RESULTS
# =============================================================================
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
out_file <- function(name) file.path(OUT_DIR, paste(DATE_STR, name))

nowcasts <- tibble(date = test_q, observed = y_test, DFM = pred_dfm, MIDAS = pred_midas,
                   `MF-BVAR` = pred_bvar, `MIDAS-ML` = pred_ml)
scores <- tibble(
  model = c("Mean (train)", "DFM", "MIDAS", "MF-BVAR", "MIDAS-ML"),
  rmse_test = c(rmse_naive, rmse(y_test, pred_dfm), rmse(y_test, pred_midas),
                rmse(y_test, pred_bvar), rmse(y_test, pred_ml)),
  notes = c(sprintf("mean GDP growth %s to %s", min(q$date), train_end),
            sprintf("nowcastDFM, 1 global factor, p=%d, %d indicators, fixed on train", DFM_FACTOR_LAGS, length(DFM_VARS)),
            sprintf("midasr nealmon, %d lags, average of %s", MIDAS_LAGS, paste(MIDAS_VARS, collapse = "+")),
            sprintf("mfbvar Minnesota lambda1=%.2f, %d lags, %s+GDP, re-estimated each quarter",
                    BVAR_LAMBDA1, BVAR_LAGS, paste(BVAR_VARS, collapse = "+")),
            sprintf("midasml sg-LASSO gamma=0.5, %d indicators x %d lags; kept: %s", length(ML_VARS), ML_LAGS,
                    if (length(kept)) paste(ML_VARS[kept], collapse = " ") else "none"))) %>%
  mutate(rmse_test = round(rmse_test, 4))

write_csv(nowcasts %>% mutate(across(-date, ~ round(.x, 4))), out_file("5_ts Tab nowcasts_test.csv"))
write_csv(scores, out_file("5_ts Tab rmse.csv"))
write_csv(live_q, out_file("5_ts Tab nowcasts_live.csv"))

p <- nowcasts %>%
  pivot_longer(-date, names_to = "series", values_to = "growth") %>%
  filter(!is.na(growth)) %>%
  mutate(series = factor(series, c("observed", "DFM", "MIDAS", "MF-BVAR", "MIDAS-ML"))) %>%
  ggplot(aes(date, growth, color = series, linewidth = series == "observed")) +
  geom_hline(yintercept = 0, color = "grey50") + geom_line() + geom_point() +
  scale_color_manual(values = c(observed = "black", DFM = "#2980B9", MIDAS = "#C0392B",
                                `MF-BVAR` = "#27AE60", `MIDAS-ML` = "#E67E22")) +
  scale_linewidth_manual(values = c(0.7, 1.3), guide = "none") +
  labs(title = "Trinidad and Tobago: classical mixed-frequency nowcasts, test quarters",
       x = NULL, y = "Real GDP growth, q/q %", color = NULL) +
  theme_minimal()
ggsave(out_file("5_ts Fig nowcasts_test.png"), p, width = 9, height = 5, dpi = 150, bg = "white")

cat("\nTest RMSE (", format(min(test_q)), " to ", format(max(test_q)), "):\n", sep = "")
print(as.data.frame(scores %>% select(model, rmse_test)), row.names = FALSE)
cat("\nDFM nowcasts after the last GDP print:\n")
print(as.data.frame(live_q), row.names = FALSE)
cat("\nWritten to", normalizePath(OUT_DIR), "\n")
