# ============================================================
# GDP Nowcasting — Activity 11: Classical mixed-frequency models
# DFM · MIDAS · MF-BVAR — Trinidad and Tobago
# IDB Caribbean Country Department
# ============================================================
# Run this script section by section, following along with
# the HTML notebook. Each section header matches a section
# in the notebook; sections depend on the ones above them.
#
# To run one section: select its lines and press Ctrl+Enter.
# Work from the folder that holds this script (activity/),
# so that ../data/ and ../output/ resolve.
# ============================================================


# ---- 0. Packages (run once if missing) -------------------------------------

# install.packages(c("tidyverse", "nowcastDFM", "midasr", "imputeTS"))
# install.packages("mfbvar")   # or devtools::install_github("ankargren/mfbvar"), needs Rtools


# ---- 1. Setup --------------------------------------------------------------

library(tidyverse)
library(midasr)
# nowcastDFM, mfbvar and imputeTS are called with pkg::function(): nowcastDFM attaches
# packages that mask base functions such as sum() and dplyr::filter()

getwd()   # should end in .../activity
set.seed(42)

# Shared helpers (same as act6, act7 and act8)
rmse <- function(observed, predicted) sqrt(mean((observed - predicted)^2))

results_file <- "../output/workshop_results.csv"

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

# Chart style, one color per model
model_colors <- c("Actual" = "black", "DFM" = "#C8102E", "MIDAS" = "#0D3B7A",
                  "MF-BVAR" = "#00A3A6")
model_lines  <- c("Actual" = "solid", "DFM" = "dashed", "MIDAS" = "dashed",
                  "MF-BVAR" = "dashed")


# ---- 2. Data ---------------------------------------------------------------

m    <- read_csv("../data/blocks_monthly.csv", show_col_types = FALSE)    # monthly % growth
q    <- read_csv("../data/blocks_quarterly.csv", show_col_types = FALSE)  # train/test split
meta <- read_csv("../data/blocks_meta.csv", show_col_types = FALSE)

predictors <- meta$ticker
labels     <- setNames(meta$label, meta$ticker)

train_end <- max(q$date[q$sample == "train"])         # 2023-03-01
test_q    <- q$date[q$sample == "test"]               # 12 quarter-end months
oos_q     <- as.Date(c("2026-06-01", "2026-09-01"))   # GDP not published yet
y_train   <- q$RGDP0000[q$sample == "train"]
y_test    <- q$RGDP0000[q$sample == "test"]

# The ragged edge: each indicator stops in a different month
m %>%
  pivot_longer(-date, names_to = "ticker") %>%
  filter(!is.na(value)) %>%
  group_by(ticker) %>%
  summarise(last_month = max(date)) %>%
  mutate(label = labels[ticker]) %>%
  arrange(last_month)

# Filled copy for MIDAS and MF-BVAR: interior gaps interpolated, end of sample = last value
mx <- m %>% filter(date <= max(oos_q)) %>%
  mutate(across(all_of(predictors), ~ imputeTS::na_interpolation(.x)))


# ---- 3. Dynamic factor model -----------------------------------------------

panel <- m %>% select(date, RGDP0000, all_of(predictors)) %>% as.data.frame()
panel_train <- panel %>% filter(date <= train_end)

# One global factor, p = 1 lag; PCA start, then EM iterations (about 5-20 s)
dfm_fit <- quietly(nowcastDFM::dfm(panel_train, p = 1, max_iter = 1500, threshold = 1e-5))
cat(sprintf("EM iterations: %d, converged = %s\n", dfm_fit$num_iter, dfm_fit$convergence == 1))

# The common factor against GDP growth
tibble(date = panel_train$date, factor = dfm_fit$Z[, 1],
       gdp = (panel_train$RGDP0000 - mean(y_train)) / sd(y_train)) %>%
  ggplot(aes(date)) +
  geom_line(aes(y = factor / sd(factor), color = "Common factor (monthly)"), linewidth = 1.1) +
  geom_point(aes(y = gdp, color = "Real GDP growth (quarterly)"), size = 2.5, na.rm = TRUE) +
  scale_color_manual(values = c("#C8102E", "black")) +
  labs(title = "DFM: the common factor behind the 15 indicators",
       subtitle = "Train sample, both series standardized",
       x = NULL, y = "Standardized", color = NULL) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

# Nowcast each test quarter: parameters fixed on train, that quarter's GDP hidden
dfm_nowcast <- function(d) {
  vintage <- panel %>% filter(date <= d)
  vintage$RGDP0000[vintage$date == d] <- NA
  filled  <- quietly(nowcastDFM::predict_dfm(vintage, dfm_fit))
  filled$RGDP0000[filled$date == d]
}
pred_dfm <- sapply(test_q, dfm_nowcast)
rmse_dfm <- rmse(y_test, pred_dfm)
cat(sprintf("DFM test RMSE: %.4f\n", rmse_dfm))

save_result(4, "act11", "DFM", rmse_dfm,
            "nowcastDFM, 1 global factor, p=1, 15 indicators, fixed on train")


# ---- 4. MIDAS --------------------------------------------------------------

n_train <- length(y_train)
first_test_month <- 3 * n_train + 1   # monthly row where the test sample starts

midas_one <- function(ticker, n_lags) {
  x <- mx[[ticker]]
  dat <- list(y = y_train, x = x[1:(3 * n_train)])
  # mls(x, lags, 3, nealmon): monthly lags of x, 3 months per quarter, Almon weights
  fit <- midas_r(y ~ mls(x, 0:(n_lags - 1), 3, nealmon), data = dat,
                 start = list(x = c(1, -0.5)))
  pred <- as.numeric(forecast(fit, newdata = list(x = x[first_test_month:length(x)]))$mean)
  list(fit = fit,
       pred = pred[seq_along(test_q)],                        # 12 test quarters
       oos  = pred[length(test_q) + seq_along(oos_q)])        # 2026Q2 and 2026Q3
}

# Example: LNG production, 6 monthly lags (current and previous quarter)
lng <- midas_one("RENE5005", 6)
summary(lng$fit)
cat(sprintf("MIDAS (LNG, 6 lags) test RMSE: %.4f\n", rmse(y_test, lng$pred)))

# Weight on each monthly lag
theta_hat <- coef(lng$fit)[c("x1", "x2")]
tibble(lag = 0:5, coefficient = nealmon(theta_hat, 6)) %>%
  ggplot(aes(lag, coefficient)) +
  geom_col(fill = "#0D3B7A") +
  scale_x_continuous(breaks = 0:5, labels = c("0\n(last month\nof quarter)", 1:5)) +
  labs(title = "MIDAS: weight on each monthly lag",
       subtitle = "LNG production, exponential Almon polynomial",
       x = "Months back from the quarter's last month", y = "Coefficient") +
  theme_minimal(base_size = 13)


# ---- 4.1 YOUR TURN: fix the final MIDAS ------------------------------------
# Choose the indicator(s) and the number of monthly lags. With several indicators,
# one MIDAS is fitted per indicator and the nowcasts are averaged.
# Candidates: "RENE5005" (LNG production), "UNTL0003" (night lights, oil and gas),
#             "MCRD5000" (private sector credit)

midas_vars <- c("UNTL0003")   # <- one or more tickers
n_lags     <- 6               # <- number of monthly lags (3 to 6)

midas_runs <- lapply(midas_vars, function(v) midas_one(v, n_lags))
pred_midas <- rowMeans(sapply(midas_runs, `[[`, "pred"))
oos_midas  <- rowMeans(sapply(midas_runs, `[[`, "oos"))
rmse_midas <- rmse(y_test, pred_midas)
cat(sprintf("MIDAS test RMSE: %.4f\n", rmse_midas))

save_result(4, "act11", "MIDAS", rmse_midas,
            sprintf("midasr nealmon, %d monthly lags, avg of %s, fixed on train",
                    n_lags, paste(midas_vars, collapse = "+")))


# ---- 5. MF-BVAR ------------------------------------------------------------
# Re-estimated for each quarter on all data up to its last month (about 15 s)

bvar_vars <- c("RENE5005", "UNTL0003", "MCRD5000")   # K = 4 with GDP: keep it small

bvar_nowcast <- function(d) {
  v <- mx %>% filter(date <= d)
  Y <- lapply(bvar_vars, function(s) ts(v[[s]], start = c(2013, 1), frequency = 12))
  names(Y) <- bvar_vars
  Y$GDP <- ts(q$RGDP0000[q$date < d], start = c(2013, 1), frequency = 4)  # GDP known before d
  # lambda1 = overall tightness of the Minnesota prior
  prior <- mfbvar::set_prior(Y = Y, n_lags = 4, n_reps = 1000, n_burnin = 500,
                             n_fcst = 1, lambda1 = 0.2)
  fit   <- mfbvar::estimate_mfbvar(prior, prior = "minn", variance = "iw")
  draws <- predict(fit, pred_bands = NULL)
  mean(draws$fcst[draws$variable == "GDP" & draws$fcst_date == d])
}

set.seed(42)
pred_bvar <- sapply(test_q, bvar_nowcast)
rmse_bvar <- rmse(y_test, pred_bvar)
cat(sprintf("MF-BVAR test RMSE: %.4f\n", rmse_bvar))

save_result(4, "act11", "MF-BVAR", rmse_bvar,
            "mfbvar Minnesota lambda1=0.2, 4 lags, LNG+NTL O&G+credit+GDP, re-estimated each quarter")


# ---- 6. Actuals vs nowcasts on the test quarters ---------------------------

tibble(date = test_q, Actual = y_test, DFM = pred_dfm, MIDAS = pred_midas,
       `MF-BVAR` = pred_bvar) %>%
  pivot_longer(-date, names_to = "series", values_to = "value") %>%
  mutate(series = factor(series, levels = names(model_colors))) %>%
  ggplot(aes(x = date, y = value, color = series, linetype = series)) +
  geom_hline(yintercept = 0, color = "grey60") +
  geom_line(linewidth = 1.1) +
  geom_point(data = ~ dplyr::filter(., series == "Actual"), size = 2.5) +
  scale_color_manual(values = model_colors) +
  scale_linetype_manual(values = model_lines) +
  labs(title = "Actuals vs nowcasts on the test quarters",
       subtitle = "Trinidad and Tobago real GDP growth, 2023Q2-2026Q1",
       x = NULL, y = "% growth, q/q", color = NULL, linetype = NULL) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")


# ---- 7. Out-of-sample nowcast: 2026Q2 and 2026Q3 ---------------------------
# GDP is published to 2026Q1; the indicators run to October 2026.

live_dfm <- quietly(nowcastDFM::predict_dfm(panel, dfm_fit))   # Kalman fills the ragged edge
oos_dfm  <- live_dfm$RGDP0000[match(oos_q, live_dfm$date)]

set.seed(42)
oos_bvar <- sapply(oos_q, bvar_nowcast)                        # re-estimated, GDP to 2026Q1

oos <- tibble(date = oos_q, DFM = oos_dfm, MIDAS = oos_midas, `MF-BVAR` = oos_bvar)
oos %>%
  mutate(quarter = paste0(format(date, "%Y"), "Q", (as.integer(format(date, "%m")) + 2) %/% 3),
         across(DFM:`MF-BVAR`, ~ round(.x, 2))) %>%
  select(quarter, DFM, MIDAS, `MF-BVAR`)

history <- tibble(date = q$date[q$date >= as.Date("2022-03-01")],
                  Actual = q$RGDP0000[q$date >= as.Date("2022-03-01")])

bind_rows(
  history %>% pivot_longer(-date, names_to = "series", values_to = "value"),
  tibble(date = test_q, DFM = pred_dfm, MIDAS = pred_midas, `MF-BVAR` = pred_bvar) %>%
    bind_rows(oos) %>%
    pivot_longer(-date, names_to = "series", values_to = "value")
) %>%
  mutate(series = factor(series, levels = names(model_colors))) %>%
  ggplot(aes(x = date, y = value, color = series, linetype = series)) +
  annotate("rect", xmin = max(test_q) + 45, xmax = max(oos_q) + 45,
           ymin = -Inf, ymax = Inf, fill = "grey90") +
  annotate("text", x = mean(oos_q), y = Inf, label = "Out of sample",
           vjust = 1.5, size = 3.5, color = "grey40") +
  geom_hline(yintercept = 0, color = "grey60") +
  geom_line(linewidth = 1.1) +
  geom_point(data = ~ dplyr::filter(., series == "Actual" | date %in% oos_q), size = 2.5) +
  scale_color_manual(values = model_colors) +
  scale_linetype_manual(values = model_lines) +
  labs(title = "Nowcasts for the quarters not yet published",
       subtitle = "Test quarters 2023Q2-2026Q1, out-of-sample 2026Q2-2026Q3",
       x = NULL, y = "% growth, q/q", color = NULL, linetype = NULL) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")


# ---- 8. Results table ------------------------------------------------------
# Every model of the workshop, on the same 12 test quarters

show_results()
