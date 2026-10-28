# =============================================================================
# Activity 6 (R): Regularization with Ridge, LASSO and Elastic Net
# GDP Growth Nowcasting Workshop, IDB, Port of Spain
# Tuesday · Block 1 · about 20-30 minutes
#
# Packages: tidyverse, glmnet
#
# What you do:
#   Estimate the three regularized models from the theory block on the same quarterly
#   data as the Python notebook act6_regularization.ipynb, and compare them with OLS.
#   The flow repeats in Blocks 2 and 4:
#     1. split the sample into TRAIN (2013Q1-2023Q1) and TEST (2023Q2-2026Q1)
#     2. ESTIMATE on train
#     3. PREDICT the test quarters
#     4. EVALUATE with the test RMSE and save it to ../output/workshop_results.csv
#
#   Almost everything is already written: run it section by section. The sections marked
#   YOUR TURN hold one parameter that you change live to see what moves.
#   Each YOUR TURN line holds a starting value so the script runs end to end;
#   replace it with your own choice. The worked version is sols/act6_regularization.R.
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# install.packages(c("tidyverse", "glmnet"))   # once; already in the workshop environment

library(tidyverse)   # readr, dplyr, ggplot2 ...
library(glmnet)      # Ridge, LASSO, Elastic Net

# Work from the folder that holds this script, so that ../data/ resolves.
# In VS Code / RStudio: Session -> Set Working Directory -> To Source File Location.
getwd()

set.seed(42)

# --- Helpers (the same in act6, act7 and act12) --------------------------------

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

# =============================================================================
# 1. DATA AND TRAIN / TEST SPLIT
# =============================================================================
# blocks_quarterly.csv is already built (the same file the Python notebooks read):
#   date      first day of the quarter's last month (2023Q2 -> 2023-06-01)
#   RGDP0000  real GDP growth, quarter on quarter, %
#   sample    "train" (41 quarters, 2013Q1-2023Q1) or "test" (12 quarters, 2023Q2-2026Q1)
#   15 predictors: quarterly averages of monthly % growth, seasonally adjusted

q    <- read_csv("../data/blocks_quarterly.csv", show_col_types = FALSE)
meta <- read_csv("../data/blocks_meta.csv", show_col_types = FALSE)

predictors <- meta$ticker
labels     <- setNames(meta$label, meta$ticker)   # ticker -> readable name

train <- q %>% filter(sample == "train")
test  <- q %>% filter(sample == "test")

y_train <- train$RGDP0000
y_test  <- test$RGDP0000

# Standardize the predictors with the TRAIN mean and standard deviation (never the test
# ones: that would leak information from the future). After this every coefficient is
# "pp of GDP growth per one standard deviation of the predictor", so coefficients of
# different series can be compared with each other, and the penalty treats them equally.
mu <- colMeans(train[, predictors])
s  <- sapply(train[, predictors], sd)
X_train <- scale(as.matrix(train[, predictors]), center = mu, scale = s)
X_test  <- scale(as.matrix(test[, predictors]),  center = mu, scale = s)

# A first look: volatility and correlation with GDP growth in the training sample
tibble(ticker = predictors, label = labels[predictors], group = meta$group,
       sd_train = round(s, 2),
       corr_gdp = round(cor(train[, predictors], y_train)[, 1], 2)) %>%
  arrange(desc(abs(corr_gdp)))

# =============================================================================
# 2. STARTING POINT: OLS
# =============================================================================
# Before penalizing anything we need a number to beat.

df_train <- data.frame(RGDP0000 = y_train, X_train)
ols <- lm(RGDP0000 ~ ., data = df_train)
summary(ols)   # full diagnostics: the regularized models below do not report p-values

pred_ols <- predict(ols, newdata = data.frame(X_test))
rmse_ols <- rmse(y_test, pred_ols)
cat(sprintf("OLS test RMSE: %.4f\n", rmse_ols))

# Naive benchmark: predict the train-sample mean every quarter
cat(sprintf("Naive (train mean) test RMSE: %.4f\n", rmse(y_test, mean(y_train))))

save_result(1, "act6", "OLS", rmse_ols, "15 predictors, no penalty")

# =============================================================================
# 3. CROSS-VALIDATION FOLDS THAT RESPECT TIME
# =============================================================================
# cv.glmnet picks lambda by K-fold cross-validation. By default it assigns quarters to
# folds AT RANDOM. With time series that is a problem: GDP growth is autocorrelated, so a
# randomly held-out quarter has its neighbours (t-1, t+1) in the training folds and is
# "easy" to predict. The CV error comes out too optimistic and favours too little
# penalty. Contiguous blocks keep each held-out stretch of quarters together, which is
# closer to what nowcasting is: predicting quarters the model has not seen.

n_folds <- 5
foldid  <- cut(seq_along(y_train), breaks = n_folds, labels = FALSE)
tapply(format(train$date, "%Y-%m"), foldid, function(d) paste(range(d), collapse = " to "))

# Cross-validated RMSE for one (alpha, lambda) pair with these folds (used in the tables)
cv_rmse <- function(alpha, lambda) {
  err <- numeric(0)
  for (k in 1:n_folds) {
    fit <- glmnet(X_train[foldid != k, ], y_train[foldid != k], alpha = alpha,
                  standardize = FALSE)
    p   <- predict(fit, X_train[foldid == k, , drop = FALSE], s = lambda)
    err <- c(err, y_train[foldid == k] - p)
  }
  sqrt(mean(err^2))
}

# =============================================================================
# 4. RIDGE (alpha = 0)
# =============================================================================
# Ridge adds the L2 penalty  lambda * sum(beta^2): every coefficient shrinks toward zero,
# none becomes exactly zero.
#   glmnet(x, y, alpha = 0)   alpha = 0 is Ridge, alpha = 1 is LASSO, in between Elastic Net
#   lambda                    strength of the penalty; large = more shrinkage, 0 = OLS
# The data are already standardized, so standardize = FALSE.

cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0, foldid = foldid, standardize = FALSE)

# Chart 1 · The cross-validation curve. Left: little penalty, the model overfits.
# Right: so much penalty that it stops following the data. The dotted lines mark
# lambda.min (lowest CV error) and lambda.1se (simplest model within one s.e. of it).
plot(cv_ridge, sign.lambda = 1, main = "Ridge: cross-validation curve")   # log(lambda), more penalty to the right

# Look at the curve before trusting lambda.min. Here it has no U: the CV error keeps
# falling until the largest lambda, where every coefficient is practically zero and Ridge
# predicts the train mean. With 41 quarters (one of them COVID, -12%) these predictors carry
# little signal that survives out of sample, and cross-validation says so. Random folds
# give the same answer here; the contiguous folds matter more in longer, smoother samples.
pred_ridge <- predict(cv_ridge, X_test, s = "lambda.min")[, 1]
rmse_ridge <- rmse(y_test, pred_ridge)
cat(sprintf("Ridge: lambda.min = %.3f   test RMSE = %.4f   (OLS: %.4f)\n",
            cv_ridge$lambda.min, rmse_ridge, rmse_ols))

save_result(1, "act6", "Ridge", rmse_ridge,
            sprintf("lambda=%.3f (cv.glmnet, contiguous 5 folds)", cv_ridge$lambda.min))

# =============================================================================
# 5. LASSO (alpha = 1)
# =============================================================================
# LASSO uses the L1 penalty  lambda * sum(|beta|), which sets some coefficients EXACTLY to
# zero: it selects variables.

# --- Exploring the parameter ---------------------------------------------------
# Instead of editing lambda by hand and re-running, try several values at once.
# CV RMSE uses only the training sample; test RMSE is shown to learn from, but choosing
# on it would be cheating (12 test quarters are easy to overfit).

lambdas_explore <- c(0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)
lasso_path <- glmnet(X_train, y_train, alpha = 1, standardize = FALSE)

exploration_lasso <- tibble(
  lambda    = lambdas_explore,
  cv_rmse   = sapply(lambdas_explore, function(l) cv_rmse(1, l)),
  test_rmse = sapply(lambdas_explore, function(l)
                rmse(y_test, predict(lasso_path, X_test, s = l)[, 1])),
  n_vars    = sapply(lambdas_explore, function(l)
                sum(coef(lasso_path, s = l)[-1, 1] != 0))
) %>% mutate(across(c(cv_rmse, test_rmse), ~ round(.x, 4)))
exploration_lasso

# What cv.glmnet would pick on its own, with the same folds
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1, foldid = foldid, standardize = FALSE)
cat(sprintf("cv.glmnet: lambda.min = %.3f   lambda.1se = %.3f\n",
            cv_lasso$lambda.min, cv_lasso$lambda.1se))

# ---- YOUR TURN: fix the final LASSO ------------------------------------------
# From the table above, pick the lambda with the lowest CV RMSE (or the balance between
# error and number of variables you prefer) and write it here.
lambda_lasso <- 0.01   # <- starting value: replace with your choice
# ------------------------------------------------------------------------------

lasso <- glmnet(X_train, y_train, alpha = 1, lambda = lambda_lasso, standardize = FALSE)
pred_lasso <- predict(lasso, X_test)[, 1]
rmse_lasso <- rmse(y_test, pred_lasso)
beta_lasso <- coef(lasso)[-1, 1]

cat(sprintf("LASSO test RMSE: %.4f   (OLS: %.4f)\n", rmse_lasso, rmse_ols))
cat(sprintf("Variables that survive: %d of %d\n", sum(beta_lasso != 0), length(predictors)))
tibble(variable = labels[predictors], coefficient = round(beta_lasso, 4)) %>%
  filter(coefficient != 0) %>% arrange(desc(abs(coefficient)))

save_result(1, "act6", "LASSO", rmse_lasso, sprintf("lambda=%g", lambda_lasso))

# --- Chart 2 · The regularization path -----------------------------------------
# The central chart of the block. Each line is one coefficient; the horizontal axis is
# log(lambda), so the penalty GROWS to the right. In LASSO the lines hit zero and stay
# there: the variable left the model. In Ridge they approach zero but never touch it.
# Question for the group: which series is the last one to die in the LASSO panel?
# (Base-R alternative: plot(lasso_path, xvar = "lambda", label = TRUE).)

ridge_path <- glmnet(X_train, y_train, alpha = 0, standardize = FALSE)

path_long <- function(fit, name) {
  b <- as.matrix(coef(fit))[-1, ]
  as_tibble(t(b)) %>%
    mutate(log_lambda = log(fit$lambda)) %>%
    pivot_longer(-log_lambda, names_to = "ticker", values_to = "coef") %>%
    mutate(model = name, variable = labels[ticker])
}

bind_rows(path_long(lasso_path, "LASSO (L1)"), path_long(ridge_path, "Ridge (L2)")) %>%
  ggplot(aes(log_lambda, coef, color = variable)) +
  geom_hline(yintercept = 0, color = "black") +
  geom_line(linewidth = 0.8) +
  facet_wrap(~ model, scales = "free_x") +
  labs(title = "Regularization path: who touches zero and who only gets close",
       x = "log(lambda)  (more penalty to the right)", y = "Coefficient (per s.d.)",
       color = NULL) +
  theme_minimal()

# =============================================================================
# 6. ELASTIC NET (0 < alpha < 1)
# =============================================================================
# Elastic Net mixes both penalties:  lambda * [ alpha * |beta|  +  (1 - alpha)/2 * beta^2 ].
# The L1 part selects, the L2 part stabilizes the estimate when predictors are correlated
# (LASSO tends to keep one of a correlated group, somewhat arbitrarily).
#   alpha   the mix: 1 = pure LASSO, 0 = pure Ridge (glmnet's alpha is sklearn's l1_ratio)
#   lambda  the overall strength, chosen by cross-validation for each alpha

# --- Exploring the parameter ---------------------------------------------------
# As in the Python notebook, hold the overall strength fixed and move only the mix.
# (Letting cv.glmnet choose lambda for each alpha sends every mix to the null model here,
# for the same reason as LASSO above.)
lambda_enet    <- 0.5
alphas_explore <- c(0.1, 0.25, 0.5, 0.75, 0.9)

exploration_enet <- map_dfr(alphas_explore, function(a) {
  fit <- glmnet(X_train, y_train, alpha = a, lambda = lambda_enet, standardize = FALSE)
  tibble(alpha     = a,
         cv_rmse   = round(cv_rmse(a, lambda_enet), 4),
         test_rmse = round(rmse(y_test, predict(fit, X_test)[, 1]), 4),
         n_vars    = sum(coef(fit)[-1, 1] != 0))
})
exploration_enet

# ---- YOUR TURN: fix the final Elastic Net ------------------------------------
# Pick alpha (the mix) from the table above.
alpha_enet <- 0.1   # <- starting value: replace with your choice
# ------------------------------------------------------------------------------

enet      <- glmnet(X_train, y_train, alpha = alpha_enet, lambda = lambda_enet,
                    standardize = FALSE)
pred_enet <- predict(enet, X_test)[, 1]
rmse_enet <- rmse(y_test, pred_enet)
beta_enet <- coef(enet)[-1, 1]

cat(sprintf("Elastic Net test RMSE: %.4f   (OLS: %.4f)\n", rmse_enet, rmse_ols))
cat(sprintf("Variables that survive: %d of %d\n", sum(beta_enet != 0), length(predictors)))

save_result(1, "act6", "Elastic Net", rmse_enet,
            sprintf("alpha=%g, lambda=%g", alpha_enet, lambda_enet))

# =============================================================================
# 7. TEACHING CHARTS
# =============================================================================

# --- Chart 3 · The four models on the same variables ---------------------------
# Missing LASSO / Elastic Net bars are variables the penalty removed; the change in height
# against OLS is how much each model shrank the ones it kept. Because the predictors are
# standardized, comparing bars ACROSS variables is meaningful here too.

coef_comparison <- tibble(
  variable      = labels[predictors],
  OLS           = coef(ols)[-1],
  Ridge         = coef(cv_ridge, s = "lambda.min")[-1, 1],
  LASSO         = beta_lasso,
  `Elastic Net` = beta_enet
)
coef_comparison %>% mutate(across(-variable, ~ round(.x, 3)))

coef_comparison %>%
  pivot_longer(-variable, names_to = "model", values_to = "coef") %>%
  mutate(model = factor(model, levels = c("OLS", "Ridge", "LASSO", "Elastic Net")),
         variable = fct_reorder(variable, abs(coef), .fun = max)) %>%
  ggplot(aes(x = variable, y = coef, fill = model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  geom_hline(yintercept = 0) +
  coord_flip() +
  scale_fill_manual(values = c("#95A5A6", "#2980B9", "#C0392B", "#27AE60")) +
  labs(title = "Estimated coefficients, four models", x = NULL,
       y = "Coefficient (pp of GDP growth per s.d.)", fill = NULL) +
  theme_minimal()

# --- Chart 4 · Observed vs predicted on the test quarters ----------------------
tibble(date = test$date, Observed = y_test, OLS = pred_ols, Ridge = pred_ridge,
       LASSO = pred_lasso, `Elastic Net` = pred_enet) %>%
  pivot_longer(-date, names_to = "series", values_to = "growth") %>%
  ggplot(aes(date, growth, color = series, linewidth = series == "Observed")) +
  geom_hline(yintercept = 0, color = "grey50") +
  geom_line() + geom_point() +
  scale_color_manual(values = c(Observed = "black", OLS = "#95A5A6", Ridge = "#2980B9",
                                LASSO = "#C0392B", `Elastic Net` = "#27AE60")) +
  scale_linewidth_manual(values = c(0.6, 1.3), guide = "none") +
  labs(title = "Regularized models on the test quarters", x = NULL,
       y = "Real GDP growth, q/q %", color = NULL) +
  theme_minimal()

# =============================================================================
# 8. RESULTS TABLE (shared with the Python notebooks)
# =============================================================================

show_results()
