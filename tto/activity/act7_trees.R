# =============================================================================
# Activity 7 (R): Tree-based models (a single tree, Random Forest, Gradient Boosting)
# GDP Growth Nowcasting Workshop, IDB, Port of Spain
# Tuesday · Block 2 · about 20-30 minutes
#
# Packages: tidyverse, rpart, rpart.plot, randomForest, xgboost, glmnet
#
# What you do:
#   Estimate three tree-based models on exactly the same data and train/test split as
#   Activity 6, so the test RMSEs are directly comparable with OLS, Ridge, LASSO and
#   Elastic Net (and with the Python notebook act7_trees.ipynb). Run section by section;
#   the sections marked YOUR TURN hold the parameters you change live.
#   Each YOUR TURN line holds a starting value so the script runs end to end;
#   replace it with your own choice. The worked version is sols/act7_trees.R.
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# install.packages(c("tidyverse", "rpart", "rpart.plot", "randomForest", "xgboost", "glmnet"))

library(tidyverse)      # readr, dplyr, ggplot2 ...
library(rpart)          # a single regression tree
library(rpart.plot)     # drawing it
library(randomForest)   # Random Forest
library(xgboost)        # Gradient Boosting
library(glmnet)         # LASSO, only for the comparison at the end

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

# =============================================================================
# 1. DATA AND TRAIN / TEST SPLIT (identical to Activity 6)
# =============================================================================

q    <- read_csv("../data/blocks_quarterly.csv", show_col_types = FALSE)
meta <- read_csv("../data/blocks_meta.csv", show_col_types = FALSE)

predictors <- meta$ticker
labels     <- setNames(meta$label, meta$ticker)   # ticker -> readable name

train <- q %>% filter(sample == "train")
test  <- q %>% filter(sample == "test")

y_train <- train$RGDP0000
y_test  <- test$RGDP0000

# Trees split on thresholds, so they do not need standardized predictors.
X_train <- as.matrix(train[, predictors])
X_test  <- as.matrix(test[, predictors])
df_train <- data.frame(RGDP0000 = y_train, X_train)
df_test  <- data.frame(X_test)

# Reference: what was saved in Activity 6 (if you ran it)
show_results()

# =============================================================================
# 2. A SINGLE TREE (rpart)
# =============================================================================
# A regression tree splits the sample in two again and again, each time choosing the
# variable and threshold that most reduce the squared error inside the two groups. It
# predicts the average GDP growth of the training quarters that end up in each leaf.
#
#   maxdepth   how many splits can be chained. Deeper = more flexible = memorizes noise.
#   minsplit   minimum quarters in a node to try a split. Higher = shorter trees.
#   minbucket  minimum quarters in a leaf. Higher = less overfitted predictions.
#   cp         complexity parameter: a split must improve the fit by this share to be kept.
#              Set to 0 here so that only maxdepth limits the tree.
# With 41 training quarters, pruning is not a technical detail: it is what makes the
# model usable.

fit_tree <- function(depth, minbucket = 2) {
  rpart(RGDP0000 ~ ., data = df_train, method = "anova",
        control = rpart.control(maxdepth = depth, minsplit = 6, minbucket = minbucket,
                                cp = 0, xval = 0))
}

# --- Chart 1 · Where overfitting starts -----------------------------------------
# Training error always falls with depth: a deep enough tree memorizes every quarter.
# Test error falls at first and then turns up. The turning point is where the model
# stops learning and starts memorizing.

depths <- 1:8
depth_table <- tibble(
  depth      = depths,
  rmse_train = sapply(depths, function(d) rmse(y_train, predict(fit_tree(d), df_train))),
  rmse_test  = sapply(depths, function(d) rmse(y_test,  predict(fit_tree(d), df_test)))
)
depth_table %>% mutate(across(-depth, ~ round(.x, 4)))

depth_table %>%
  pivot_longer(-depth, names_to = "sample", values_to = "rmse") %>%
  mutate(sample = recode(sample, rmse_train = "Train", rmse_test = "Test")) %>%
  ggplot(aes(depth, rmse, color = sample)) +
  geom_line(linewidth = 1) + geom_point(size = 2) +
  scale_color_manual(values = c(Train = "#2980B9", Test = "#C0392B")) +
  scale_x_continuous(breaks = depths) +
  labs(title = "The tree learns up to a point, and then memorizes",
       x = "Maximum depth of the tree", y = "RMSE", color = NULL) +
  theme_minimal()

# ---- YOUR TURN: fix the final tree -------------------------------------------
# From the chart above, pick the depth where the test error is lowest (just before it
# starts to rise).
max_depth <- 6   # <- starting value: replace with your choice
# ------------------------------------------------------------------------------

tree <- fit_tree(max_depth)
pred_tree  <- predict(tree, df_test)
rmse_tree  <- rmse(y_test, pred_tree)
rmse_tree_train <- rmse(y_train, predict(tree, df_train))

cat(sprintf("Train RMSE : %.4f\n", rmse_tree_train))
cat(sprintf("Test RMSE  : %.4f\n", rmse_tree))
cat(sprintf("Gap        : %.4f  <- if it grows, the tree is overfitting\n",
            rmse_tree - rmse_tree_train))
cat(sprintf("Leaves     : %d\n", sum(tree$frame$var == "<leaf>")))

save_result(2, "act7", "Decision Tree", rmse_tree, sprintf("maxdepth=%d", max_depth))

# --- Chart 2 · The tree from the inside -----------------------------------------
# The teaching advantage of trees: you can draw them and read them. Each box shows the
# predicted GDP growth and the share of training quarters that reached it; each split
# shows the variable and threshold. Which variable is at the root? Compare it with the
# last series to survive LASSO in Activity 6.
rpart.plot(tree, type = 4, extra = 101, roundint = FALSE, digits = 2,
           split.fun = function(x, labs, digits, varlen, faclen) {
             for (tk in predictors) labs <- sub(paste0("^", tk), labels[[tk]], labs)
             labs
           },
           main = "Decision tree for T&T real GDP growth")

# =============================================================================
# 3. RANDOM FOREST (randomForest)
# =============================================================================
# Many trees, each on a bootstrap sample of the quarters and, at every split, choosing
# among a random subset of variables only. The double randomization makes the trees
# wrong in different ways, and averaging them cancels part of the error.
#
#   ntree      number of trees. More is better up to a point; past ~300 the gain is small.
#   mtry       variables drawn as candidates at each split: the main source of diversity.
#              (Regression default: p/3 = 5.)
#   nodesize   minimum quarters in a leaf of each tree.
#
# The forest also validates itself without touching the test sample: each tree leaves out
# about a third of the quarters (its "out-of-bag" quarters), and those are predicted as if
# they were new data. That is the OOB error printed below.

# ---- YOUR TURN: estimate the forest ------------------------------------------
ntree <- 50    # <- starting value: replace with your choice
mtry  <- 15    # <- starting value (15 = every variable = bagging)
# ------------------------------------------------------------------------------

set.seed(42)
rf <- randomForest(x = X_train, y = y_train, ntree = ntree, mtry = mtry, nodesize = 3,
                   importance = TRUE)
pred_rf <- predict(rf, X_test)
rmse_rf <- rmse(y_test, pred_rf)

cat(sprintf("Random Forest test RMSE: %.4f   (single tree: %.4f)\n", rmse_rf, rmse_tree))
cat(sprintf("OOB RMSE: %.4f   OOB R2: %.3f   <- internal validation, no test data used\n",
            sqrt(tail(rf$mse, 1)), tail(rf$rsq, 1)))

save_result(2, "act7", "Random Forest", rmse_rf, sprintf("ntree=%d, mtry=%d", ntree, mtry))

# --- Chart 3 · The forest from the inside ----------------------------------------
# predict.all = TRUE returns every tree's prediction. Each grey line is one tree; the
# green line is their average, which is what the forest reports. Each tree alone is
# erratic, but they err in different directions.
all_trees  <- predict(rf, X_test, predict.all = TRUE)$individual   # 12 quarters x ntree
rmse_trees <- apply(all_trees, 2, function(p) rmse(y_test, p))

as_tibble(all_trees, .name_repair = ~ paste0("t", seq_along(.x))) %>%
  mutate(date = test$date) %>%
  pivot_longer(-date, names_to = "tree", values_to = "pred") %>%
  ggplot(aes(date, pred)) +
  geom_line(aes(group = tree), color = "grey60", alpha = 0.15) +
  geom_hline(yintercept = 0, color = "grey40") +
  geom_line(data = tibble(date = test$date, y = y_test), aes(y = y), linewidth = 1.2) +
  geom_point(data = tibble(date = test$date, y = y_test), aes(y = y), size = 2) +
  geom_line(data = tibble(date = test$date, y = pred_rf), aes(y = y),
            color = "#27AE60", linewidth = 1.2) +
  labs(title = "Each grey line is one tree; the forest (green) is their average",
       subtitle = "Black: observed GDP growth", x = NULL, y = "Real GDP growth, q/q %") +
  theme_minimal()

hist(rmse_trees, breaks = 30, col = "lightgrey", border = "grey",
     main = "Test RMSE of each tree in the forest", xlab = "Test RMSE")
abline(v = rmse_rf, col = "#27AE60", lwd = 3)
abline(v = rmse_tree, col = "#8E44AD", lwd = 2, lty = 2)
legend("topright", bty = "n", lwd = c(3, 2), lty = c(1, 2), col = c("#27AE60", "#8E44AD"),
       legend = c(sprintf("Forest (%.2f)", rmse_rf), sprintf("Single tree (%.2f)", rmse_tree)))
cat(sprintf("The forest beats %.0f%% of its own trees\n", 100 * mean(rmse_trees > rmse_rf)))

# =============================================================================
# 4. GRADIENT BOOSTING (xgboost)
# =============================================================================
# Here the trees are not averaged but chained: each new (small) tree is fitted to the
# errors left by the sum of the previous ones, so the model corrects itself round by round.
#
#   nrounds     number of sequential rounds (trees).
#   eta         learning rate: how much each tree is allowed to contribute. Low values
#               need more rounds but generalize better.
#   max_depth   depth of each tree; boosting works with shallow trees (2-3).
#   subsample   share of quarters drawn for each tree (stochastic boosting).
# eta and nrounds move together: lowering eta without raising nrounds leaves the model
# half estimated.

dtrain <- xgb.DMatrix(X_train, label = y_train)
dtest  <- xgb.DMatrix(X_test,  label = y_test)

xgb_params <- function(eta) list(objective = "reg:squarederror", eta = eta, max_depth = 2,
                                 min_child_weight = 3, subsample = 0.8, nthread = 1)

fit_xgb <- function(eta, nrounds) {
  set.seed(42)
  xgb.train(params = xgb_params(eta), data = dtrain, nrounds = nrounds, verbose = 0)
}

# Contiguous folds, as in Activity 6, for a CV error that uses the training sample only
n_folds <- 5
folds <- split(seq_along(y_train), cut(seq_along(y_train), n_folds, labels = FALSE))
cv_xgb <- function(eta, nrounds) {
  set.seed(42)
  cv <- xgb.cv(params = xgb_params(eta), data = dtrain, nrounds = nrounds, folds = folds,
               verbose = 0)
  tail(cv$evaluation_log$test_rmse_mean, 1)
}

# --- Exploring the pair of parameters --------------------------------------------
combos <- expand_grid(eta = c(0.3, 0.1, 0.03), nrounds = c(25, 100, 300))
exploration_xgb <- combos %>%
  mutate(cv_rmse   = map2_dbl(eta, nrounds, cv_xgb),
         test_rmse = map2_dbl(eta, nrounds, ~ rmse(y_test, predict(fit_xgb(.x, .y), dtest)))) %>%
  mutate(across(c(cv_rmse, test_rmse), ~ round(.x, 4)))
exploration_xgb

# ---- YOUR TURN: fix the final boosting model ---------------------------------
# Copy the (eta, nrounds) pair you prefer from the table above.
eta     <- 0.3   # <- starting value: replace with your choice
nrounds <- 300   # <- starting value: replace with your choice
# ------------------------------------------------------------------------------

gbt <- fit_xgb(eta, nrounds)
pred_gbt <- predict(gbt, dtest)
rmse_gbt <- rmse(y_test, pred_gbt)
cat(sprintf("Gradient Boosting test RMSE: %.4f\n", rmse_gbt))

save_result(2, "act7", "Gradient Boosting", rmse_gbt,
            sprintf("xgboost eta=%g, nrounds=%d, max_depth=2", eta, nrounds))

# --- Chart 4 · Train vs test, round by round --------------------------------------
# Training error always falls: adding trees never worsens the fit on data already seen.
# The question is in which round the model stops improving on unseen data. The watchlist
# records both errors after every round. Run on purpose with far more rounds than needed.
# Caveat: the test curve uses the same test sample we report, so picking the round by
# looking at it is cheating in the strict sense. It is here to show the phenomenon.

n_long <- 500
set.seed(42)
gbt_long <- xgb.train(params = xgb_params(0.1), data = dtrain, nrounds = n_long,
                      watchlist = list(train = dtrain, test = dtest), verbose = 0)
log_long <- gbt_long$evaluation_log
best_round <- which.min(log_long$test_rmse)
cat(sprintf("Round with the lowest test RMSE: %d of %d (RMSE %.4f; final round %.4f)\n",
            best_round, n_long, min(log_long$test_rmse), tail(log_long$test_rmse, 1)))

log_long %>%
  pivot_longer(-iter, names_to = "sample", values_to = "rmse") %>%
  mutate(sample = recode(sample, train_rmse = "Train", test_rmse = "Test")) %>%
  ggplot(aes(iter, rmse, color = sample)) +
  geom_line(linewidth = 0.9) +
  geom_vline(xintercept = best_round, linetype = "dashed", color = "#27AE60") +
  annotate("text", x = best_round, y = max(log_long$test_rmse), hjust = -0.1,
           label = paste("early stopping\n(round", best_round, ")"), color = "#27AE60") +
  scale_color_manual(values = c(Train = "#2980B9", Test = "#E67E22")) +
  labs(title = "Gradient Boosting (eta = 0.1): train vs test RMSE by round",
       x = "Boosting round", y = "RMSE", color = NULL) +
  theme_minimal()
# In production, xgb.train(..., early_stopping_rounds = 20) with a validation set carved
# out of the TRAINING sample stops the boosting automatically.

# --- Chart 5 · The three predictions on the test quarters ------------------------
# Note how flat the single tree is: a shallow tree has only a few possible predictions
# (one per leaf), and if every test quarter lands in the same leaf it predicts a constant.
tibble(date = test$date, Observed = y_test, `Decision Tree` = pred_tree,
       `Random Forest` = pred_rf, `Gradient Boosting` = pred_gbt) %>%
  pivot_longer(-date, names_to = "series", values_to = "growth") %>%
  ggplot(aes(date, growth, color = series, linewidth = series == "Observed")) +
  geom_hline(yintercept = 0, color = "grey50") +
  geom_line() + geom_point() +
  scale_color_manual(values = c(Observed = "black", `Decision Tree` = "#8E44AD",
                                `Random Forest` = "#27AE60", `Gradient Boosting` = "#E67E22")) +
  scale_linewidth_manual(values = c(0.6, 1.3), guide = "none") +
  labs(title = "Tree-based models on the test quarters", x = NULL,
       y = "Real GDP growth, q/q %", color = NULL) +
  theme_minimal()

# =============================================================================
# 5. VARIABLE IMPORTANCE VS LASSO COEFFICIENTS
# =============================================================================
# Trees report how much each variable reduced the error over all the splits it took part
# in (Random Forest: IncNodePurity; xgboost: Gain). LASSO answers the same question the
# opposite way, by deleting coefficients of a linear model. When both agree on the same
# two or three series, the result is solid. When they differ, either the variable enters
# GDP non-linearly (the linear model missed it) or LASSO dropped it although it matters.
# All three are rescaled to shares that sum to 1.

# Re-estimate the Activity 6 LASSO (standardized predictors, lambda = 0.5)
mu <- colMeans(X_train); s <- apply(X_train, 2, sd)
lasso <- glmnet(scale(X_train, mu, s), y_train, alpha = 1, lambda = 0.5, standardize = FALSE)
lasso_abs <- abs(coef(lasso)[-1, 1])

xgb_gain <- setNames(rep(0, length(predictors)), predictors)
imp_x <- xgb.importance(feature_names = predictors, model = gbt)
xgb_gain[imp_x$Feature] <- imp_x$Gain

rf_imp <- importance(rf, type = 2)[predictors, 1]   # IncNodePurity

share <- function(v) if (sum(v) > 0) v / sum(v) else v
importances <- tibble(
  variable            = labels[predictors],
  `Random Forest`     = share(rf_imp),
  `Gradient Boosting` = share(xgb_gain),
  `LASSO (|coef|)`    = share(lasso_abs)
)
importances %>% mutate(across(-variable, ~ round(.x, 3))) %>% arrange(desc(`Random Forest`))

importances %>%
  mutate(variable = fct_reorder(variable, `Random Forest`)) %>%
  pivot_longer(-variable, names_to = "model", values_to = "share") %>%
  ggplot(aes(variable, share, fill = model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  coord_flip() +
  scale_fill_manual(values = c(`Random Forest` = "#27AE60", `Gradient Boosting` = "#E67E22",
                               `LASSO (|coef|)` = "#C0392B")) +
  labs(title = "Three models, one question: which series inform GDP?",
       x = NULL, y = "Relative importance (share)", fill = NULL) +
  theme_minimal()

# =============================================================================
# 6. RESULTS TABLE (shared with the Python notebooks)
# =============================================================================

show_results()
