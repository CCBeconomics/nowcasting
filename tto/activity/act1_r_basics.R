# =============================================================================
# Activity 1 (R): R basics with GDP, oil and night-time lights in Trinidad and Tobago
# GDP Growth Nowcasting Workshop, IDB, Port of Spain
#
# The same data as act1_python_basics.ipynb:
#   ../data/gdp.csv        real GDP, TT$ millions, quarterly (date = 1st day of the quarter's last month)
#   ../data/eia_brent.csv  Brent crude, US$ per barrel, daily
#   ../data/ntl_shape.csv  night-time lights over the whole country, daily
#
# Lines with ___ are for you to complete. The solution is act1_r_basics_solutions.R.
# =============================================================================

# =============================================================================
# SETUP
# =============================================================================

# Install the packages once (they are already in the workshop environment):
# install.packages(c("tidyverse", "lubridate"))

library(tidyverse)   # dplyr, tidyr, readr, ggplot2 ...
library(lubridate)   # dates

# Work from the folder that holds this script, so that ../data/ resolves.
# In VS Code / RStudio: Session -> Set Working Directory -> To Source File Location,
# or uncomment and edit:
# setwd("C:/path/to/public/activity")
getwd()

# =============================================================================
# BASIC R OPERATIONS
# =============================================================================

# Arithmetic and logical operators
2 + 5
2 * 5
2^5
2 + 5 > 2 * 5
2^5 != 2 * 5

# Assignment
country <- "Trinidad and Tobago"
country

# Data types
gdp_2026q1 <- 34645         # numeric: real GDP in 2026Q1, TT$ millions
is_island <- TRUE           # logical
class(country); class(gdp_2026q1); class(is_island)

# Vectors: one type only
x <- c(10, 20, 30)
x * 2

# Lists: can hold different types
x <- list(10, "Brent", 10 < 20)
str(x)

# Matrices: two-dimensional, one type only
x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2)
x

# A data frame: columns of equal length, possibly different types
df <- data.frame(
  country = c("Trinidad and Tobago", "Barbados", "Jamaica", "Guyana"),
  energy  = c(TRUE, FALSE, FALSE, TRUE)
)
df
str(df)

# Selecting
df$country           # a column, with $
df["country"]        # a column, with []
df[1, ]              # first row
df[1, "country"]     # one cell

# Conditionals
brent <- 65
if (brent >= 80) {
  print("High oil price")
} else if (brent >= 60) {
  print("Medium oil price")
} else {
  print("Low oil price")
}

# Loops: a for loop over the rows of df, with an if inside
for (i in 1:nrow(df)) {
  if (df$energy[i]) {
    print(paste(df$country[i], "is an energy exporter"))
  } else {
    print(paste(df$country[i], "is not"))
  }
}

# A while loop runs as long as the condition is TRUE
i <- 1
while (i < 6) {
  print(i)
  i <- i + 1
}

# =============================================================================
# THE PIPE AND THE TIDYVERSE VERBS
# =============================================================================
# %>% feeds the output of one function into the next.
#   filter() rows, select() columns, rename() a column (new = old), mutate() new columns,
#   arrange() sort, summarise() collapse rows (by group with group_by())

df %>%
  filter(energy) %>%
  mutate(exporter = "energy") %>%
  arrange(country) %>%
  select(country, exporter)

# =============================================================================
# 1. IMPORT THE DATA
# =============================================================================

gdp <- read_csv("../data/gdp.csv")      # readr: dates in ISO format are parsed automatically
gdp <- gdp %>% rename(gdp = rgdp)
str(gdp)
tail(gdp)

### Activity 1: import the Brent price, and rename its column "brent price" to brent.
### (Names with spaces need back-ticks in R: `brent price`.)
oil <- ___
oil <- oil %>% rename(brent = ___)
tail(oil)

ntl <- read_csv("../data/ntl_shape.csv") %>% select(date, shape_mean)
head(ntl)

# =============================================================================
# 2. ONE FREQUENCY FOR ALL THREE
# =============================================================================
# GDP is quarterly and stamped on the first day of the quarter's LAST month
# (2025Q4 -> 2025-12-01). Daily series must be averaged into quarters and stamped
# the same way before they can be joined.
#   floor_date(date, "quarter")       -> first day of the quarter (2025-10-01)
#   %m+% months(2)                    -> two months later        (2025-12-01)

ntl_q <- ntl %>%
  mutate(date = floor_date(date, "quarter") %m+% months(2)) %>%
  group_by(date) %>%
  summarise(ntl = mean(shape_mean, na.rm = TRUE), days = n())
tail(ntl_q)

### Activity 2: do the same for the oil price.
oil_q <- oil %>%
  mutate(date = ___) %>%
  group_by(date) %>%
  summarise(brent = ___)
tail(oil_q)

# =============================================================================
# 3. JOIN
# =============================================================================

tt <- gdp %>%
  left_join(ntl_q, by = "date") %>%
  left_join(oil_q, by = "date")
head(tt)

### Activity 3: join again, but starting from ntl_q. How many rows do you get, and why?
tt2 <- ___ %>%
  left_join(___, by = "date")
nrow(tt); nrow(tt2)

# =============================================================================
# 4. TIDYVERSE: FILTER, MUTATE, SUMMARISE
# =============================================================================

# Year-on-year growth: compare each quarter with the same quarter a year earlier
tt <- tt %>%
  arrange(date) %>%
  mutate(gdp_yoy   = 100 * (gdp   / lag(gdp, 4)   - 1),
         ntl_yoy   = 100 * (ntl   / lag(ntl, 4)   - 1),
         brent_yoy = 100 * (brent / lag(brent, 4) - 1))
tail(tt %>% select(date, gdp_yoy, ntl_yoy, brent_yoy))

tt %>%
  summarise(avg_gdp_yoy = mean(gdp_yoy, na.rm = TRUE),
            sd_gdp_yoy  = sd(gdp_yoy, na.rm = TRUE))

### Activity 4 (a): keep the quarters since 2015.
tt %>% filter(date >= ___)

### Activity 4 (b): the average Brent price since 2015.
tt %>%
  filter(date >= as.Date("2015-01-01")) %>%
  summarise(avg_brent = ___)

# Correlations of year-on-year growth (pairwise, ignoring missing quarters)
tt %>% select(gdp_yoy, ntl_yoy, brent_yoy) %>% cor(use = "pairwise.complete.obs")

# =============================================================================
# 5. PLOTS WITH GGPLOT2
# =============================================================================

ggplot(tt, aes(x = date, y = gdp)) +
  geom_line(color = "#004e70", linewidth = 1) +
  labs(title = "Trinidad and Tobago: real GDP", x = NULL, y = "TT$ millions, constant prices") +
  theme_minimal()

# Growth rates on one chart: one geom_line() per series; a label for color inside aes()
# builds the legend
ggplot(tt, aes(x = date)) +
  geom_line(aes(y = gdp_yoy, color = "GDP")) +
  geom_line(aes(y = ntl_yoy, color = "Night lights")) +
  labs(title = "Year-on-year growth, %", x = NULL, y = NULL, color = NULL) +
  theme_minimal()

# Distribution of GDP growth
ggplot(tt, aes(x = gdp_yoy)) +
  geom_density(fill = "lightblue", alpha = 0.7, na.rm = TRUE) +
  labs(title = "Distribution of real GDP growth", x = "Real GDP, YoY %", y = "Density") +
  theme_minimal()

# Scatter plot with a fitted line: night-time lights and GDP growth
ggplot(tt, aes(x = ntl_yoy, y = gdp_yoy)) +
  geom_point(color = "darkred", size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "black") +
  labs(title = "Night-time lights and GDP growth",
       x = "Night-time lights, YoY %", y = "Real GDP, YoY %") +
  theme_minimal()

### Activity 5: a scatter plot of GDP growth (y) against Brent growth (x), with a fitted line.
ggplot(tt, aes(x = ___, y = ___)) +
  geom_point(color = "darkred", size = 2, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "black") +
  labs(title = "Oil prices and GDP growth", x = "Brent, YoY %", y = "Real GDP, YoY %") +
  theme_minimal()

# Save the quarterly table for later
write_csv(tt, "../output/tt_quarterly.csv")
