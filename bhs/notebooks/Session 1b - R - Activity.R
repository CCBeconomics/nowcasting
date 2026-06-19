
# =============================================================================
# ACTIVITY SOLUTION
# =============================================================================

# For the activity replace the XXX with the corresponding code

# Load libraries
# Load required packages
library(tidyverse)
library(lubridate)
library(openxlsx)

# 1. Read (import) both files, "us_gdp.xlsx" and "us_stocks.xlsx":
us_gdp <- read.xlsx("us_gdp.xlsx")
us_gdp

# Complete for us_stocks:
us_stocks
us_stocks

# 2. Clean the data (check date columns data types)
us_gdp <- us_gdp %>%
  mutate(date = make_date(date))

us_stocks <- XXX %>%
  XXX(date = make_date(date))

# 3. Join the data frames
us_merged <- left_join(us_stocks_clean, XXX, by = "date")

# 4. Obtain the average and standard deviation of the NASDAQ
nasdaq_stats <- us_merged %>%
  summarise(
    avg_nasdaq = mean(nasdaq),
    sd_nasdaq = sd(XXX)
  )

nasdaq_stats

# 5. Make a scatter plot
ggplot(us_merged, aes(x = XXX, y = XXX)) +
  geom_point(color = "darkblue", size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(
    title = "Relationship between NASDAQ and GDP",
    x = "NASDAQ Index",
    y = "Real GDP"
  ) +
  theme_minimal()

# 6. Write (export) the merged data frame as a .xlsx file
write.xlsx(XXX, "us_merged.xlsx")
