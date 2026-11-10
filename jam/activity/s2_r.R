# Introduction to R for Economic Analysis
# Inter-American Development Bank Workshop
# Complete R Script

# =============================================================================
# SETUP AND PACKAGE INSTALLATION
# =============================================================================

# Install packages (run this once manually in console)
install.packages(c("tidyverse", "lubridate", "openxlsx"))

# Load required packages
library(tidyverse)
library(lubridate)
library(openxlsx)

# =============================================================================
# BASIC R OPERATIONS
# =============================================================================

# First R program
print("Hello world")

# Arithmetic operators
2 + 5
2 * 5
2^5

# Logical operators
2 + 5 > 2 * 5
2^5 != 2 * 5

# Assignment operator
country <- "Jamaica"
country

# =============================================================================
# DATA TYPES AND STRUCTURES
# =============================================================================

# Data types
country <- "Jamaica"
gdp <- 22300
is_island <- TRUE

class(country)
class(gdp)
class(is_island)

# Vectors
x <- c(10, 20, 30)
x

# Lists (can hold different data types)
x <- c(10, 20, "apple", "pineapple", 10<20, 10<5)
x

# Matrices
x <- matrix(c(1,2,3,4,5,6), nrow = 3, ncol = 2)
x

# Data frames
df <- data.frame(
  country = c("US", "UK", "JM"),
  gdp_pc = c(94400, 67600, 13900)
)
df

# Data exploration
str(df)

# View data frame
View(df)

# =============================================================================
# DATA SELECTION AND MANIPULATION
# =============================================================================

# Selecting values with $ operator
df$country

# Selecting values with [] operator
df["country"]
df[1,]
df[1,1]

# =============================================================================
# CONTROL STRUCTURES
# =============================================================================

# Conditionals
score <- 72

if(score >= 90){
  print("A")
} else if(score >= 80){
  print("B")
} else {
  print("Below B")
}

# Loops
# For loop
for(i in 1:5){ 
  print(i) 
}

# While loop
i <- 1
while (i < 6) {
  print(i)
  i <- i + 1
}

# Sapply (more efficient than for loops)
sapply(1:5, function(i) i^2)

# =============================================================================
# TIDYVERSE OPERATIONS
# =============================================================================

# Using pipe operator (%>%)
df %>%
  filter(gdp_pc < 80000) %>%
  mutate(gdp_pc_thousands = gdp_pc / 1000) %>%
  arrange(desc(gdp_pc))

# Tidyverse functions in action
df %>%
  summarise(
    avg_gdp_pc = mean(gdp_pc),
    median_gdp_pc = median(gdp_pc),
    sd_gdp_pc = sd(gdp_pc)
  )

# =============================================================================
# DATA JOINING EXERCISE
# =============================================================================

# Create sample dataframes for joining exercise
gdp <- data.frame(
  country = c("US", "UK", "JM", "US", "UK", "JM", "US", "UK", "JM"),
  year = c(2021, 2021, 2021, 2022, 2022, 2022, 2023, 2023, 2023),
  gdp = c(20895, 17234, 11876, 22418, 16987, 12654, 22783, 18321, 12189),
  growth_rate = c(0.0187, 0.0162, 0.0231, 0.0512, 0.0268, 0.0194, 0.0428, 0.0315, 0.0183)
)

stocks <- data.frame(
  country = c("US", "UK", "JM", "US", "UK", "JM", "US", "UK", "JM"),
  year = c(2021, 2021, 2021, 2022, 2022, 2022, 2023, 2023, 2023),
  stock_index = c(3924, 3117, 1428, 4385, 2976, 1618, 4632, 3284, 1531),
  market_cap = c(41234, 28765, 15892, 41987, 32134, 14678, 46789, 30567, 16432)
)

# Display the dataframes
print("GDP Data:")
gdp

print("Stocks Data:")
stocks

# Join dataframes
merged_df <- left_join(gdp, stocks, by = c("country", "year"))
print("Merged Data:")
merged_df

merged_df %>%
  filter(country == "JM")

# Descriptive statistics
summary(merged_df[c('country', 'gdp', 'stock_index')])

# =============================================================================
# WORKING DIRECTORY AND FILE OPERATIONS
# =============================================================================

# Check current working directory
getwd()

# Change working directory (uncomment and modify path as needed)
setwd("C:\\Users\\agmaz\\Desktop\\Nowcast Training R\\Jamaica")

# =============================================================================
# DATA IMPORT AND EXPORT
# =============================================================================

# Import Excel files
gdp_jam <- read.xlsx("gdp.xlsx")
stocks_jam <- read.xlsx("stocks.xlsx")

print("GDP Data:")
gdp_jam

print("Stocks Data:")
stocks_jam

# Join the imported data
jam_data <- gdp_jam %>% 
  left_join(stocks_jam, by = "date") %>% 
  data.frame()

print("Merged Jamaica Data:")
jam_data

# Date conversion
jam_data$date <- make_date(jam_data$date)
print("Data with proper date format:")
jam_data
str(jam_data)

# Export sample files (for demonstration)
write.xlsx(jam_data, "jam_data.xlsx")

# =============================================================================
# DATA VISUALIZATION WITH GGPLOT2
# =============================================================================

# Line plot
ggplot(jam_data, aes(x = date, y = real_gdp)) +
  geom_line(color = "blue", linewidth = 1) +
  labs(title = "Real GDP Over Time", x = "Date", y = "Real GDP") +
  theme_minimal()

# Density plot
ggplot(jam_data, aes(x = real_gdp)) +
  geom_density(fill = "lightblue", alpha = 0.7) +
  labs(title = "Density Distribution of Real GDP", x = "Real GDP", y = "Density") +
  theme_minimal()

# Scatter plot
ggplot(jam_data, aes(y = real_gdp, x = jse_index)) +
  geom_point(color = "darkred", size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "black") +
  labs(title = "Relationship between Stock Index and GDP", 
       x = "JSE Index", y = "Real GDP") +
  theme_minimal()


# =============================================================================
# ACTIVITY
# =============================================================================

# 1. Read (import) both files
us_gdp <- XXX("us_gdp.xlsx")
us_stocks <- XXX.xlsx("us_stocks.xlsx")

us_gdp
us_XXX

# 2. Clean the data (check date columns data types)
us_gdp_clean <- us_gdp %>%
  mutate(date = make_date(date))

us_stocks_clean <- us_stocks %>%
  mutate(date = make_date(date))

# 3. Join the data frames
us_merged <- us_gdp_clean %>%
  left_join(us_stocks_clean, by = "XXX")

# 4. Obtain the average and standard deviation of the NASDAQ
nasdaq_stats <- us_merged %>%
  XXX(
    avg_nasdaq = XXX(nasdaq, na.rm = TRUE),
    sd_nasdaq = sd(nasdaq, na.rm = TRUE)
  )

nasdaq_stats

# 5. Make a scatter plot
ggplot(XXX, aes(x = nasdaq, y = real_gdp)) +
  geom_point(color = "darkblue", size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(
    title = "Relationship between NASDAQ and GDP",
    x = "NASDAQ Index",
    y = "Real GDP"
  ) +
  theme_minimal()

# 6. Write (export) the merged data frame as a .xlsx file
XXX.xlsx(us_merged, "us_merged.xlsx")

