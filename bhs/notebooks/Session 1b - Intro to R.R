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
country <- "The Bahamas"
country

# =============================================================================
# DATA TYPES AND STRUCTURES
# =============================================================================

# Data types
country <- "The Bahamas"
gdp <- 12500
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
  country = c("US", "UK", "BS"),
  gdp = c(23000, 18000, 5000)
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
  filter(gdp > 10000) %>%
  mutate(gdp_million = gdp / 1000) %>%
  arrange(desc(gdp))

# Tidyverse functions in action
df %>%
  summarise(
    avg_gdp = mean(gdp),
    median_gdp = median(gdp),
    sd_gdp = sd(gdp)
  )

# =============================================================================
# DATA JOINING EXERCISE
# =============================================================================

# Create sample dataframes for joining exercise
gdp <- data.frame(
  country = c("US", "UK", "BS", "US", "UK", "BS", "US", "UK", "BS"),
  year = c(2021, 2021, 2021, 2022, 2022, 2022, 2023, 2023, 2023),
  gdp = c(20895, 17234, 11876, 22418, 16987, 12654, 22783, 18321, 12189),
  growth_rate = c(0.0187, 0.0162, 0.0231, 0.0512, 0.0268, 0.0194, 0.0428, 0.0315, 0.0183)
)

stocks <- data.frame(
  country = c("US", "UK", "BS", "US", "UK", "BS", "US", "UK", "BS"), 
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

# Descriptive statistics
summary(merged_df[c('country', 'gdp', 'stock_index')])

# =============================================================================
# WORKING DIRECTORY AND FILE OPERATIONS
# =============================================================================

# Check current working directory
getwd()

# Change working directory (uncomment and modify path as needed)
# setwd("C:\\Users\\User\\Documents\\R")

# =============================================================================
# DATA IMPORT AND EXPORT
# =============================================================================

# Note: These examples assume you have the files in your working directory
# For demonstration, we'll create sample files first

# Create sample GDP data for export/import demonstration
sample_gdp <- data.frame(
  date = c(2021, 2022, 2023),
  real_gdp = c(11876, 12654, 12189),
  growth_rate = c(0.0231, 0.0194, 0.0183)
)

# Create sample stocks data for export/import demonstration
sample_stocks <- data.frame(
  date = c(2021, 2022, 2023),
  bisx_index = c(1428, 1618, 1531),
  market_cap = c(15892, 14678, 16432)
)

# Export sample files (for demonstration)
write.csv(sample_gdp, "gdp.csv", row.names = FALSE)
write.xlsx(sample_gdp, "gdp.xlsx")
write.xlsx(sample_stocks, "stocks.xlsx")

# Import data
data_csv <- read.csv("gdp.csv", sep = ",")
print("Imported CSV data:")
data_csv

# Import Excel files
gdp_bhs <- read.xlsx("gdp.xlsx")
stocks_bhs <- read.xlsx("stocks.xlsx")

print("GDP Data:")
gdp_bhs

print("Stocks Data:")
stocks_bhs

# =============================================================================
# DATA CLEANING AND MERGING
# =============================================================================

# Join the imported data
bhs_data <- gdp_bhs %>% 
  left_join(stocks_bhs, by = "date") %>% 
  data.frame()

print("Merged Bahamas Data:")
bhs_data

# Date conversion
bhs_data$date <- make_date(bhs_data$date)
print("Data with proper date format:")
bhs_data
str(bhs_data)

# =============================================================================
# DATA VISUALIZATION WITH GGPLOT2
# =============================================================================

# Line plot
ggplot(bhs_data, aes(x = date, y = real_gdp)) +
  geom_line(color = "blue", linewidth = 1) +
  labs(title = "Real GDP Over Time", x = "Date", y = "Real GDP") +
  theme_minimal()

# Density plot
ggplot(bhs_data, aes(x = real_gdp)) +
  geom_density(fill = "lightblue", alpha = 0.7) +
  labs(title = "Density Distribution of Real GDP", x = "Real GDP", y = "Density") +
  theme_minimal()

# Scatter plot
ggplot(bhs_data, aes(y = real_gdp, x = bisx_index)) +
  geom_point(color = "darkred", size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "black") +
  labs(title = "Relationship between Stock Index and GDP", 
       x = "BISX Index", y = "Real GDP") +
  theme_minimal()

# Looping through plots
for(c in unique(merged_df$country)) {
  temp <- merged_df %>% filter(country == c)
  print(
    ggplot(temp, aes(year, gdp)) +
      geom_line(color = "steelblue", linewidth = 1) +
      geom_point(color = "steelblue", size = 2) +
      labs(title = paste("GDP for", c), x = "Year", y = "GDP") +
      theme_minimal()
  )
}
