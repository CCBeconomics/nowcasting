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

# Arithmetic operators
2 + 5
2 * 5
2^5

# Logical operators
2 + 5 > 2 * 5
2^5 != 2 * 5

# Assignment operator
country <- "Barbados"
country

# =============================================================================
# DATA TYPES AND STRUCTURES
# =============================================================================

# Data types
country <- "Barbados"
gdp <- 8480
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
  country = c("US", "UK", "BB"),
  gdp_pc = c(94400, 67600, 29020)
)
df

# Check structure of data frame
str(df)


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


# =============================================================================
# WORKING DIRECTORY AND FILE OPERATIONS
# =============================================================================

# Check current working directory
getwd()

# Change working directory (uncomment and modify path as needed)
#setwd("C:\\Users\\agmaz\\Desktop\\Nwcst Training R\\Barbados")


# =============================================================================
# DATA IMPORT
# =============================================================================

# Import Excel files
gdp_bb <- read.csv("gdp.csv")

print("GDP Data:")
str(gdp_bb)
head(gdp_bb)


# Date conversion
library(lubridate)
gdp_bb$date <- mdy(gdp_bb$date)
print("Data with proper date format:")
str(gdp_bb)
head(gdp_bb)

### Activity No.1: Import stocks.csv file and convert its date column

stocks_bb <- ___

print("Stocks Data:")
head(stocks_bb)

str(stocks_bb)

stocks_bb$date <- ___
str(stocks_bb)

# =============================================================================
# DATA JOINING 
# =============================================================================

# Join the imported data
bb_data <- gdp_bb %>% 
  left_join(stocks_bb, by = "date") %>% 
  data.frame()

print("Merged Barbados Data:")
head(bb_data)


### Activity No.2: Join the imported data, starting with "stocks_bb"

bb_data2 <- ___ %>% 
  left_join(___, by = "date") %>% 
  data.frame()

print("Merged Barbados Data:")
head(bb_data2)

# What is different about the column order?
summary(bb_data2[c('gdp', 'sp500')])

# Export sample files (for demonstration)
write.xlsx(bb_data2, "bb_data.xlsx")


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
    avg_gdp_pc    = mean(gdp_pc),
    median_gdp_pc = median(gdp_pc),
    sd_gdp_pc     = sd(gdp_pc)
  )

### Activity No.3 (a): Filter bb_data by date since 2015Q1

bb_data %>%
  filter(date > ___)

### Activity No.3 (b): Find the average value for sp500

bb_data %>%
  summarise(
    avg_sp500 = ___
  )


# =============================================================================
# DATA VISUALIZATION WITH GGPLOT2
# =============================================================================

# Line plot
ggplot(gdp_bb, aes(x = date, y = gdp)) +
  geom_line(color = "blue", linewidth = 1) +
  labs(title = "Real GDP Over Time", x = "Date", y = "Real GDP") +
  theme_minimal()

# Density plot
ggplot(gdp_bb, aes(x = gdp)) +
  geom_density(fill = "lightblue", alpha = 0.7) +
  labs(title = "Density Distribution of Real GDP", x = "Real GDP", y = "Density") +
  theme_minimal()

# Scatter plot
ggplot(gdp_bb, aes(y = gdp, x = gdp_business_other_services)) +
  geom_point(color = "darkred", size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "black") +
  labs(title = "Relationship between GDP Business Other Services and GDP", 
       x = "RGDP Other Services", y = "Real GDP") +
  theme_minimal()


### Activity No.4: Create a scatter plot between gdp and sp500

ggplot(bb_data, aes(y = ___, x = ___)) +
  geom_point(color = "darkred", size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "black") +
  labs(title = "Relationship between Stock Index and GDP", 
       x = "S&P500", y = "Real GDP") +
  theme_minimal()

