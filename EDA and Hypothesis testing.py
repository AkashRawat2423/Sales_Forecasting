# -*- coding: utf-8 -*-
"""Product sales forecasting.ipynb

# EDA and Hypothesis testing

## EDA

1: Import Libraries and Load Data
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
train_df = pd.read_csv('TRAIN.csv', parse_dates=['Date'])

# Display basic info
print(train_df.info())
print(train_df.head())

# Check for missing values
print("Missing values per column:\n", train_df.isnull().sum())

"""2: Exploratory Data Analysis (EDA)"""

# 2.1 Univariate Analysis

# Sales Distribution
plt.figure(figsize=(10, 5))
sns.histplot(train_df['Sales'], bins=30, kde=True)
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# Box Plot for Sales
plt.figure(figsize=(7, 4))
sns.boxplot(x=train_df['Sales'])
plt.title('Box Plot of Sales')
plt.show()

"""***1. Sales Distribution Histogram:***

    This histogram displays the frequency of different sales values.

    The distribution is right-skewed, meaning most sales are on the lower end, with fewer instances of very high sales.

    A density curve is overlaid to show the overall trend of sales data.


***2. Box Plot of Sales:***

    This plot shows the distribution of sales values, highlighting the median, interquartile range, and outliers.

    There are many outliers on the higher end, indicating some exceptionally high sales values.

    The sales data has a wide spread, with most values concentrated on the lower end.
"""

#2.2 Bivariate Analysis

# Sales vs. Discounts
plt.figure(figsize=(8, 5))
sns.boxplot(x=train_df['Discount'], y=train_df['Sales'])
plt.title('Sales Distribution with and without Discounts')
plt.xlabel('Discount')
plt.ylabel('Sales')
plt.show()

# Correlation Matrix
numeric_cols = train_df.select_dtypes(include=['number']) # Selecting only numeric columns
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

"""***1. Sales Distribution with and without Discounts:***

    This box plot compares sales when discounts were offered versus when they were not.

    Sales appear to be higher on average when discounts were applied.

    Both categories have a significant number of high-value outliers.

***2. Correlation Matrix:***

    Sales and #Order have a strong positive correlation (0.94), indicating that higher orders lead to higher sales.

    Holiday has a slight negative correlation with both Sales (-0.15) and #Order (-0.14), suggesting lower sales on holidays.

    Store_id has almost no correlation with other variables, meaning store identity doesn’t impact sales directly.
"""

# 2.3 Time Series Analysis

# Sales Over Time
plt.figure(figsize=(12, 5))
sns.lineplot(data=train_df, x='Date', y='Sales')
plt.title('Sales Trend Over Time')
plt.xticks(rotation=45)
plt.show()

# Sales by Month
train_df['Date'] = pd.to_datetime(train_df['Date'], errors='coerce')
train_df['Month'] = train_df['Date'].dt.month
plt.figure(figsize=(8, 5))
sns.boxplot(x='Month', y='Sales', data=train_df)
plt.title('Sales Variation Across Months')
plt.show()

"""***1. Sales Trend Over Time:***

    This time-series plot shows sales fluctuations over time.

    There are recurring peaks and dips, indicating periodic trends in sales.

    Some sharp declines may represent seasonal drops or anomalies.

***2. Sales Variation Across Months:***

    This box plot shows how sales vary month by month.

    The median sales are fairly consistent across months, but outliers exist in all months.

    Some months may have a wider spread, indicating higher variation in sales.
"""

# 2.4 Categorical Data Analysis

#Sales by Store Type
plt.figure(figsize=(8, 5))
sns.boxplot(x='Store_Type', y='Sales', data=train_df)
plt.title('Sales Across Different Store Types')
plt.show()

# Sales by Location Type
plt.figure(figsize=(8, 5))
sns.boxplot(x='Location_Type', y='Sales', data=train_df)
plt.title('Sales Across Different Locations')
plt.show()

"""***1. Sales Across Different Store Types:***

    This plot compares sales distribution across different store types (S1, S2, S3, S4).

    Some store types have higher median sales than others, with Store Type S4 showing the highest spread.

    Outliers are present in all categories, suggesting some stores experience occasional high sales.

***2. Sales Across Different Locations:***

    This box plot visualizes sales distribution across different location types (L1, L2, L3, L4, L5).

    Locations L2 and L3 have the highest sales medians, while L4 and L5 have lower values.

    The presence of outliers in all locations suggests some high-revenue stores regardless of location.
"""

# 2.5 Handle Missing Values

# Check for missing values
missing_values = train_df.isnull().sum()
print("Missing values:\n", missing_values)
train_df.ffill(inplace=True)
print("Missing values after treatment : \n", train_df.isnull().sum())

# 2.6 Outlier Detection

# Detect outliers using IQR
Q1 = train_df['Sales'].quantile(0.25)
Q3 = train_df['Sales'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier threshold
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
outliers = train_df[(train_df['Sales'] < lower_bound) | (train_df['Sales'] > upper_bound)]
print("Number of outliers:", len(outliers))

"""## Hypothesis Testing"""

# 1 Impact of Discounts on Sales

from scipy.stats import ttest_ind

# Separate sales based on discount
sales_with_discount = train_df[train_df['Discount'] == 'Yes']['Sales']
sales_without_discount = train_df[train_df['Discount'] == 'No']['Sales']

# Perform t-test
t_stat, p_value = ttest_ind(sales_with_discount, sales_without_discount, equal_var=False)
print(f"T-statistic: {t_stat}, P-value: {p_value}")

"""Discounts significantly boost sales, as shown by the high t-statistic and p-value of 0."""

# 2 Effect of Holidays on Sales

sales_holiday = train_df[train_df['Holiday'] == 1]['Sales']
sales_non_holiday = train_df[train_df['Holiday'] == 0]['Sales']

t_stat, p_value = ttest_ind(sales_holiday, sales_non_holiday, equal_var=False)
print(f"T-statistic: {t_stat}, P-value: {p_value}")

"""Sales are lower on holidays, indicating reduced customer activity on those days."""

# 3 Sales Differences Across Store Types

from scipy.stats import f_oneway

# Separate sales by store type
store_types = train_df['Store_Type'].unique()
sales_by_store = [train_df[train_df['Store_Type'] == store]['Sales'] for store in store_types]

# Perform ANOVA
f_stat, p_value = f_oneway(*sales_by_store)
print(f"F-statistic: {f_stat}, P-value: {p_value}")

"""Store type strongly influences sales, with significant differences across categories."""

# 4 Regional Sales Variability

from scipy.stats import kruskal

# Separate sales by region
regions = train_df['Region_Code'].unique()
sales_by_region = [train_df[train_df['Region_Code'] == region]['Sales'] for region in regions]

# Perform test
h_stat, p_value = kruskal(*sales_by_region)
print(f"H-statistic: {h_stat}, P-value: {p_value}")

"""Regional sales vary considerably, highlighting geographic differences in performance."""

# 5 Correlation between Orders and Sales

from scipy.stats import pearsonr

corr, p_value = pearsonr(train_df['#Order'], train_df['Sales'])
print(f"Pearson Correlation: {corr}, P-value: {p_value}")

"""Orders and sales show a strong positive correlation, meaning more orders lead to higher sales."""