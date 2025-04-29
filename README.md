# _** Product Sales Forecasting**_ 

## Problem Statement

_In the competitive retail industry, accurate sales forecasting is critical for inventory management, 
financial planning, promotional strategy, and overall business operations. This project aims to forecast 
daily product sales per store using historical data that includes store demographics, holiday indicators, 
discounts, and order volume._

## Target Metric

* Primary Target Variable: **Sales**
* Evaluation Metrics:
     - Mean Absolute Error (MAE)
     - Root Mean Squared Error (RMSE)
     - R² Score

## Project Workflow

1. Exploratory Data Analysis (EDA)

- Sales Trends: Identified patterns based on time, store type, and regional distribution.

- Holiday Impact: Found decreased sales during holidays.

- Discount Impact: Noticed higher sales when discounts were applied.

- Store Type & Location Insights:

      Urban stores generally had higher sales.

      Certain store types consistently outperformed others.

- Sales vs. Orders: High correlation (0.94) confirmed that order volume is a strong driver of revenue.

2. Hypothesis Testing

- T-Test: Proved that discounts significantly increase sales (p-value = 0.0).

- T-Test on Holidays: Confirmed that holidays reduce average sales (p-value = 0.0).

- ANOVA: Revealed significant variation in sales across different store types.

- Kruskal-Wallis Test: Showed significant regional impact on sales figures.

3. Data Preprocessing

- Missing Value Handling: Imputed using mean/median as appropriate.

- Encoding: One-Hot Encoding for categorical variables (Store_Type, Location_Type).

- Scaling: MinMaxScaler for numeric features.

- Lag Features: Created Sales_Lag_7, Sales_Lag_30 for temporal dependency modeling.

- Split: 80% training, 20% testing.

4. Machine Learning Models

|       Model        |  MAE   |  RMSE  | R² Score |
|:------------------:|:------:|:------:|:--------:|
| Linear Regression	 | 0.0136 | 0.0188 |  0.9449  |
|      XGBoost       | 0.0088 | 0.0141 |  0.9692  |
|        LSTM        | 0.0229 | 0.0271 |  0.8860  |


* XGBoost performed the best:
  - Captured non-linear relationships
  - Required minimal feature engineering
  - Generalized well on unseen data

## Deployment Steps

1. Model Serialization:
 Trained model saved using joblib for XGBoost.

2. Web API Development:
- Created a REST API using Flask or FastAPI to take inputs and return predictions.
- Input: JSON containing store ID, date, discount, holiday, etc.
- Output: Forecasted sales.

3. Frontend or Tableau Integration:
  Optionally visualized real-time forecasts using Tableau dashboards or a web dashboard.

## Insights & Recommendations

- Apply targeted discounts: Especially in high-performing store types and urban areas.

- Avoid promotions during holidays: As holiday sales naturally drop.

- Focus inventory on regions showing higher variability in sales.

- Consider external features like weather or local events to improve LSTM performance.

- Automate daily forecasts to dynamically adjust operations and marketing plans.

## Files in This Repository

| File      | Folder	Description                              |
|-----------|-------------------------------------------------|
| Notebooks | EDA, Hypothesis Testing, and Modeling Notebooks |
| data      | Raw and cleaned datasets                        |
| model     | Trained model files (.pkl or .joblib)           |
| app       | Deployment files (Flask or FastAPI app)         |
| README.md | Project overview                                |

 
