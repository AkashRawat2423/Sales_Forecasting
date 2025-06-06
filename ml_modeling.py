# -*- coding: utf-8 -*-
"""ML Modeling.ipynb

# Step 1: Data Processing

1.1 Data Cleaning
"""

import pandas as pd
import numpy as np
import joblib
import os

# Load data
train_df = pd.read_csv("TRAIN.csv")

# Handle missing values (example: replace with median or mode)
train_df['Sales'] = train_df['Sales'].fillna(train_df['Sales'].median())
train_df['#Order'] = train_df['#Order'].fillna(train_df['#Order'].mean())

# Remove duplicates
train_df = train_df.drop_duplicates()

# Display basic stats to verify cleaning
print(train_df.info())

"""1.2 Feature Engineering

"""

# Convert 'Date' column to datetime format
train_df['Date'] = pd.to_datetime(train_df['Date'], format="%d-%m-%Y")

# Extract additional time features
train_df['DayOfWeek'] = train_df['Date'].dt.dayofweek
train_df['WeekOfYear'] = train_df['Date'].dt.isocalendar().week
train_df['Year'] = train_df['Date'].dt.year
train_df['IsWeekend'] = (train_df['DayOfWeek'] >= 5).astype(int)

train_df['Sales_Lag_7'] = train_df['Sales'].shift(7)
train_df['Sales_Moving_Avg_7'] = train_df['Sales'].rolling(window=7).mean()
train_df['Sales_Lag_30'] = train_df['Sales'].shift(30)

# Fill missing values caused by shifting
train_df.fillna(0, inplace=True)

"""1.3 Data Transformation"""

from sklearn.preprocessing import OneHotEncoder

# Ensure column names are clean (remove leading/trailing spaces)
train_df.columns = train_df.columns.str.strip()

# List of categorical columns to encode
categorical_cols = ['Store_Type', 'Location_Type', 'Discount', 'Region_Code']

# Filter only the existing categorical columns in the dataframe
existing_categorical_cols = [col for col in categorical_cols if col in train_df.columns]

# Initialize and fit the OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
encoded_features = encoder.fit_transform(train_df[existing_categorical_cols])

# Convert the encoded features to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(existing_categorical_cols))

# Drop original categorical columns and concatenate encoded data
train_df = train_df.drop(columns=existing_categorical_cols).reset_index(drop=True)
train_df = pd.concat([train_df, encoded_df], axis=1)

print("One-hot encoding applied successfully!")
print(train_df.head())  # Check the transformed dataframe

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_df[['Sales', 'Sales_Lag_7', 'Sales_Moving_Avg_7', 'Sales_Lag_30', '#Order']] = scaler.fit_transform(
    train_df[['Sales', 'Sales_Lag_7', 'Sales_Moving_Avg_7', 'Sales_Lag_30', '#Order']]
)

# Save both the scaler and scaling columns
joblib.dump(scaler, 'scaler.pkl')

"""1.4 Train-Test Split"""

from sklearn.model_selection import train_test_split

# Define independent and dependent variables
X = train_df.drop(columns=['Sales', 'ID', 'Date'])
y = train_df['Sales']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

"""# Step 2: Model Selection

2.1 Baseline Model: Linear Regression
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Train linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Saving the feature order
os.makedirs("model", exist_ok=True)
feature_order = list(X_train.columns)
joblib.dump(feature_order, "model/feature_order.pkl")

# Predict on test data
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
print("Baseline Model Metrics:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lr)}")
print(f"MSE: {mean_squared_error(y_test, y_pred_lr)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr))}")

"""2.2 Complex Models"""

from xgboost import XGBRegressor

# Train an XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict on test data
y_pred_xgb = xgb_model.predict(X_test)

# Predict and evaluate
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Model Metrics:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_xgb)}")
print(f"MSE: {mean_squared_error(y_test, y_pred_xgb)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb))}")

# Time Series Model: Prophet
from prophet import Prophet
import pandas as pd

# Prepare data for Prophet
prophet_df = train_df[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})

# Train Prophet model
prophet = Prophet()
prophet.fit(prophet_df)

# Make future predictions
future = prophet.make_future_dataframe(periods=30)
forecast = prophet.predict(future)

# Plot forecast
prophet.plot(forecast)

"""These plots show actual vs. predicted sales over time. The black dots represent actual sales data, while the blue line represents the model's forecasted values. The shaded region indicates the confidence interval, reflecting the uncertainty in predictions. The model captures overall trends and seasonal patterns, but high fluctuations in actual sales suggest variability.

2.3 Deep Learning Model: LSTM
"""

# 1. Checking Data Types
# Before reshaping for LSTM, checking if all columns are numeric:
print(X_train.dtypes)
print(y_train.dtypes)

# 2. Convert Categorical Features
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Convert categorical variables into numerical
categorical_cols = ['Store_Type', 'Location_Type', 'Region_Code', 'Discount']
for col in categorical_cols:
    if col in X_train.columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])  # Ensure consistency

# 3. Ensure No Missing Values
print(X_train.isnull().sum())
print(y_train.isnull().sum())

# Fill or drop missing values
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
y_train.fillna(0, inplace=True)

# 4. Converting Data to float Type
X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)

# 5. Reshapeing  Data for LSTM
X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 6. Training the LSTM Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# Define LSTM model
model = Sequential([
    Input(shape=(1, X_train.shape[1])),
    LSTM(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=1)

# Predict
y_pred_lstm = model.predict(X_test_lstm)

"""#Step 3: Model Evaluation and Validation

3.1 Compute Error Metrics
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Function to print evaluation metrics
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"{model_name} Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("\n")

# Evaluate models
evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_xgb, "XGBoost")
evaluate_model(y_test, y_pred_lstm.flatten(), "LSTM")

"""3.2 Residual Analysis"""

import seaborn as sns
import matplotlib.pyplot as plt

residuals = y_test - y_pred_xgb
sns.histplot(residuals, bins=30, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

"""This histogram displays the distribution of residuals (errors) in the sales predictions. A sharp peak around zero suggests that the model's predictions are generally accurate. The presence of few extreme values indicates occasional large errors in forecasts. A well-centered and symmetric distribution implies that the model does not systematically overestimate or underestimate sales. The overall narrow spread of errors suggests that the model performs well."""

# Saving trained model & encoder
joblib.dump(lr_model, 'sales_forecast.pkl')
joblib.dump(encoder, 'encoder.pkl')
print("Models saved successfully!")

# Downloading Model Files
from google.colab import files
files.download('sales_forecast.pkl')
files.download('encoder.pkl')
files.download("model/feature_order.pkl")
files.download("scaler.pkl")