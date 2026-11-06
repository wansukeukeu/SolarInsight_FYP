# --- DATA PREPROCESSING FOR MACHINE LEARNING ---

import pandas as pd

# 1. Load the two Plant 1 CSVs
gen_df = pd.read_csv('Plant_1_Generation_Data.csv')
weather_df = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')

# 2. Convert DATE_TIME to datetime objects in both dataframes to ensure a perfect match
gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], dayfirst=True)
weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], dayfirst=True)

# 3. The generation data has multiple rows per timestamp (one for each inverter). 
# We need to sum the AC_POWER for the whole plant per 15-minute interval.
plant_generation = gen_df.groupby('DATE_TIME')['AC_POWER'].sum().reset_index()

# 4. Merge the total generation with the weather sensor features
merged_df = pd.merge(plant_generation, weather_df[['DATE_TIME', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']], on='DATE_TIME', how='inner')

# Verify the merge
print(merged_df.head())
print(f"Total rows ready for machine learning: {len(merged_df)}")

import numpy as np

# --- FEATURE ENGINEERING ---

# 1. Ensure data is sorted chronologically before shifting
merged_df = merged_df.sort_values('DATE_TIME').reset_index(drop=True)

# 2. Extract standard hour and month
merged_df['Hour'] = merged_df['DATE_TIME'].dt.hour
merged_df['Month'] = merged_df['DATE_TIME'].dt.month

# 3. Cyclical encoding for Hour (0-23) and Month (1-12)
merged_df['hour_sin'] = np.sin(2 * np.pi * merged_df['Hour'] / 24.0)
merged_df['hour_cos'] = np.cos(2 * np.pi * merged_df['Hour'] / 24.0)

merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['Month'] / 12.0)
merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['Month'] / 12.0)

# 4. Create Lag Features (15-min intervals: 4 periods = 1 hr, 96 periods = 24 hrs)
merged_df['generation_lag_1h'] = merged_df['AC_POWER'].shift(4)
merged_df['generation_lag_24h'] = merged_df['AC_POWER'].shift(96)

# 5. Clean up temporary columns and drop the first 24 hours of rows 
# (since they won't have 24-hour lag data available)
merged_df = merged_df.drop(columns=['Hour', 'Month'])
final_ml_df = merged_df.dropna().reset_index(drop=True)

print("--- Engineered Features ---")
print(final_ml_df[['DATE_TIME', 'AC_POWER', 'hour_sin', 'generation_lag_1h', 'generation_lag_24h']].head())
print(f"\nFinal rows ready for XGBoost: {len(final_ml_df)}")

# --- MACHINE LEARNING MODELING / TRAINING ---

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 1. Define Features (X) and Target (y)
# Drop DATE_TIME (non-numeric) and AC_POWER (our target)
X = final_ml_df.drop(columns=['DATE_TIME', 'AC_POWER'])
y = final_ml_df['AC_POWER']

# 2. Train/Test Split (80% training, 20% testing)
# shuffle=False is critical for time-series forecasting!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 3. Initialize and train the Linear Regression baseline
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred_lr = lr_model.predict(X_test)

# 5. Evaluate the model performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae = mean_absolute_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)

print("\n--- Linear Regression Baseline Performance ---")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R-squared (R2): {r2:.4f}")

# --- XGBOOST MODELING / TRAINING ---

import xgboost as xgb
import joblib

# 1. Initialize and train the tuned XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=150, 
    learning_rate=0.05, 
    max_depth=6, 
    subsample=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# 2. Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test)

# 3. Evaluate the model performance
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_r2 = r2_score(y_test, y_pred_xgb)

# 4. Calculate Normalized Metrics to check against your PRD KPIs
mean_generation = y_test.mean()
n_rmse = (xgb_rmse / mean_generation) * 100
n_mae = (xgb_mae / mean_generation) * 100

print("\n--- Tuned XGBoost Model Performance ---")
print(f"RMSE: {xgb_rmse:.2f}")
print(f"MAE: {xgb_mae:.2f}")
print(f"R-squared (R2): {xgb_r2:.4f}")
print(f"Normalized RMSE (KPI Target <= 8%): {n_rmse:.2f}%")
print(f"Normalized MAE (KPI Target <= 6%): {n_mae:.2f}%")

# 5. Save the trained model for the Streamlit dashboard
joblib.dump(xgb_model, 'xgboost_solar_model.pkl')
print("\nModel successfully saved as 'xgboost_solar_model.pkl'")