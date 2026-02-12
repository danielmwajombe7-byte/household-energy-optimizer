#!/usr/bin/env python
# coding: utf-8

# ===============================
# Energy Consumption Optimizer - Excel Version
# ===============================

# Step 0: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load dataset directly from Excel
excel_path = r"C:\Users\user\Desktop\ML_PRROJECT\individual_household_power.xlsx"

df = pd.read_excel(excel_path)

# Step 2: Combine Date + Time into datetime
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)

# Step 3: Convert numeric columns
numeric_cols = ['Global_active_power','Global_reactive_power','Voltage',
                'Sub_metering_1','Sub_metering_2','Sub_metering_3']
df[numeric_cols] = df[numeric_cols].astype(float)

# Step 4: Feature engineering
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month

# Rolling average past 3 hours
df['rolling_avg_3h'] = df['Global_active_power'].rolling(window=3).mean().bfill()

# Step 5: Drop any rows with NaN in features or target
df_clean = df.dropna(subset=['Global_active_power','rolling_avg_3h','hour','day_of_week','month'])

# Step 6: Prepare X and y
X = df_clean[['hour','day_of_week','month','rolling_avg_3h']]
y = df_clean['Global_active_power']

# Confirm lengths
print("X length:", len(X))
print("y length:", len(y))

# Step 7: Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 8: Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Predict
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

# Step 10: Detect High Energy Usage
threshold = y_train.mean() + y_train.std()
high_usage_hours = X_test[y_pred > threshold]
print("⚡ High energy usage predicted at these times:")
print(high_usage_hours.head())

# Step 11: Generate Recommendations
def recommend_action(pred_value, threshold):
    if pred_value > threshold:
        return "⚡ High usage predicted! Turn off non-essential appliances or delay heavy usage."
    else:
        return "✅ Energy usage normal."

# Sample recommendations for last 10 predictions
for pred in y_pred[-10:]:
    print(recommend_action(pred, threshold))

# Step 12: Optional - Plot a sample
plt.figure(figsize=(12,6))
plt.plot(df_clean['datetime'][:1000], df_clean['Global_active_power'][:1000])
plt.xlabel("Datetime")
plt.ylabel("Global Active Power (kW)")
plt.title("Energy Usage Sample")
plt.show()
