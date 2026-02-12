# energy_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# ===============================
# Title
# ===============================
st.set_page_config(page_title="⚡ Energy Usage Control Dashboard", layout="wide")
st.title("⚡ Energy Usage Control Dashboard")

# ===============================
# Show current working directory (optional debug)
# ===============================
st.write("Current working directory:", os.getcwd())

# ===============================
# Load dataset directly
# ===============================
DATA_PATH = r"C:\Users\user\Desktop\ML_PRROJECT\individual+household+electric+power+consumption.csv"

try:
    df = pd.read_csv(DATA_PATH, sep=';', na_values='?')
    st.success("✅ Dataset loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Could not find file at {DATA_PATH}. Make sure the path is correct.")
    st.stop()

# ===============================
# Preprocessing
# ===============================
# Combine Date + Time
df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)

# Convert numeric columns
numeric_cols = ['Global_active_power','Global_reactive_power','Voltage',
                'Sub_metering_1','Sub_metering_2','Sub_metering_3']
df[numeric_cols] = df[numeric_cols].astype(float)

# Feature engineering
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month

# Rolling average past 3 hours
df['rolling_avg_3h'] = df['Global_active_power'].rolling(window=3).mean().bfill()

# Drop NaNs
df_clean = df.dropna(subset=['Global_active_power','rolling_avg_3h','hour','day_of_week','month'])

# Features and target
X = df_clean[['hour','day_of_week','month','rolling_avg_3h']]
y = df_clean['Global_active_power']

# ===============================
# Train Random Forest Model
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.write(f"Model RMSE: {rmse:.4f} kW")

# ===============================
# Threshold for High Energy Usage
# ===============================
threshold = y_train.mean() + y_train.std()

# ===============================
# Sidebar - User Input for Prediction
# ===============================
st.sidebar.header("Predict Energy Usage")
hour_input = st.sidebar.slider("Hour (0-23)", 0, 23, 12)
day_input = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
month_input = st.sidebar.slider("Month (1-12)", 1, 12, 6)
rolling_input = st.sidebar.number_input("Past 3h Rolling Avg (kW)", 
                                        min_value=0.0, max_value=float(df['Global_active_power'].max()), 
                                        value=float(df['Global_active_power'].mean()), step=0.1)

input_data = pd.DataFrame({
    'hour': [hour_input],
    'day_of_week': [day_input],
    'month': [month_input],
    'rolling_avg_3h': [rolling_input]
})

predicted_usage = model.predict(input_data)[0]

if predicted_usage > threshold:
    recommendation = "⚡ High usage predicted! Turn off non-essential appliances."
else:
    recommendation = "✅ Energy usage normal."

st.sidebar.markdown(f"**Predicted Energy Usage:** {predicted_usage:.2f} kW")
st.sidebar.markdown(f"**Recommendation:** {recommendation}")

# ===============================
# Main Dashboard Visualizations
# ===============================
st.subheader("📊 Energy Usage Sample Chart")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df_clean['datetime'][:1000], df_clean['Global_active_power'][:1000], color='blue')
ax.set_xlabel("Datetime")
ax.set_ylabel("Global Active Power (kW)")
ax.set_title("Energy Usage (Sample 1000 points)")
st.pyplot(fig)

st.subheader("⚡ High Energy Usage Hours")
high_usage_hours = X_test[y_pred > threshold].copy()
high_usage_hours['Predicted_Power'] = y_pred[y_pred > threshold]
st.dataframe(high_usage_hours.sort_values('Predicted_Power', ascending=False).head(10))

# ===============================
# Dataset Preview
# ===============================
st.subheader("📄 Dataset Preview")
st.dataframe(df_clean.head(10))
