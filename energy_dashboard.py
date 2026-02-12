# Install kwanza ikiwa huna
# pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.title("⚡ Energy Usage Control Dashboard")

# Upload Excel
uploaded_file = st.file_uploader("Upload Excel dataset", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Process dataset
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    numeric_cols = ['Global_active_power','Global_reactive_power','Voltage',
                    'Sub_metering_1','Sub_metering_2','Sub_metering_3']
    df[numeric_cols] = df[numeric_cols].astype(float)

    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['rolling_avg_3h'] = df['Global_active_power'].rolling(window=3).mean().bfill()

    df_clean = df.dropna(subset=['Global_active_power','rolling_avg_3h','hour','day_of_week','month'])
    X = df_clean[['hour','day_of_week','month','rolling_avg_3h']]
    y = df_clean['Global_active_power']

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write(f"Model RMSE: {rmse:.2f}")

    threshold = y_train.mean() + y_train.std()
    high_usage_hours = X_test[y_pred > threshold]
    st.write("⚡ High energy usage predicted at these times:")
    st.dataframe(high_usage_hours.head())

    # Recommendations
    st.subheader("Recommendations for last 10 predictions")
    def recommend_action(pred_value, threshold):
        if pred_value > threshold:
            return "⚡ High usage predicted! Turn off non-essential appliances or delay heavy usage."
        else:
            return "✅ Energy usage normal."
    
    recs = [recommend_action(p, threshold) for p in y_pred[-10:]]
    for r in recs:
        st.write(r)

    # Plot energy usage
    st.subheader("Energy Usage Sample")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df_clean['datetime'][:1000], df_clean['Global_active_power'][:1000])
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Global Active Power (kW)")
    st.pyplot(fig)
