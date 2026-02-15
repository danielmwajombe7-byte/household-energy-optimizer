import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor
import numpy as np

st.set_page_config(
    page_title="Smart Energy Consumption",
    page_icon="⚡",
    layout="wide"
)

# ==========================
# Load dataset safely
# ==========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("tanzania_power_data.csv", sep=";", engine="python")
        if "Date" in df.columns and "Time" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
            df.drop(columns=["Date", "Time"], inplace=True)
        for col in df.columns:
            if col != "Datetime":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)

        # Map your CSV columns to standard names
        col_map = {}
        if "Sub_metering_1" in df.columns:
            col_map["Sub_metering_1"] = "Kitchen_Power"
        if "Sub_metering_2" in df.columns:
            col_map["Sub_metering_2"] = "Laundry_Power"
        if "Global_reactive_power" in df.columns:
            col_map["Global_reactive_power"] = "Extra_Loss"

        df.rename(columns=col_map, inplace=True)

        # Ensure required columns exist
        for col in ["Kitchen_Power", "Laundry_Power", "Extra_Loss"]:
            if col not in df.columns:
                df[col] = 0.0  # fallback if missing

        # Create target
        df["Global_active_power"] = df["Kitchen_Power"] + df["Laundry_Power"] + df["Extra_Loss"]

    except Exception:
        # Fallback mini dataset
        df = pd.DataFrame({
            "Voltage": [220, 230, 210, 225, 240, 200, 215, 235, 220, 210],
            "Current": [5, 6, 4.5, 5.5, 6.5, 4, 4.8, 6, 5.2, 4.6],
            "Kitchen_Power": [1.2, 1.5, 1.0, 1.3, 1.8, 0.9, 1.1, 1.6, 1.4, 1.0],
            "Laundry_Power": [0.8, 1.0, 0.6, 0.9, 1.2, 0.5, 0.7, 1.1, 0.9, 0.6],
            "Extra_Loss": [0.3, 0.4, 0.2, 0.35, 0.5, 0.15, 0.25, 0.45, 0.3, 0.2]
        })
        df["Global_active_power"] = df["Kitchen_Power"] + df["Laundry_Power"] + df["Extra_Loss"]

    return df

df = load_data()

# ==========================
# Features / Target
# ==========================
FEATURES = ["Voltage", "Current", "Kitchen_Power", "Laundry_Power", "Extra_Loss"]
TARGET = "Global_active_power"

# ==========================
# Train model
# ==========================
@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)
