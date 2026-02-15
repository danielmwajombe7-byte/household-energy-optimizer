import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="Smart Energy Consumption", page_icon="⚡", layout="wide")

# ==========================
# LOAD DATA (with proper mapping)
# ==========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("tanzania_power_data.csv", sep=";", engine="python")
        # Convert Date & Time if exist
        if "Date" in df.columns and "Time" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
            df.drop(columns=["Date", "Time"], inplace=True)

        # Map original dataset columns to standard feature names
        col_map = {}
        if "Sub_metering_1" in df.columns: col_map["Sub_metering_1"] = "Kitchen_Power"
        if "Sub_metering_2" in df.columns: col_map["Sub_metering_2"] = "Laundry_Power"
        if "Global_reactive_power" in df.columns: col_map["Global_reactive_power"] = "Extra_Loss"
        if "Global_intensity" in df.columns: col_map["Global_intensity"] = "Current"
        df.rename(columns=col_map, inplace=True)

        # Ensure all features exist in dataset
        for col in ["Voltage", "Current", "Kitchen_Power", "Laundry_Power", "Extra_Loss"]:
            if col not in df.columns:
                df[col] = df[col].mean() if col in df.columns else 0.0

        # Convert all to numeric
        for col in ["Voltage", "Current", "Kitchen_Power", "Laundry_Power", "Extra_Loss"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(inplace=True)
        df["Global_active_power"] = df["Kitchen_Power"] + df["Laundry_Power"] + df["Extra_Loss"]

    except Exception:
        # fallback mini dataset
        df = pd.DataFrame({
            "Voltage": [220,230,210,225,240,200,215,235,220,210],
            "Current": [5,6,4.5,5.5,6.5,4,4.8,6,5.2,4.6],
            "Kitchen_Power": [1.2,1.5,1.0,1.3,1.8,0.9,1.1,1.6,1.4,1.0],
            "Laundry_Power": [0.8,1.0,0.6,0.9,1.2,0.5,0.7,1.1,0.9,0.6],
            "Extra_Loss": [0.3,0.4,0.2,0.35,0.5,0.15,0.25,0.45,0.3,0.2]
        })
        df["Global_active_power"] = df["Kitchen_Power"] + df["Laundry_Power"] + df["Extra_Loss"]

    return df

df = load_data()

FEATURES = ["Voltage", "Current", "Kitchen_Power", "Laundry_Power", "Extra_Loss"]
TARGET = "Global_active_power"
AVG_POWER = df[TARGET].mean()
HIGH_THRESHOLD = AVG_POWER * 1.3  # Very High Consumption

# ==========================
# TRAIN MODEL
# ==========================
@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# ==========================
# SIDEBAR MENU
# ==========================
st.sidebar.title("📂 Menu")
page = st.sidebar.radio("Chagua Page", ["Home", "Prediction", "Visualization"])

# ==========================
# SESSION STATE
# ==========================
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "user_info" not in st.session_state:
    st.session_state.user_info = {"name": None, "building": None}

# ==========================
# PAGE 1: HOME
# ==========================
if page == "Home":
    st.markdown("""
    <div style="background: linear-gradient(90deg,#0f2027,#203a43,#2c5364);
                padding:30px; border-radius:15px; text-align:center;">
        <div style="font-size:55px;color:#facc15;">💡</div>
        <h1 style="color:white;">Smart Energy Consumption AI App</h1>
        <p style="color:#d1d5db;">Machine Learning Based Energy Prediction</p>
    </div>
    """, unsaf
