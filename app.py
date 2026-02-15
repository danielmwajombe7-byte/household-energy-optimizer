import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Energy Consumption",
    page_icon="⚡",
    layout="wide"
)

# =====================================================
# LOAD TRAINED MODEL
# =====================================================
@st.cache_resource
def load_model():
    model_file = "model.pkl" 
    if not os.path.exists(model_file):
        st.error(f"❌ model.pkl not found! Make sure it is uploaded in the app folder.")
        return None
    return joblib.load(model_file)

model = load_model()
if model is None:
    st.stop()  # stop app if model not found

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    csv_file = "tanzania_power_data.csv"  # ensure this CSV is uploaded in the same folder
    if not os.path.exists(csv_file):
        st.error(f"❌ CSV file not found! Upload tanzania_power_data.csv in the app folder.")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_file, sep=";", engine="python")

    if "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
        df.drop(columns=["Date", "Time"], inplace=True)

    for col in df.columns:
        if col != "Datetime":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna()

df = load_data()
if df.empty:
    st.stop()  # stop if CSV missing

df_num = df.select_dtypes(include="number")

TARGET = "Global_active_power"
FEATURES = df_num.columns.drop(TARGET)

# =====================================================
# USER INPUTS & PREDICTION (unchanged)
# =====================================================
