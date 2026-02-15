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
# LOAD DATA
# ==========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("tanzania_power_data.csv", sep=";", engine="python")
        if "Date" in df.columns and "Time" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
            df.drop(columns=["Date", "Time"], inplace=True)

        # Map columns to standard names
        col_map = {}
        if "Sub_metering_1" in df.columns: col_map["Sub_metering_1"] = "Kitchen_Power"
        if "Sub_metering_2" in df.columns: col_map["Sub_metering_2"] = "Laundry_Power"
        if "Sub_metering_3" in df.columns: col_map["Sub_metering_3"] = "Other_Appliances"
        if "Global_reactive_power" in df.columns: col_map["Global_reactive_power"] = "Extra_Loss"
        if "Global_intensity" in df.columns: col_map["Global_intensity"] = "Current"
        df.rename(columns=col_map, inplace=True)

        # Ensure feature columns exist
        for col in ["Voltage", "Current", "Kitchen_Power", "Laundry_Power", "Extra_Loss", "Other_Appliances"]:
            if col not in df.columns:
                df[col] = 0.0

        for col in ["Voltage", "Current", "Kitchen_Power", "Laundry_Power", "Extra_Loss", "Other_Appliances"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(inplace=True)
        # Include Other Appliances in total
        df["Global_active_power"] = df["Kitchen_Power"] + df["Laundry_Power"] + df["Extra_Loss"] + df["Other_Appliances"]

    except Exception:
        df = pd.DataFrame({
            "Voltage": [220,230,210,225,240,200,215,235,220,210],
            "Current": [5,6,4.5,5.5,6.5,4,4.8,6,5.2,4.6],
            "Kitchen_Power": [1.2,1.5,1.0,1.3,1.8,0.9,1.1,1.6,1.4,1.0],
