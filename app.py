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

        col_map = {}
        if "Sub_metering_1" in df.columns: col_map["Sub_metering_1"] = "Kitchen_Power"
        if "Sub_metering_2" in df.columns: col_map["Sub_metering_2"] = "Laundry_Power"
        if "Global_reactive_power" in df.columns: col_map["Global_reactive_power"] = "Extra_Loss"
        if "Global_intensity" in df.columns: col_map["Global_intensity"] = "Current"
        df.rename(columns=col_map, inplace=True)

        for col in ["Volta]()
