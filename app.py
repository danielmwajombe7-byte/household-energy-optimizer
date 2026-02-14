import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Energy Consumption",
    page_icon="⚡",
    layout="wide"
)

# =====================================================
# NAVIGATION FUNCTION
# =====================================================
def go_to(page):
    st.session_state.page = page
    st.rerun()

if "page" not in st.session_state:
    st.session_state.page = "welcome"

# =====================================================
# LOAD DATA
# =====================================================
DATA_PATH = "tanzania_power_data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=";", engine="python")

    if "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"],
            dayfirst=True,
            errors="coerce"
        )
        df.drop(columns=["Date", "Time"], inplace=True)

    for col in df.columns:
        if col != "Datetime":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna()

df = load_data()
df_num = df.select_dtypes(include="number")

# =====================================================
# MODEL
# =====================================================
target = "Global_active_power"
features = df_num.columns.drop(target)

X = df_num[features]
y = df_num[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42
)
model.fit(X_train, y_train)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div style="
    background: linear-gradient(90deg,#0f2027,#203a43,#2c5364);
    padding:30px;
    border-radius:15px;
    text-align:center;
    margin-bottom:30px;
">
<h1 style="color:white; font-family:Segoe UI;">⚡ Smart Energy Consumption Dashboard</h1>
<p style="color:#d1d5db; font-size:16px;">Predict • Monitor • Optimize Electricity Usage</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# WELCOME PAGE
# =====================================================
if st.session_state.page == "welcome":

    st.subheader("👋 Welcome")
    st.info("This system helps you understand and reduce electricity consumption.")

    user = st.text_input("👤 Enter your name")
    building = st.selectbox(
        "🏢 Select Building Type",
        ["House", "Office", "School", "Factory"]
    )

    if st.button("➡️ Proceed to Prediction", use_container_width=True):
        if user.strip() == "":
            st.warning("Please enter your name")
        else:
            st.session_state.user = user
            st.session_state.building = building
            go_to("prediction")

# =====================================================
# PREDICTION PAGE
# =====================================================
elif st.session_state.page == "prediction":

    st.markdown(f"""
    <div style="background:#f1f5f9;padding:15px;border-radius:10px;">
    👤 <b>{st.session_state.user}</b> |
    🏠 <b>{st.session_state.building}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h3 style="text-align:center;color:#1e3a8a;">
    ENTER VALUES FOR PREDICTION
    </h3>
    """, unsafe_allow_html=True)

    # ================= INPUT GRID =================
    col1, col2 = st.columns(2)
    user_input = {}

    with col1:
        user_input["Global_active_power"] = st.number_input(
            "🔌 Total Power Usage (kW)", value=float(y.mean())
        )
        user_input["Voltage"] = st.number_input(
            "⚡ Electricity Voltage (V)", value=float(df_num["Voltage"].mean())
        )
        user_input["Sub_metering_1"] = st.number_input(
            "🍳 Kitchen Power Usage", value=float(df_num["Sub_metering_1"].mean())
        )

    with col2:
        user_input["Global_reactive_power"] = st.number_input(
            "🔥 Extra Power Loss", value=float(df_num["Global_reactive_power"].mean())
        )
        user_input["Global_intensity"] = st.number_input(
            "🔁 Current Intensity (A)", value=float(df_num["Global_intensity"].mean())
        )
