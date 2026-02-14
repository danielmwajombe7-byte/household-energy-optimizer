import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="⚡ Smart Energy Consumption",
    page_icon="⚡",
    layout="wide"
)

# =====================================================
# NAVIGATION FUNCTION (FIXED)
# =====================================================
def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()

# =====================================================
# SESSION STATE INIT
# =====================================================
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
df_numeric = df.select_dtypes(include="number")

# =====================================================
# MODEL
# =====================================================
target = "Global_active_power"
features = df_numeric.columns.drop(target)

X = df_numeric[features]
y = df_numeric[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=80,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div style="text-align:center; margin-bottom:30px;">
    <h1 style="font-family:Segoe UI; font-size:42px; color:#0f172a;">
        ⚡ Smart Energy Consumption Dashboard
    </h1>
    <p style="color:#475569; font-size:16px;">
        Predict • Monitor • Optimize Electricity Usage
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# WELCOME / DASHBOARD PAGE
# =====================================================
if st.session_state.page == "welcome":

    st.subheader("👋 Welcome")

    user_name = st.text_input("Enter your name")
    building = st.selectbox(
        "Select Building Type",
        ["House", "Office", "School", "Factory"]
    )

    if st.button("➡️ Proceed to Prediction", use_container_width=True):
        if user_name.strip() == "":
            st.warning("Please enter your name")
        else:
            st.session_state.user = user_name
            st.session_state.building = building
            go_to("prediction")

# =====================================================
# PREDICTION FORM PAGE
# =====================================================
elif st.session_state.page == "prediction":

    st.markdown(
        f"### 👤 User: **{st.session_state.user}** | 🏠 **{st.session_state.building}**"
    )

    st.markdown(
        "<h3 style='text-align:center;'>ENTER VALUES FOR PREDICTION</h3>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    user_input = {}

    with col1:
        user_input["Global_active_power"] = st.number_input(
            "Total Power Usage (kW)",
            value=float(df_numeric["Global_active_power"].mean())
        )
        user_input["Voltage"] = st.number_input(
            "Electricity Voltage (V)",
            value=float(df_numeric["Voltage"].mean())
        )
        user_input["Sub_metering_1"] = st.number_input(
            "Kitchen Power Usage",
            value=float(df_numeric["Sub_metering_1"].mean())
        )

    with col2:
        user_input["Global_reactive_power"] = st.number_input(
            "Extra Power Loss",
            value=float(df_numeric["Global_reactive_power"].mean())
        )
        user_input["Global_intensity"] = st.number_input(
            "Current Intensity (A)",
            value=float(df_numeric["Global_intensity"].mean())
        )
        user_input["Sub_metering_2"] = st.number_input(
            "Laundry Power Usage",
            value=float(df_numeric["Sub_metering_2"].mean())
        )

    st.markdown("<br>", unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        if st.button("⚡ Predict Energy Consumption", use_container_width=True):
            full_input = {}
            for f in features:
                full_input[f] = user_input.get(f, float(df_numeric[f].mean()))

            input_df = pd.DataFrame([full_input])
            st.session_state.prediction = model.predict(input_df)[0]
            go_to("result")

    with colB:
        if st.button("🏠 Go Home", use_container_width=True):
            go_to("welcome")

# =====================================================
# RESULT PAGE
# =====================================================
elif st.session_state.page == "result":

    prediction = st.session_state.prediction

    st.markdown(
        f"""
        <div style="text-align:center;">
            <h2>⚡ Predicted Energy Consumption</h2>
            <h1 style="color:#16a34a;">{prediction:.3f} kW</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    if prediction > y.mean():
        st.warning("⚠️ High energy usage detected. Consider reducing heavy appliances.")
    else:
        st.success("✅ Energy usage is within normal range.")

    show_graph = st.checkbox("📊 Show Energy Comparison Graph")

    if show_graph:
        df_plot = pd.DataFrame({
            "Stage": ["Low", "Average", "Prediction", "High"],
            "Consumption": [y.min(), y.mean(), prediction, y.max()]
        })

        fig = px.line(
            df_plot,
            x="Stage",
            y="Consumption",
            markers=True,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    colX, colY = st.columns(2)

    with colX:
        if st.button("🔁 New Prediction", use_container_width=True):
            go_to("prediction")

    with colY:
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            go_to("welcome")
