
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="⚡ Smart Energy Consumption Dashboard",
    layout="wide"
)

# ===============================
# ELECTRICITY PRICE (TZS per kWh)
# ===============================
PRICE_PER_UNIT = 350  # 1 unit = 1 kWh

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("tanzania_power_data.csv")
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

# ===============================
# FEATURES (FOR ML – DEMO)
# ===============================
FEATURES = [
    "Kitchen_Power",
    "Laundry_Power",
    "Other_Use",
    "Extra_Loss",
    "Voltage",
    "Current"
]

TARGET = "Total_Power"

# Ensure required columns exist
for col in FEATURES + [TARGET]:
    if col not in df.columns:
        df[col] = 0.0

# ===============================
# TRAIN MODEL (DEMO PURPOSE)
# ===============================
@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# ===============================
# SESSION STATE INIT
# ===============================
defaults = {
    "prediction": None,
    "duration": 1.0,
    "kitchen": 0.0,
    "laundry": 0.0,
    "other": 0.0,
    "extra": 0.0
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===============================
# SIDEBAR
# ===============================
page = st.sidebar.radio(
    "📌 Navigation",
    ["Dashboard", "Prediction", "Visualization"]
)

# ===============================
# DASHBOARD
# ===============================
if page == "Dashboard":
    st.markdown("""
    <div style="background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);
    padding:25px;border-radius:15px;color:white;text-align:center;">
        <div style="font-size:55px;">💡</div>
        <h1>Smart Energy Consumption Dashboard</h1>
        <p>Predict • Measure • Understand Electricity Usage</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Records", df.shape[0])
    c2.metric("📊 Features", len(FEATURES))
    c3.metric("🎯 Target", "Energy (kWh)")
    c4.metric("🤖 Model", "Decision Tree")

# ===============================
# PREDICTION PAGE
# ===============================
elif page == "Prediction":
    st.header("🧮 Energy Prediction Input")

    col1, col2 = st.columns(2)

    with col1:
        kitchen = st.number_input("🍳 Kitchen Power (kW)", 0.0, value=4.5)
        laundry = st.number_input("🧺 Laundry Power (kW)", 0.0, value=6.0)
        other = st.number_input("💡 Other Usage (kW)", 0.0, value=3.0)

    with col2:
        extra = st.number_input("🔥 Extra Loss (kW)", 0.0, value=2.0)
        voltage = st.number_input("⚡ Voltage (V)", 0.0, value=220.0)
        current = st.number_input("🔁 Current (A)", 0.0, value=4.5)

    st.markdown("---")

    duration = st.slider(
        "⏱️ Duration of Usage (Hours)",
        min_value=0.5,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    if st.button("🚀 Calculate Energy Consumption", use_container_width=True):
        st.session_state.kitchen = kitchen
        st.session_state.laundry = laundry
        st.session_state.other = other
        st.session_state.extra = extra
        st.session_state.duration = duration

        # Calculate total power and energy
        total_power = kitchen + laundry + other + extra
        total_units = total_power * duration  # kWh = units
        total_cost = total_units * PRICE_PER_UNIT  # TZS

        # Strong actionable advice
        if total_units <= 5:
            advice = (
                "✅ Excellent! Your energy usage is very efficient.\n"
                "- Keep using energy-saving appliances.\n"
                "- Switch off devices when not in use.\n"
                "- Continue monitoring your usage regularly."
            )
        elif total_units <= 15:
            advice = (
                "🙂 Good, but moderate usage detected.\n"
                "- Turn off devices when idle.\n"
                "- Avoid using multiple high-power appliances simultaneously.\n"
                "- Consider scheduling laundry and kitchen use efficiently."
            )
        elif total_units <= 30:
            advice = (
                "⚠️ High usage detected.\n"
                "- Consider replacing older appliances with energy-efficient ones.\n"
                "- Reduce simultaneous use of high-power devices.\n"
                "- Monitor your usage daily to spot heavy consumers."
            )
        elif total_units <= 60:
            advice = (
                "🚨 Very high consumption!\n"
                "- You are likely paying a high electricity bill.\n"
                "- Use energy-saving lighting (LED) and appliances.\n"
                "- Reduce unnecessary usage during peak hours.\n"
                "- Consider investing in solar or alternative energy sources."
            )
        else:
            advice = (
                "🔥 Extreme consumption!\n"
                "- Immediate action required!\n"
                "- Unplug unused appliances and devices.\n"
                "- Limit simultaneous usage of heavy-power appliances.\n"
                "- Consider upgrading to energy-efficient models.\n"
                "- Check for electrical losses/leaks in your house."
            )

        st.session_state.prediction = total_units

        # Display results nicely
        st.success("⚡ Electricity Usag
