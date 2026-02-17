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
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("tanzania_power_data.csv")

df = load_data()

# ===============================
# FEATURES (FOR ML – OPTIONAL)
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

# Ensure columns exist
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
    st.title("⚡ Smart Energy Consumption Dashboard")
    st.write("Predict • Visualize • Understand Your Power Usage")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Records", df.shape[0])
    c2.metric("📊 Features", len(FEATURES))
    c3.metric("🎯 Target", "Total Energy Used")
    c4.metric("📉 Model", "Decision Tree")

# ===============================
# PREDICTION
# ===============================
elif page == "Prediction":
    st.header("🧮 Enter Values for Prediction")

    col1, col2 = st.columns(2)

    with col1:
        kitchen = st.number_input(
            "Kitchen Power Usage (kW)", 0.0, value=4.5
        )
        laundry = st.number_input(
            "Laundry Power Usage (kW)", 0.0, value=6.0
        )
        other = st.number_input(
            "Other Use (Bulbs, TV, Charging) (kW)", 0.0, value=3.0
        )

    with col2:
        extra = st.number_input(
            "Extra Power Loss (kW)", 0.0, value=2.0
        )
        voltage = st.number_input(
            "Electric Voltage (V)", 0.0, value=220.0
        )
        current = st.number_input(
            "Current Intensity (A)", 0.0, value=4.5
        )

    if st.button("🚀 Calculate Total Energy Used", use_container_width=True):
        st.session_state.kitchen = kitchen
        st.session_state.laundry = laundry
        st.session_state.other = other
        st.session_state.extra = extra

        total_energy = kitchen + laundry + other + extra
        st.session_state.prediction = total_energy

        st.success(
            f"✅ Predicted Total Energy Used: **{total_energy:.2f} kW**"
        )

# ===============================
# VISUALIZATION
# ===============================
elif page == "Visualization":
    st.header("📊 Energy Usage Visualization")

    if st.session_state.prediction is None:
        st.warning("⚠️ Please calculate energy first.")
    else:
        plot_df = pd.DataFrame({
            "Category": ["Kitchen", "Laundry", "Other Use", "Extra Loss"],
            "Power (kW)": [
                st.session_state.kitchen,
                st.session_state.laundry,
                st.session_state.other,
                st.session_state.extra
            ]
        })

        fig, ax = plt.subplots()
        ax.bar(plot_df["Category"], plot_df["Power (kW)"])
        ax.set_title("Household Energy Distribution")
        ax.set_ylabel("Power (kW)")

        st.pyplot(fig)

        st.info(
            f"🔋 Total Energy Used: **{st.session_state.prediction:.2f} kW**"
        )
