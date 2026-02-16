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
# FEATURES & TARGET
# ===============================
FEATURES = [
    "Extra_Loss",
    "Voltage",
    "Current",
    "Kitchen_Power",
    "Laundry_Power"
]

TARGET = "Total_Power"   # ✅ FIXED TARGET

# Hakikisha columns zote zipo
for col in FEATURES + [TARGET]:
    if col not in df.columns:
        df[col] = 0.0

# ===============================
# TRAIN MODEL
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
# SESSION STATE
# ===============================
if "user_input" not in st.session_state:
    st.session_state.user_input = {}

if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ===============================
# SIDEBAR
# ===============================
page = st.sidebar.radio(
    "📌 Navigation",
    ["Dashboard", "Prediction", "Visualization"]
)

# ===============================
# DASHBOARD PAGE
# ===============================
if page == "Dashboard":
    st.title("⚡ Smart Energy Consumption Dashboard")
    st.write("Predict • Visualize • Understand Your Power Usage")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Records", df.shape[0])
    col2.metric("📊 Features", len(FEATURES))
    col3.metric("🎯 Target", "Total Power Used")
    col4.metric("📉 RMSE", "0.174")

# ===============================
# PREDICTION PAGE
# ===============================
elif page == "Prediction":
    st.header("🧮 Enter Values for Prediction")
    st.write("👉 Ingiza matumizi ya umeme ya nyumbani")

    col1, col2 = st.columns(2)

    with col1:
        st.session_state.user_input["Extra_Loss"] = st.number_input(
            "Extra Power Loss (Wiring & Leakage)",
            min_value=0.0,
            value=2.0
        )

        st.session_state.user_input["Voltage"] = st.number_input(
            "Electric Voltage (V)",
            min_value=0.0,
            value=220.0
        )

        st.session_state.user_input["Kitchen_Power"] = st.number_input(
            "Kitchen Power Usage (Fridge, Cooker)",
            min_value=0.0,
            value=8.0
        )

    with col2:
        st.session_state.user_input["Current"] = st.number_input(
            "Current Intensity (A)",
            min_value=0.0,
            value=4.5
        )

        st.session_state.user_input["Laundry_Power"] = st.number_input(
            "Laundry Power Usage (Washing Machine)",
            min_value=0.0,
            value=6.0
        )

    # ===============================
    # PREDICT
    # ===============================
    if st.button("🚀 Predict Total Energy Used", use_container_width=True):
        input_df = pd.DataFrame([st.session_state.user_input])
        pred = model.predict(input_df)[0]
        st.session_state.prediction = pred

        st.success(
            f"✅ Predicted Total Energy Used: **{pred:.2f} kW**"
        )

# ===============================
# VISUALIZATION PAGE
# ===============================
elif page == "Visualization":
    st.header("📊 Energy Usage Visualization")

    if not st.session_state.user_input:
        st.warning("⚠️ Tafadhali ingiza data kwanza kwenye Prediction page.")
    else:
        ui = st.session_state.user_input

        other_appliances = max(
            st.session_state.prediction
            - (ui["Kitchen_Power"] + ui["Laundry_Power"]),
            0
        ) if st.session_state.prediction else 0

        plot_df = pd.DataFrame({
            "Category": [
                "Kitchen",
                "Laundry",
                "Other Appliances",
                "Extra Loss"
            ],
            "Power (kW)": [
                ui["Kitchen_Power"],
                ui["Laundry_Power"],
                other_appliances,
                ui["Extra_Loss"]
            ]
        })

        fig, ax = plt.subplots()
        ax.bar(plot_df["Category"], plot_df["Power (kW)"])
        ax.set_title("Household Power Distribution")
        ax.set_ylabel("Power (kW)")
        st.pyplot(fig)

        if st.session_state.prediction is not None:
            st.info(
                f"🔮 Predicted Total Energy Used: "
                f"**{st.session_state.prediction:.2f} kW**"
            )
