import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="⚡ Smart Energy Consumption Dashboard",
    layout="wide"
)

# ===============================
# LOAD MODEL & DATA
# ===============================
model = joblib.load("model.pkl")
df = pd.read_csv("tanzania_power_data.csv")

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
    col2.metric("📊 Features", df.shape[1] - 1)
    col3.metric("🎯 Target", "Laundry Power Usage")
    col4.metric("📉 RMSE", "0.174")

# ===============================
# PREDICTION PAGE
# ===============================
elif page == "Prediction":
    st.header("🧮 Enter Values for Prediction")
    st.write("👉 Ingiza matumizi ya umeme ya nyumbani")

    col1, col2 = st.columns(2)

    with col1:
        st.session_state.user_input["Total_Power"] = st.number_input(
            "Total Power Used (kW)",
            min_value=0.0,
            value=28.0
        )

        st.session_state.user_input["Extra_Loss"] = st.number_input(
            "Extra Power Loss (Wiring & Leakage)",
            min_value=0.0,
            value=10.0
        )

        st.session_state.user_input["Voltage"] = st.number_input(
            "Electric Voltage (V)",
            min_value=0.0,
            value=220.0
        )

    with col2:
        st.session_state.user_input["Current"] = st.number_input(
            "Current Intensity (A)",
            min_value=0.0,
            value=4.5
        )

        st.session_state.user_input["Kitchen_Power"] = st.number_input(
            "Kitchen Power Usage (Fridge, Cooker)",
            min_value=0.0,
            value=8.0
        )

        st.session_state.user_input["Laundry_Power"] = st.number_input(
            "Laundry Power Usage (Washing Machine)",
            min_value=0.0,
            value=8.0
        )

    # ===============================
    # PREDICT BUTTON (ONE ONLY)
    # ===============================
    if st.button("🚀 Predict Energy Consumption", use_container_width=True):
        input_df = pd.DataFrame([st.session_state.user_input])
        prediction = model.predict(input_df)[0]
        st.session_state.prediction = prediction

        st.success(f"✅ Predicted Laundry Power Usage: **{prediction:.2f} kW**")

# ===============================
# VISUALIZATION PAGE
# ===============================
elif page == "Visualization":
    st.header("📊 Energy Usage Visualization")

    if not st.session_state.user_input:
        st.warning("⚠️ Tafadhali ingiza data kwanza kwenye Prediction page.")
    else:
        ui = st.session_state.user_input

        # ===============================
        # BAR CHART
        # ===============================
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
                ui["Total_Power"] - (ui["Kitchen_Power"] + ui["Laundry_Power"]),
                ui["Extra_Loss"]
            ]
        })

        fig, ax = plt.subplots()
        ax.bar(plot_df["Category"], plot_df["Power (kW)"])
        ax.set_ylabel("Power (kW)")
        ax.set_title("Household Power Distribution")

        st.pyplot(fig)

        # ===============================
        # SHOW PREDICTION
        # ===============================
        if st.session_state.prediction is not None:
            st.info(
                f"🔮 Predicted Laundry Power Usage: "
                f"**{st.session_state.prediction:.2f} kW**"
            )
