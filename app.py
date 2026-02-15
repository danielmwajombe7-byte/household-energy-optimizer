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
        df["Global_active_power"] = df["Kitchen_Power"] + df["Laundry_Power"] + df["Extra_Loss"] + df["Other_Appliances"]

    except Exception:
        # fallback dataset fully fixed
        df = pd.DataFrame({
            "Voltage": [220, 230, 210, 225, 240, 200, 215, 235, 220, 210],
            "Current": [5, 6, 4.5, 5.5, 6.5, 4, 4.8, 6, 5.2, 4.6],
            "Kitchen_Power": [1.2, 1.5, 1.0, 1.3, 1.8, 0.9, 1.1, 1.6, 1.4, 1.0],
            "Laundry_Power": [0.8, 1.0, 0.6, 0.9, 1.2, 0.5, 0.7, 1.1, 0.9, 0.6],
            "Extra_Loss": [0.3, 0.4, 0.2, 0.35, 0.5, 0.15, 0.25, 0.45, 0.3, 0.2],
            "Other_Appliances": [0.5, 0.6, 0.4, 0.5, 0.7, 0.3, 0.4, 0.6, 0.5, 0.4]
        })
        df["Global_active_power"] = df["Kitchen_Power"] + df["Laundry_Power"] + df["Extra_Loss"] + df["Other_Appliances"]

    return df

df = load_data()
FEATURES = ["Voltage", "Current", "Kitchen_Power", "Laundry_Power", "Extra_Loss", "Other_Appliances"]
TARGET = "Global_active_power"

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
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Visualization"])

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
    """, unsafe_allow_html=True)

    st.subheader("👤 User Information")
    st.session_state.user_info["name"] = st.text_input("Enter your name", value=st.session_state.user_info["name"])
    st.session_state.user_info["building"] = st.selectbox(
        "Select Building Type", ["House", "Office", "School", "Factory"],
        index=0 if st.session_state.user_info["building"] is None else ["House","Office","School","Factory"].index(st.session_state.user_info["building"])
    )

    if not st.session_state.user_info["name"] or not st.session_state.user_info["building"]:
        st.warning("Please enter your name and select building type to proceed.")
    else:
        st.success(f"Welcome {st.session_state.user_info['name']}! You selected {st.session_state.user_info['building']}.")

# ==========================
# PAGE 2: PREDICTION
# ==========================
elif page == "Prediction":
    st.subheader("👤 User Information (Required)")
    st.session_state.user_info["name"] = st.text_input("Enter your name", value=st.session_state.user_info["name"])
    st.session_state.user_info["building"] = st.selectbox(
        "Select Building Type", ["House", "Office", "School", "Factory"],
        index=0 if st.session_state.user_info["building"] is None else ["House","Office","School","Factory"].index(st.session_state.user_info["building"])
    )

    if not st.session_state.user_info["name"] or not st.session_state.user_info["building"]:
        st.warning("Please enter your name and select building type to predict.")
    else:
        st.markdown("### ⚡ Energy Prediction Inputs")
        col1, col2 = st.columns(2)
        user_input = {}
        with col1:
            user_input["Extra_Loss"] = st.number_input("Extra Power Loss", value=float(df["Extra_Loss"].mean()))
            user_input["Voltage"] = st.number_input("Electric Voltage (V)", value=float(df["Voltage"].mean()))
            user_input["Kitchen_Power"] = st.number_input("Kitchen Power Usage", value=float(df["Kitchen_Power"].mean()))
        with col2:
            user_input["Current"] = st.number_input("Current Intensity (A)", value=float(df["Current"].mean()))
            user_input["Laundry_Power"] = st.number_input("Laundry Power Usage", value=float(df["Laundry_Power"].mean()))
            user_input["Other_Appliances"] = st.number_input("Other Appliances Usage (TV, Fridge, Lights)", value=float(df["Other_Appliances"].mean()))

        if st.button("⚡ Predict Energy Consumption", use_container_width=True):
            input_df = pd.DataFrame([{f: user_input[f] for f in FEATURES}])
            st.session_state.prediction = model.predict(input_df)[0]
            avg = df[TARGET].mean()

            st.markdown(f"""
            <div style="text-align:center; background:#ecfeff; padding:25px; border-radius:15px;">
                <h2>⚡ Predicted Energy Consumption</h2>
                <h1 style="color:#0f766e;">{st.session_state.prediction:.2f} kW</h1>
                <p>User: <b>{st.session_state.user_info['name']}</b> | Building: <b>{st.session_state.user_info['building']}</b></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📌 Smart Advice")
            if st.session_state.prediction > avg * 1.3:
                st.error("⚠️ Very High Energy Consumption\n- Avoid using high-power devices simultaneously\n- Shift laundry to off-peak hours\n- Switch off unused devices")
            elif st.session_state.prediction > avg:
                st.warning("⚠️ Moderately High Consumption\n- Reduce kitchen appliance usage\n- Use energy-saving bulbs")
            else:
                st.success("✅ Energy Usage is Efficient\n- You are using electricity wisely")

# ==========================
# PAGE 3: VISUALIZATION
# ==========================
elif page == "Visualization":
    st.subheader("📊 Energy Consumption Comparison")
    graph_type = st.selectbox("Select Graph Type", ["Bar Chart", "Line Chart", "Area Chart", "Scatter Plot", "Pie Chart"])

    if st.session_state.prediction is None:
        st.warning("Please make a prediction first on the Prediction page.")
    else:
        last_pred = st.session_state.prediction
        plot_df = pd.DataFrame({
            "Category": ["Kitchen", "Laundry", "Other Appliances", "Extra Loss"],
            "Power (kW)": [
                user_input["Kitchen_Power"],
                user_input["Laundry_Power"],
                user_input["Other_Appliances"],
                user_input["Extra_Loss"]
            ]
        })
        # Total prediction comparison
        plot_df_total = pd.DataFrame({
            "Level": ["Average", "Your Usage", "Max"],
            "Power (kW)": [df[TARGET].mean(), last_pred, df[TARGET].max()]
        })

        if graph_type == "Bar Chart":
            fig = px.bar(plot_df, x="Category", y="Power (kW)", color="Category", template="plotly_white")
        elif graph_type == "Line Chart":
            fig = px.line(plot_df_total, x="Level", y="Power (kW)", markers=True, template="plotly_white")
        elif graph_type == "Area Chart":
            fig = px.area(plot_df_total, x="Level", y="Power (kW)", template="plotly_white")
        elif graph_type == "Scatter Plot":
            fig = px.scatter(plot_df_total, x="Level", y="Power (kW)", size="Power (kW)", color="Level", template="plotly_white")
        elif graph_type == "Pie Chart":
            fig = px.pie(plot_df, values="Power (kW)", names="Category", template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)
