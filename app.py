import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="⚡ Smart Energy Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ================== LOAD & CLEAN DATA ==================
DATA_PATH = "tanzania_power_data.csv"

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv(DATA_PATH, sep=";", engine="python")
    if "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
        df.drop(columns=["Date", "Time"], inplace=True)
    for col in df.columns:
        if col != "Datetime":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    return df

df = load_and_clean_data()
df_numeric = df.select_dtypes(include="number")

target_column = "Global_active_power"
feature_columns = df_numeric.columns.drop(target_column)

X = df_numeric[feature_columns]
y = df_numeric[target_column]

model = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

# ================== FRIENDLY COLUMN NAMES ==================
friendly_names = {
    "Global_active_power": "Total Power Used (kW)",
    "Global_reactive_power": "Extra Power Loss",
    "Voltage": "Electric Voltage (V)",
    "Global_intensity": "Current Intensity (A)",
    "Sub_metering_1": "Kitchen Power Usage",
    "Sub_metering_2": "Laundry Power Usage",
    "Sub_metering_3": "Water Heater / AC Usage"
}

# ================== LANDING / LOGIN PAGE ==================
if "login_done" not in st.session_state:
    st.markdown("""
    <div style="text-align:center; padding: 40px; background-color: #0a9396; border-radius:15px;">
        <h1 style="font-family: 'Helvetica', sans-serif; color:#fff; font-size: 50px;">⚡ Smart Energy Dashboard</h1>
        <p style="font-size:22px; color:#e9d8a6;">Predict • Visualize • Understand Your Power Usage</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("👤 Enter Your Details")
    user_name = st.text_input("Full Name", placeholder="John Doe")
    user_email = st.text_input("Email Address", placeholder="john@example.com")
    
    st.subheader("🏠 Select Your Building Type")
    building_type = st.selectbox("Choose your building type:", options=["House", "Office", "School", "Factory"])
    
    if st.button("➡️ Proceed to Prediction Form"):
        if not user_name.strip():
            st.error("❌ Please enter your name to continue.")
        else:
            st.session_state['login_done'] = True
            st.session_state['user_name'] = user_name
            st.session_state['user_email'] = user_email
            st.session_state['building_type'] = building_type
            st.experimental_rerun()

# ================== PREDICTION PAGE ==================
if "login_done" in st.session_state:
    st.markdown(f"<h2 style='color:#0a9396;'>Welcome, {st.session_state['user_name']}!</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:gray;'>Building Type: {st.session_state['building_type']}</p>", unsafe_allow_html=True)
    
    left, right = st.columns([1, 1.4])
    
    # ===== LEFT: INPUTS =====
    with left:
        st.subheader("🧮 Enter Values for Prediction")
        user_input = {}
        
        feature_mapping = {
            "House": [f for f in ["Global_active_power", "Global_reactive_power",
                                  "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2"] if f in df_numeric.columns],
            "Office": [f for f in ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"] if f in df_numeric.columns],
            "School": [f for f in ["Global_active_power", "Voltage", "Sub_metering_1", "Sub_metering_3"] if f in df_numeric.columns],
            "Factory": [f for f in ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_3"] if f in df_numeric.columns]
        }
        selected_features = feature_mapping[st.session_state['building_type']]
        
        for col in selected_features:
            label = friendly_names.get(col, col)
            user_input[col] = st.number_input(label, value=float(df_numeric[col].mean()), step=0.1)
        
        predict_btn = st.button("🚀 Predict Energy Consumption", use_container_width=True)
        show_metrics = st.checkbox("Show Dataset & Model Metrics")
        if show_metrics:
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📄 Records", len(df))
            c2.metric("📊 Features", len(feature_columns))
            c3.metric("🎯 Target", friendly_names.get(target_column, target_column))
            c4.metric("📉 RMSE", f"{rmse:.3f}")

    # ===== RIGHT: RESULTS =====
    with right:
        st.subheader("📈 Prediction Results")
        if predict_btn:
            input_data = {}
            for col in feature_columns:
                if col in user_input:
                    input_data[col] = user_input[col]
                else:
                    input_data[col] = float(df_numeric[col].mean())
            input_df = pd.DataFrame([input_data])
            
            prediction = model.predict(input_df)[0]
            
            st.markdown(f"<h2 style='color:#ff7f0e;'>⚡ Predicted Consumption</h2>"
                        f"<h1>{prediction:.3f}</h1>", unsafe_allow_html=True)
            
            if prediction > y.mean():
                st.warning("⚠️ High energy usage detected. Reduce heavy appliances.")
            else:
                st.success("✅ Energy usage is normal.")
            
            zoom_df = pd.DataFrame({
                "Stage": ["Low", "Average", "Your Prediction", "High"],
                "Consumption": [y.min(), y.mean(), prediction, y.max()]
            })
            fig = px.line(zoom_df, x="Stage", y="Consumption", markers=True,
                          title="🔍 Energy Usage Comparison", line_shape="spline",
                          template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Fill values and click Predict")
    
    st.divider()
    with st.expander("🔎 View Dataset Preview"):
        st.dataframe(df.head(50))
