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
    page_title="⚡ Smart Energy Prediction Dashboard",
    page_icon="⚡",
    layout="wide"
)

# =====================================================
# SESSION STATE SETUP
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "prediction_value" not in st.session_state:
    st.session_state.prediction_value = None

# =====================================================
# LOAD DATA
# =====================================================
DATA_PATH = "tanzania_power_data.csv"
if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset not found. Make sure CSV is in same folder as app.py")
    st.stop()

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv(DATA_PATH, sep=";", engine="python")
    if "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
        df.drop(columns=["Date","Time"], inplace=True)
    for col in df.columns:
        if col != "Datetime":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    return df

df = load_and_clean_data()
df_numeric = df.select_dtypes(include="number")
if df_numeric.shape[1] < 2:
    st.error("❌ Dataset must contain at least 2 numeric columns")
    st.stop()

# =====================================================
# MODEL TRAINING
# =====================================================
target_column = "Global_active_power"
feature_columns = df_numeric.columns.drop(target_column)
X = df_numeric[feature_columns]
y = df_numeric[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42)
model.fit(X_train, y_train)
rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

friendly_names = {
    "Global_active_power": "Total Power Used (kW)",
    "Global_reactive_power": "Extra Power Loss",
    "Voltage": "Electric Voltage (V)",
    "Global_intensity": "Current Intensity (A)",
    "Sub_metering_1": "Kitchen Power Usage",
    "Sub_metering_2": "Laundry Power Usage",
    "Sub_metering_3": "Water Heater / AC Usage"
}

# =====================================================
# LOGIN / WELCOME PAGE
# =====================================================
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;color:#ff6600;font-family:verdana;'>⚡ Welcome to Smart Energy Dashboard</h1>", unsafe_allow_html=True)
    st.image("electricity_logo.png", width=100)  # Add a more advanced logo in the folder
    st.write("Please enter your details to continue:")
    
    user_name = st.text_input("Your Name")
    user_email = st.text_input("Your Email")
    
    if st.button("Login"):
        if user_name.strip() == "":
            st.warning("Please enter your name!")
        else:
            st.session_state.logged_in = True
            st.session_state.user_name = user_name
            st.session_state.user_email = user_email
            st.rerun()

# =====================================================
# PREDICTION FORM PAGE
# =====================================================
else:
    st.markdown(f"<h2 style='text-align:center;color:#1a75ff;'>Welcome, {st.session_state.user_name}!</h2>", unsafe_allow_html=True)
    st.divider()
    
    st.subheader("🏠 Select Building Type")
    building_type = st.selectbox(
        "Choose your building type:",
        options=["House","Office","School","Factory"]
    )
    
    st.markdown("<h3 style='text-align:center;color:#ff6600;'>ENTER VALUES FOR PREDICTION</h3>", unsafe_allow_html=True)
    
    # Define feature mapping per building type
    feature_mapping = {
        "House": [f for f in ["Global_active_power","Global_reactive_power","Voltage","Global_intensity","Sub_metering_1","Sub_metering_2"] if f in df_numeric.columns],
        "Office": [f for f in ["Global_active_power","Global_reactive_power","Voltage","Global_intensity"] if f in df_numeric.columns],
        "School": [f for f in ["Global_active_power","Voltage","Sub_metering_1","Sub_metering_3"] if f in df_numeric.columns],
        "Factory": [f for f in ["Global_active_power","Global_reactive_power","Voltage","Global_intensity","Sub_metering_3"] if f in df_numeric.columns]
    }
    
    selected_features = feature_mapping[building_type]
    
    # ================= LEFT & RIGHT GRID INPUTS =================
    input_cols = st.columns(2)
    user_input = {}
    
    def get_value(col):
        return st.number_input(friendly_names.get(col,col), value=float(df_numeric[col].mean()), step=0.1)
    
    # Row 1
    if "Global_active_power" in selected_features:
        user_input["Global_active_power"] = input_cols[0].number_input(friendly_names["Global_active_power"], value=float(df_numeric["Global_active_power"].mean()), step=0.1)
    if "Global_reactive_power" in selected_features:
        user_input["Global_reactive_power"] = input_cols[1].number_input(friendly_names["Global_reactive_power"], value=float(df_numeric["Global_reactive_power"].mean()), step=0.1)
    
    # Row 2
    if "Voltage" in selected_features:
        user_input["Voltage"] = input_cols[0].number_input(friendly_names["Voltage"], value=float(df_numeric["Voltage"].mean()), step=0.1)
    if "Global_intensity" in selected_features:
        user_input["Global_intensity"] = input_cols[1].number_input(friendly_names["Global_intensity"], value=float(df_numeric["Global_intensity"].mean()), step=0.1)
    
    # Row 3
    if "Sub_metering_1" in selected_features:
        user_input["Sub_metering_1"] = input_cols[0].number_input(friendly_names["Sub_metering_1"], value=float(df_numeric["Sub_metering_1"].mean()), step=0.1)
    if "Sub_metering_2" in selected_features:
        user_input["Sub_metering_2"] = input_cols[1].number_input(friendly_names["Sub_metering_2"], value=float(df_numeric["Sub_metering_2"].mean()), step=0.1)
    
    st.markdown("<h3 style='text-align:center;'>🚀 Predict Energy Consumption</h3>", unsafe_allow_html=True)
    if st.button("Predict"):
        # Prepare input_df for model
        input_data = {}
        for col in feature_columns:
            if col in user_input:
                input_data[col] = user_input[col]
            else:
                input_data[col] = float(df_numeric[col].mean())
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.session_state.prediction_done = True
        st.session_state.prediction_value = prediction
        st.rerun()
    
    # ================= SHOW PREDICTION RESULTS =================
    if st.session_state.prediction_done:
        st.divider()
        st.markdown(f"<h2 style='text-align:center;color:#ff6600;'>⚡ Predicted Consumption: {st.session_state.prediction_value:.3f}</h2>", unsafe_allow_html=True)
        if st.session_state.prediction_value > y.mean():
            st.warning("⚠️ High energy usage detected. Reduce heavy appliances.")
        else:
            st.success("✅ Energy usage is normal.")
        
        if st.checkbox("Show Graph"):
            zoom_df = pd.DataFrame({
                "Stage": ["Low","Average","Your Prediction","High"],
                "Consumption": [y.min(), y.mean(), st.session_state.prediction_value, y.max()]
            })
            fig = px.line(
                zoom_df,
                x="Stage",
                y="Consumption",
                markers=True,
                title="🔍 Energy Usage Comparison",
                line_shape="spline",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Go Home"):
            st.session_state.prediction_done = False
            st.rerun()
