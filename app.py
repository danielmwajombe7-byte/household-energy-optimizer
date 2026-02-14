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
# TITLE
# =====================================================
st.markdown("""
<h1 style='text-align:center;'>⚡ Smart Energy Consumption Dashboard</h1>
<p style='text-align:center;color:gray;'>
Predict • Visualize • Understand Your Power Usage
</p>
""", unsafe_allow_html=True)

st.divider()

# =====================================================
# LOAD & CLEAN DATA (ONCE)
# =====================================================
DATA_PATH = "tanzania_power_data.csv"

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset not found. Make sure CSV is in same folder as app.py")
    st.stop()

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv(DATA_PATH, sep=";", engine="python")

    # Combine Date & Time
    if "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"],
            dayfirst=True,
            errors="coerce"
        )
        df.drop(columns=["Date", "Time"], inplace=True)

    # Convert all to numeric except Datetime
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
# FEATURES & TARGET
# =====================================================
target_column = "Global_active_power"
feature_columns = df_numeric.columns.drop(target_column)

X = df_numeric[feature_columns]
y = df_numeric[target_column]

# =====================================================
# TRAIN MODEL
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=80,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)
rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

# =====================================================
# FRIENDLY COLUMN NAMES
# =====================================================
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
# TWO COLUMN LAYOUT
# =====================================================
left, right = st.columns([1, 1.4])

# ================= LEFT: BUILDING TYPE & INPUTS =================
# ================= LEFT: BUILDING TYPE & INPUTS =================
with left:
    st.subheader("🏠 Select Building Type")
    building_type = st.selectbox(
        "Choose your building type:",
        options=["House", "Office", "School", "Factory"]
    )
    st.markdown(f"**You selected:** {building_type}")
    st.divider()
    
    st.subheader("🧮 Enter Values for Prediction")
    user_input = {}

    # Feature mapping per building type
    feature_mapping = {
        "House": ["Global_active_power", "Global_reactive_power",
                  "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2"],
        "Office": ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"],
        "School": ["Global_active_power", "Voltage", "Sub_metering_1", "Sub_metering_3"],
        "Factory": ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_3"]
    }

    # Chuja tu features ambazo dataset ina
    selected_features = [f for f in feature_mapping[building_type] if f in df_numeric.columns]

    # Ikiwa kuna features ambazo hazipo, toa warning
    missing_features = set(feature_mapping[building_type]) - set(selected_features)
    if missing_features:
        st.warning(f"⚠️ The following features are missing in your dataset and will be skipped: {', '.join(missing_features)}")

    # Create inputs dynamically kwa features zilizopo tu
    for col in selected_features:
        label = friendly_names.get(col, col)
        user_input[col] = st.number_input(
            label,
            value=float(df_numeric[col].mean()),
            step=0.1
        )

    predict_btn = st.button(
        "🚀 Predict Energy Consumption",
        use_container_width=True,
        key="predict_btn"
    )

    # Option to show/hide metrics
    show_metrics = st.checkbox("Show Dataset & Model Metrics")

    if show_metrics:
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📄 Records", len(df))
        c2.metric("📊 Features", len(feature_columns))
        c3.metric("🎯 Target", friendly_names.get(target_column, target_column))
        c4.metric("📉 RMSE", f"{rmse:.3f}")


# ================= RIGHT: RESULTS =================
with right:
    st.subheader("📈 Prediction Results")

    if predict_btn:
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]

        st.markdown(
            f"<h2 style='color:#ff7f0e;'>⚡ Predicted Consumption</h2>"
            f"<h1>{prediction:.3f}</h1>",
            unsafe_allow_html=True
        )

        if prediction > y.mean():
            st.warning("⚠️ High energy usage detected. Reduce heavy appliances.")
        else:
            st.success("✅ Energy usage is normal.")

        zoom_df = pd.DataFrame({
            "Stage": ["Low", "Average", "Your Prediction", "High"],
            "Consumption": [y.min(), y.mean(), prediction, y.max()]
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

    else:
        st.info("👈 Fill values and click Predict")

# =====================================================
# DATA PREVIEW
# =====================================================
st.divider()
with st.expander("🔎 View Dataset Preview"):
    st.dataframe(df.head(50))

