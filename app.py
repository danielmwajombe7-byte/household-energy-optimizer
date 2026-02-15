import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Energy Consumption",
    page_icon="⚡",
    layout="wide"
)

# =====================================================
# MINI BUILT-IN DATASET
# =====================================================
@st.cache_data
def load_data():
    df = pd.DataFrame({
        "Voltage": [220, 230, 210, 225, 240, 200, 215, 235, 220, 210],
        "Current": [5, 6, 4.5, 5.5, 6.5, 4, 4.8, 6, 5.2, 4.6],
        "Kitchen_Power": [1.2, 1.5, 1.0, 1.3, 1.8, 0.9, 1.1, 1.6, 1.4, 1.0],
        "Laundry_Power": [0.8, 1.0, 0.6, 0.9, 1.2, 0.5, 0.7, 1.1, 0.9, 0.6],
        "Extra_Loss": [0.3, 0.4, 0.2, 0.35, 0.5, 0.15, 0.25, 0.45, 0.3, 0.2]
    })
    df["Global_active_power"] = df["Kitchen_Power"] + df["Laundry_Power"] + df["Extra_Loss"]
    return df

df = load_data()
FEATURES = ["Voltage", "Current", "Kitchen_Power", "Laundry_Power", "Extra_Loss"]
TARGET = "Global_active_power"

# =====================================================
# TRAIN MODEL
# =====================================================
@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

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
<div style="font-size:55px;color:#facc15;">💡</div>
<h1 style="color:white;">Smart Energy Consumption AI App</h1>
<p style="color:#d1d5db;">
Machine Learning Based Energy Prediction
</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# USER INPUT
# =====================================================
st.subheader("👤 User Information")

name = st.text_input("Enter your name")
building = st.selectbox(
    "Select Building Type",
    ["House", "Office", "School", "Factory"]
)

st.divider()

st.markdown(
    "<h3 style='text-align:center;color:#1e3a8a;'>ENTER VALUES FOR PREDICTION</h3>",
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)
user_input = {}

with col1:
    user_input["Extra_Loss"] = st.number_input(
        "Extra Power Loss",
        value=float(df["Extra_Loss"].mean())
    )
    user_input["Voltage"] = st.number_input(
        "Electric Voltage (V)",
        value=float(df["Voltage"].mean())
    )
    user_input["Kitchen_Power"] = st.number_input(
        "Kitchen Power Usage",
        value=float(df["Kitchen_Power"].mean())
    )

with col2:
    user_input["Current"] = st.number_input(
        "Current Intensity (A)",
        value=float(df["Current"].mean())
    )
    user_input["Laundry_Power"] = st.number_input(
        "Laundry Power Usage",
        value=float(df["Laundry_Power"].mean())
    )

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# PREDICTION
# =====================================================
if st.button("⚡ Predict Energy Consumption", use_container_width=True):

    if name.strip() == "":
        st.warning("Please enter your name")
    else:
        # Ensure input columns match FEATURES exactly
        input_df = pd.DataFrame([{f: user_input[f] for f in FEATURES}])
        prediction = model.predict(input_df)[0]
        avg = df[TARGET].mean()

        st.markdown(f"""
        <div style="text-align:center;
            background:#ecfeff;
            padding:25px;
            border-radius:15px;">
        <h2>⚡ Predicted Energy Consumption</h2>
        <h1 style="color:#0f766e;">{prediction:.2f} kW</h1>
        <p>User: <b>{name}</b> | Building: <b>{building}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # =====================================================
        # ADVICE
        # =====================================================
        st.markdown("### 📌 Smart Advice")

        if prediction > avg * 1.3:
            st.error("""
            ⚠️ **Very High Energy Consumption**
            
            Possible causes:
            - Many high-power appliances used at the same time
            
            Advice:
            - Avoid using cooker, iron and washing machine together  
            - Shift laundry to off-peak hours  
            - Switch off unused devices
            """)
        elif prediction > avg:
            st.warning("""
            ⚠️ **Moderately High Consumption**
            
            Advice:
            - Reduce kitchen appliance usage  
            - Use energy-saving bulbs
            """)
        else:
            st.success("""
            ✅ **Energy Usage is Efficient**
            
            - You are using electricity wisely  
            - Keep it up 💚
            """)

        # =====================================================
        # VISUALIZATION
        # =====================================================
        st.markdown("### 📊 Energy Consumption Comparison")

        graph_type = st.radio(
            "Select Graph Type",
            ["Bar Chart", "Line Chart"],
            horizontal=True
        )

        plot_df = pd.DataFrame({
            "Level": ["Low", "Average", "Your Usage", "High"],
            "Power (kW)": [
                df[TARGET].min(),
                avg,
                prediction,
                df[TARGET].max()
            ]
        })

        if graph_type == "Bar Chart":
            fig = px.bar(
                plot_df,
                x="Level",
                y="Power (kW)",
                color="Level",
                template="plotly_white"
            )
        else:
            fig = px.line(
                plot_df,
                x="Level",
                y="Power (kW)",
                markers=True,
                template="plotly_white"
            )

        st.plotly_chart(fig, use_container_width=True)
