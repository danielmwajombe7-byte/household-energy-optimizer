import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Energy Consumption",
    page_icon="⚡",
    layout="wide"
)

# =====================================================
# LOAD TRAINED MODEL (IMPORTANT FOR TEST 2)
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# =====================================================
# LOAD DATA (FOR MEAN / VISUALIZATION ONLY)
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("tanzania_power_data.csv", sep=";", engine="python")

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
df_num = df.select_dtypes(include="number")

TARGET = "Global_active_power"
FEATURES = df_num.columns.drop(TARGET)

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
# USER INPUT SECTION
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
    user_input["Global_reactive_power"] = st.number_input(
        "Extra Power Loss",
        value=float(df_num["Global_reactive_power"].mean())
    )
    user_input["Voltage"] = st.number_input(
        "Electric Voltage (V)",
        value=float(df_num["Voltage"].mean())
    )
    user_input["Sub_metering_1"] = st.number_input(
        "Kitchen Power Usage",
        value=float(df_num["Sub_metering_1"].mean())
    )

with col2:
    user_input["Global_intensity"] = st.number_input(
        "Current Intensity (A)",
        value=float(df_num["Global_intensity"].mean())
    )
    user_input["Sub_metering_2"] = st.number_input(
        "Laundry Power Usage",
        value=float(df_num["Sub_metering_2"].mean())
    )
    user_input["Sub_metering_3"] = st.number_input(
        "Other Appliances Usage",
        value=float(df_num["Sub_metering_3"].mean())
    )

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# PREDICTION
# =====================================================
if st.button("⚡ Predict Energy Consumption", use_container_width=True):

    if name.strip() == "":
        st.warning("Please enter your name")
    else:
        # Prepare full feature input
        full_input = {f: user_input.get(f, float(df_num[f].mean())) for f in FEATURES}
        input_df = pd.DataFrame([full_input])

        prediction = model.predict(input_df)[0]
        avg = df_num[TARGET].mean()

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
        # ADVICE SECTION
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
                df_num[TARGET].min(),
                avg,
                prediction,
                df_num[TARGET].max()
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
