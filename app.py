import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Energy Consumption",
    page_icon="💡",
    layout="wide"
)

# =====================================================
# NAVIGATION FUNCTION
# =====================================================
def go_to(page):
    st.session_state.page = page
    st.rerun()

if "page" not in st.session_state:
    st.session_state.page = "welcome"

# =====================================================
# LOAD DATA
# =====================================================
DATA_PATH = "tanzania_power_data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=";", engine="python")
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

# =====================================================
# MODEL
# =====================================================
target = "Global_active_power"
features = df_num.columns.drop(target)

X = df_num[features]
y = df_num[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42
)
model.fit(X_train, y_train)

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
    font-family: 'Segoe UI', sans-serif;
">
<div style="font-size:60px; color:#facc15;">💡</div>
<h1 style="color:white; margin:0;">Smart Energy Consumption Dashboard</h1>
<p style="color:#d1d5db; font-size:16px;">Predict • Monitor • Optimize Electricity Usage</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# WELCOME PAGE
# =====================================================
if st.session_state.page == "welcome":
    st.markdown("### 👋 Welcome")
    st.info("This system helps you understand and reduce electricity consumption.", icon="💡")
    user = st.text_input("👤 Enter your name")
    building = st.selectbox(
        "🏢 Select Building Type",
        ["House", "Office", "School", "Factory"]
    )
    if st.button("➡️ Proceed to Prediction", use_container_width=True):
        if user.strip() == "":
            st.warning("Please enter your name")
        else:
            st.session_state.user = user
            st.session_state.building = building
            go_to("prediction")

# =====================================================
# PREDICTION PAGE
# =====================================================
elif st.session_state.page == "prediction":
    st.markdown(f"""
    <div style="background:#e0f2fe;padding:15px;border-radius:10px;margin-bottom:15px;">
    👤 <b>{st.session_state.user}</b> | 🏠 <b>{st.session_state.building}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h3 style="text-align:center;color:#1e3a8a;">
    ENTER VALUES FOR PREDICTION
    </h3>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    user_input = {}

    with col1:
        user_input["Global_active_power"] = st.number_input(
            "🔌 Total Power Usage (kW)", value=float(y.mean())
        )
        user_input["Voltage"] = st.number_input(
            "⚡ Electricity Voltage (V)", value=float(df_num["Voltage"].mean())
        )
        user_input["Sub_metering_1"] = st.number_input(
            "🍳 Kitchen Power Usage", value=float(df_num["Sub_metering_1"].mean())
        )

    with col2:
        user_input["Global_reactive_power"] = st.number_input(
            "🔥 Extra Power Loss", value=float(df_num["Global_reactive_power"].mean())
        )
        user_input["Global_intensity"] = st.number_input(
            "🔁 Current Intensity (A)", value=float(df_num["Global_intensity"].mean())
        )
        user_input["Sub_metering_2"] = st.number_input(
            "🧺 Laundry Power Usage", value=float(df_num["Sub_metering_2"].mean())
        )

    st.markdown("<br>", unsafe_allow_html=True)
    colA, colB = st.columns(2)

    with colA:
        if st.button("⚡ Predict Energy Consumption", use_container_width=True):
            full = {f: user_input.get(f, float(df_num[f].mean())) for f in features}
            input_df = pd.DataFrame([full])
            st.session_state.prediction = model.predict(input_df)[0]
            go_to("result")

    with colB:
        if st.button("🏠 Go Home", use_container_width=True):
            go_to("welcome")

# =====================================================
# RESULT PAGE
# =====================================================
elif st.session_state.page == "result":
    pred = st.session_state.prediction
    avg = y.mean()

    st.markdown(f"""
    <div style="text-align:center;
        background: linear-gradient(90deg,#0f1727,#203a43,#2c5364);
        padding:25px;
        border-radius:15px;margin-bottom:15px;">
    <div style="font-size:50px; color:#facc15;">💡</div>
    <h2 style="color:#fcd34d;">Predicted Energy Consumption</h2>
    <h1 style="color:#22c55e;">{pred:.2f} kW</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📌 Smart Advice")
    if pred > avg * 1.3:
        st.error("""
        ⚠️ **Very High Consumption in Short Time**
        - Multiple high-power appliances used simultaneously  
        - Possible energy wastage  
        **Advice:** Shift some appliances to off-peak hours and switch off unused devices.
        """)
    elif pred > avg:
        st.warning("""
        ⚠️ **Moderately High Consumption**
        - Reduce heavy appliance usage  
        - Use energy-saving devices
        """)
    else:
        st.success("""
        ✅ **Energy usage is efficient**
        - Continue good energy habits
        """)

    st.divider()

    st.markdown("### 📊 Energy Usage Visualization")
    show_graph = st.checkbox("Show Energy Graph")
    if show_graph:
        graph_type = st.radio(
            "Select Graph Type",
            ["Bar Chart", "Line Chart", "Scatter Chart"],
            horizontal=True
        )
        df_plot = pd.DataFrame({
            "Level": ["Low", "Average", "Your Usage", "High"],
            "Power (kW)": [y.min(), avg, pred, y.max()]
        })
        if graph_type == "Bar Chart":
            fig = px.bar(df_plot, x="Level", y="Power (kW)", color="Level", template="plotly_white")
        elif graph_type == "Line Chart":
            fig = px.line(df_plot, x="Level", y="Power (kW)", markers=True, template="plotly_white")
        else:
            fig = px.scatter(df_plot, x="Level", y="Power (kW)", size="Power (kW)", color="Level", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    colX, colY = st.columns(2)
    with colX:
        if st.button("🔁 New Prediction", use_container_width=True):
            go_to("prediction")
    with colY:
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            go_to("welcome")
