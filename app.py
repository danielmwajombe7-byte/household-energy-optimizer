import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="⚡ Smart Energy Consumption Dashboard",
    page_icon="💡",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("tanzania_power_data.csv")

df = load_data()

# =====================================================
# FEATURES & TARGET
# =====================================================
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

# =====================================================
# TRAIN MODEL (Decision Tree)
# =====================================================
@st.cache_resource
def train_model(data):
    X = data[FEATURES]
    y = data[TARGET]
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# =====================================================
# SESSION STATE
# =====================================================
for key in ["prediction", "kitchen", "laundry", "other", "extra"]:
    if key not in st.session_state:
        st.session_state[key] = 0.0

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div style="
background: linear-gradient(90deg,#0f2027,#203a43,#2c5364);
padding:30px;
border-radius:15px;
text-align:center;
color:white;
margin-bottom:25px;">
<h1>⚡ Smart Energy Consumption Dashboard</h1>
<p>Predict • Visualize • Control Household Electricity Usage</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
page = st.sidebar.radio(
    "📌 Navigation",
    ["Dashboard", "Prediction", "Visualization"]
)

# =====================================================
# DASHBOARD PAGE
# =====================================================
if page == "Dashboard":
    st.subheader("📊 System Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Records", df.shape[0])
    c2.metric("📥 Inputs", len(FEATURES))
    c3.metric("🎯 Target", "Total Power (kW)")
    c4.metric("🤖 Model", "Decision Tree")

    st.info(
        "💡 This AI system helps households understand, predict, "
        "and optimize electricity consumption based on appliance usage."
    )

# =====================================================
# PREDICTION PAGE
# =====================================================
elif page == "Prediction":
    st.subheader("🧮 Enter Appliance Power Usage")

    col1, col2 = st.columns(2)

    with col1:
        kitchen = st.number_input("🍳 Kitchen Power (kW)", min_value=0.0, value=4.5)
        laundry = st.number_input("🧺 Laundry Power (kW)", min_value=0.0, value=6.0)
        other = st.number_input("💡 Other Usage (TV, Bulbs, Charging) (kW)", min_value=0.0, value=3.0)

    with col2:
        extra = st.number_input("🔥 Extra Power Loss (kW)", min_value=0.0, value=2.0)
        voltage = st.number_input("⚡ Voltage (V)", min_value=0.0, value=220.0)
        current = st.number_input("🔁 Current (A)", min_value=0.0, value=4.5)

    if st.button("🚀 Calculate Total Energy", use_container_width=True):
        st.session_state.kitchen = kitchen
        st.session_state.laundry = laundry
        st.session_state.other = other
        st.session_state.extra = extra

        total_energy = kitchen + laundry + other + extra
        st.session_state.prediction = total_energy

        st.success(f"✅ **Total Energy Used: {total_energy:.2f} kW**")

        if total_energy > 15:
            st.error("⚠️ High energy usage detected! Consider reducing heavy appliances.")
        elif total_energy > 8:
            st.warning("⚠️ Moderate energy usage. Try to optimize usage.")
        else:
            st.success("✅ Energy usage is efficient. Good job!")

# =====================================================
# VISUALIZATION PAGE
# =====================================================
elif page == "Visualization":
    st.subheader("📊 Energy Usage Visualization")

    if st.session_state.prediction == 0:
        st.warning("⚠️ Please calculate energy first from Prediction page.")
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

        chart_type = st.radio(
            "Select Chart Type",
            ["Bar Chart", "Pie Chart"],
            horizontal=True
        )

        fig, ax = plt.subplots()

        if chart_type == "Bar Chart":
            ax.bar(plot_df["Category"], plot_df["Power (kW)"])
            ax.set_ylabel("Power (kW)")
        else:
            ax.pie(
                plot_df["Power (kW)"],
                labels=plot_df["Category"],
                autopct="%1.1f%%",
                startangle=90
            )

        ax.set_title("Household Energy Distribution")
        st.pyplot(fig)

        st.info(f"🔋 **Total Energy Used: {st.session_state.prediction:.2f} kW**")
