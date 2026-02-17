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
# ELECTRICITY PRICE (TZS per kWh)
# ===============================
PRICE_PER_UNIT = 350  # 1 unit = 1 kWh

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("tanzania_power_data.csv")
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

# ===============================
# FEATURES (FOR ML – DEMO)
# ===============================
FEATURES = [
    "Kitchen_Power",
    "Laundry_Power",
    "Other_Use",
    "Extra_Loss",
    "Voltage",
    "Current"
]

TARGET = "Total_Power"

# Ensure required columns exist
for col in FEATURES + [TARGET]:
    if col not in df.columns:
        df[col] = 0.0

# ===============================
# TRAIN MODEL (DEMO PURPOSE)
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
# SESSION STATE INIT
# ===============================
defaults = {
    "prediction": None,
    "duration": 1.0,
    "kitchen": 0.0,
    "laundry": 0.0,
    "other": 0.0,
    "extra": 0.0
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===============================
# SIDEBAR
# ===============================
page = st.sidebar.radio(
    "📌 Navigation",
    ["Dashboard", "Prediction", "Visualization"]
)

# ===============================
# DASHBOARD
# ===============================
if page == "Dashboard":
    st.markdown("""
    <div style="background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);
    padding:25px;border-radius:15px;color:white;text-align:center;">
        <div style="font-size:55px;">💡</div>
        <h1>Smart Energy Consumption Dashboard</h1>
        <p>Predict • Measure • Understand Electricity Usage</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(f"""
    <div style="background:#4ade80;padding:20px;border-radius:10px;text-align:center;color:white;">
    <h3>📄 Records</h3>
    <p style="font-size:25px;">{df.shape[0]}</p>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div style="background:#60a5fa;padding:20px;border-radius:10px;text-align:center;color:white;">
    <h3>📊 Features</h3>
    <p style="font-size:25px;">{len(FEATURES)}</p>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
    <div style="background:#facc15;padding:20px;border-radius:10px;text-align:center;color:white;">
    <h3>🎯 Target</h3>
    <p style="font-size:25px;">Energy (kWh)</p>
    </div>
    """, unsafe_allow_html=True)

    c4.markdown(f"""
    <div style="background:#f472b6;padding:20px;border-radius:10px;text-align:center;color:white;">
    <h3>🤖 Model</h3>
    <p style="font-size:25px;">Decision Tree</p>
    </div>
    """, unsafe_allow_html=True)

    # Optional: Mini bar chart for feature demo
    st.subheader("Feature Power Example (kW)")
    demo_features = [4.5, 6.0, 3.0, 2.0]
    demo_names = ["Kitchen", "Laundry", "Other", "Extra Loss"]
    fig, ax = plt.subplots(figsize=(6,2))
    ax.barh(demo_names, demo_features, color="#60a5fa")
    ax.set_xlabel("Power (kW)")
    st.pyplot(fig)

# ===============================
# PREDICTION PAGE
# ===============================
elif page == "Prediction":
    st.header("🧮 Energy Prediction Input")

    col1, col2 = st.columns(2)

    with col1:
        kitchen = st.number_input("🍳 Kitchen Power (kW)", 0.0, value=4.5)
        laundry = st.number_input("🧺 Laundry Power (kW)", 0.0, value=6.0)
        other = st.number_input("💡 Other Usage (kW)", 0.0, value=3.0)

    with col2:
        extra = st.number_input("🔥 Extra Loss (kW)", 0.0, value=2.0)
        voltage = st.number_input("⚡ Voltage (V)", 0.0, value=220.0)
        current = st.number_input("🔁 Current (A)", 0.0, value=4.5)

    st.markdown("---")

    duration = st.slider(
        "⏱️ Duration of Usage (Hours)",
        min_value=0.5,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    if st.button("🚀 Calculate Energy Consumption", use_container_width=True):
        st.session_state.kitchen = kitchen
        st.session_state.laundry = laundry
        st.session_state.other = other
        st.session_state.extra = extra
        st.session_state.duration = duration

        # Total power and units
        feature_energy = {
            "Kitchen": kitchen * duration,
            "Laundry": laundry * duration,
            "Other Use": other * duration,
            "Extra Loss": extra * duration
        }
        total_units = sum(feature_energy.values())
        total_cost = total_units * PRICE_PER_UNIT

        # Identify features with high consumption (>30% of total)
        advice_list = []
        for feat, energy in feature_energy.items():
            contribution = (energy / total_units) * 100
            if contribution > 30:
                if feat == "Kitchen":
                    advice_list.append("⚠️ Kitchen consumes a lot! Cook efficiently, batch cooking, use lids.")
                elif feat == "Laundry":
                    advice_list.append("⚠️ Laundry consumes a lot! Wash full loads, avoid peak hours.")
                elif feat == "Other Use":
                    advice_list.append("⚠️ Other usage is high! Turn off devices when idle.")
                elif feat == "Extra Loss":
                    advice_list.append("⚠️ Extra losses are high! Check wiring and appliances.")

        if not advice_list:
            advice_list.append("✅ Your energy usage is balanced across appliances.")

        final_advice = "\n".join(advice_list)
        st.session_state.prediction = total_units

        # Display results
        st.success("⚡ Electricity Usage Summary")
        st.markdown(
            f"""
### 🔌 Units & Duration
- **Total Units Used:** `{total_units:.2f} units (kWh)`  
- **Duration of Usage:** `{duration} hours`  

---

### 💰 Estimated Cost
- **Total Cost:** `{total_cost:,.0f} TZS`

---

### 🧠 Advice Based on High Consumption Features
{final_advice}
"""
        )

# ===============================
# VISUALIZATION PAGE
# ===============================
elif page == "Visualization":
    st.header("📊 Energy Usage Visualization")

    if st.session_state.prediction is None:
        st.warning("⚠️ Please calculate energy first from the Prediction page.")
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

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(plot_df["Category"], plot_df["Power (kW)"], color="#38bdf8")
        ax.set_title("Household Power Distribution")
        ax.set_ylabel("Power (kW)")
        st.pyplot(fig)

        # Show units, duration, and total cost
        total_cost = st.session_state.prediction * PRICE_PER_UNIT
        st.info(
            f"""
🔋 **Total Units Used:** {st.session_state.prediction:.2f} units (kWh)  
⏱️ **Duration:** {st.session_state.duration} hours  
💰 **Total Cost:** {total_cost:,.0f} TZS
"""
        )
