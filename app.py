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
# APPLICATION FEATURES
# ===============================
APP_FEATURES = {
    "Household": ["Kitchen_Power", "Laundry_Power", "Other_Use", "Extra_Loss"],
    "Industry": ["Machine_Power", "Lighting_Power", "HVAC_Power", "Extra_Loss"],
    "Hospital": ["ICU_Power", "Lab_Power", "Lighting_Power", "Extra_Loss"],
    "Other": ["General_Use", "Lighting_Power", "Extra_Loss"]
}

TARGET = "Total_Power"

# Ensure columns exist in df
all_features = set(sum(APP_FEATURES.values(), []))
for col in all_features.union({TARGET}):
    if col not in df.columns:
        df[col] = 0.0

# ===============================
# TRAIN DEMO MODEL
# ===============================
@st.cache_resource
def train_model(df):
    X = df[list(all_features)]
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
    "feature_values": {},
    "user_name": "",
    "application_type": "Household"
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

    # User input
    st.subheader("👤 User Info & Application")
    col1, col2 = st.columns(2)
    with col1:
        user_name = st.text_input("Enter your name", st.session_state.user_name)
        st.session_state.user_name = user_name
    with col2:
        application_type = st.selectbox("Select Application Type", list(APP_FEATURES.keys()), index=list(APP_FEATURES.keys()).index(st.session_state.application_type))
        st.session_state.application_type = application_type

    # Metrics with mini chart
    st.subheader("📊 Dashboard Summary")
    feature_example = {feat.replace("_Power","").replace("_"," "): 5 for feat in APP_FEATURES[application_type]}
    col1, col2, col3 = st.columns(3)
    col1.metric("📄 Records", df.shape[0])
    col2.metric("📊 Features", len(APP_FEATURES[application_type]))
    col3.metric("🎯 Target", "Energy (kWh)")

    fig, ax = plt.subplots(figsize=(6,2))
    ax.barh(list(feature_example.keys()), list(feature_example.values()), color="#60a5fa")
    ax.set_xlabel("Power (kWh)")
    st.pyplot(fig)

# ===============================
# PREDICTION PAGE
# ===============================
elif page == "Prediction":
    st.header("🧮 Energy Prediction Input")
    features = APP_FEATURES[st.session_state.application_type]
    feature_values = {}
    col1, col2 = st.columns(2)
    for i, feat in enumerate(features):
        label = feat.replace("_Power","").replace("_"," ")
        if i %2 == 0:
            feature_values[feat] = col1.number_input(f"{label} Power (kW)", 0.0, value=5.0)
        else:
            feature_values[feat] = col2.number_input(f"{label} Power (kW)", 0.0, value=5.0)

    st.markdown("---")
    duration = st.slider("⏱️ Duration of Usage (Hours)", min_value=0.5, max_value=24.0, value=5.0, step=0.5)

    if st.button("🚀 Calculate Energy Consumption", use_container_width=True):
        st.session_state.feature_values = feature_values
        st.session_state.duration = duration

        # Total units per feature
        feature_energy = {feat.replace("_Power","").replace("_"," "): val*duration for feat,val in feature_values.items()}
        total_units = sum(feature_energy.values())
        total_cost = total_units * PRICE_PER_UNIT

        # Advice based on features consuming >30%
        advice_list = []
        for feat, energy in feature_energy.items():
            if (energy/total_units)*100 > 30:
                advice_list.append(f"⚠️ {feat} consumes a lot! Consider efficient use.")

        if not advice_list:
            advice_list.append("✅ Your energy usage is balanced across features.")

        final_advice = "\n".join(advice_list)
        st.session_state.prediction = total_units

        # Display results
        st.success(f"⚡ Electricity Usage Summary for {st.session_state.user_name}")
        st.markdown(
            f"""
### 🔌 Units & Duration
- **Total Units Used:** `{total_units:.2f} units` (1 unit = 1 kWh)  
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
        feature_values = st.session_state.feature_values
        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(feature_values.keys(), feature_values.values(), color="#38bdf8")
        ax.set_title(f"{st.session_state.application_type} Power Distribution")
        ax.set_ylabel("Power (kWh)")
        st.pyplot(fig)

        total_cost = st.session_state.prediction * PRICE_PER_UNIT
        st.info(
            f"""
🔋 **Total Units Used:** {st.session_state.prediction:.2f} units (1 unit = 1 kWh)  
⏱️ **Duration:** {st.session_state.duration} hours  
💰 **Total Cost:** {total_cost:,.0f} TZS
"""
        )
