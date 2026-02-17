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
PRICE_PER_UNIT = 350

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
APPLICATION_FEATURES = {
    "Household": ["Kitchen", "Laundry", "Other Use", "Extra Loss"],
    "Industry": ["Machine Power", "HVAC", "Lighting", "Extra Loss"],
    "Hospital": ["ICU", "Labs", "Lighting", "Extra Loss"],
    "School": ["Classrooms", "Labs", "Lighting", "Extra Loss"],
    "Mining": ["Excavators", "Conveyors", "Lighting", "Extra Loss"],
    "Other": ["Custom1", "Custom2", "Custom3", "Extra Loss"]
}

# ===============================
# FEATURES FOR ML DEMO (all)
# ===============================
ALL_FEATURES = ["Kitchen", "Laundry", "Other Use", "Extra Loss",
                "Machine Power", "HVAC", "Lighting", "ICU", "Labs",
                "Classrooms", "Excavators", "Conveyors", "Custom1", "Custom2", "Custom3"]
TARGET = "Total_Power"

# Ensure columns exist in df
for col in ALL_FEATURES + [TARGET]:
    if col not in df.columns:
        df[col] = 0.0

# ===============================
# TRAIN MODEL
# ===============================
@st.cache_resource
def train_model(df):
    X = df[ALL_FEATURES]
    y = df[TARGET]
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# ===============================
# SESSION STATE INIT
# ===============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "application" not in st.session_state:
    st.session_state.application = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "feature_values" not in st.session_state:
    st.session_state.feature_values = {}
if "duration" not in st.session_state:
    st.session_state.duration = 1.0

# ===============================
# LOGIN PAGE
# ===============================
if not st.session_state.logged_in:
    st.title("⚡ Energy Prediction Access")
    st.subheader("Enter Your Details to Access Prediction")

    username = st.text_input("👤 Username")
    application = st.selectbox("🏢 Application Type", list(APPLICATION_FEATURES.keys()))

    if st.button("🔑 Enter"):
        if username.strip() == "" or application.strip() == "":
            st.warning("⚠️ Verbal Warning. Please provide both Username and Application Type!")
        else:
            st.session_state.username = username
            st.session_state.application = application
            st.session_state.logged_in = True
            st.experimental_rerun()  # go directly to prediction

# ===============================
# PREDICTION PAGE
# ===============================
else:
    st.title("🧮 Energy Prediction Input")
    st.subheader(f"Application: {st.session_state.application} | User: {st.session_state.username}")

    # Dynamic input fields based on application
    features = APPLICATION_FEATURES[st.session_state.application]
    cols = st.columns(2)
    feature_values = {}

    for i, feat in enumerate(features):
        col = cols[i % 2]
        feature_values[feat] = col.number_input(f"{feat} Power (kW)", 0.0, value=5.0)

    duration = st.slider(
        "⏱️ Duration of Usage (Hours)",
        min_value=0.5,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    if st.button("🚀 Predict Energy Usage", use_container_width=True):
        st.session_state.feature_values = feature_values
        st.session_state.duration = duration

        # Calculate total units
        total_units = sum(feature_values.values()) * duration
        total_cost = total_units * PRICE_PER_UNIT
        st.session_state.prediction = total_units

        # Identify high consumption features (>30%)
        advice_list = []
        for feat, val in feature_values.items():
            energy = val * duration
            contribution = (energy / total_units) * 100
            if contribution > 30:
                advice_list.append(f"⚠️ {feat} consumes a lot! Consider efficient usage.")

        if not advice_list:
            advice_list.append("✅ Your energy usage is balanced across appliances.")

        final_advice = "\n".join(advice_list)

        # Display results
        st.success("⚡ Predict Energy Used")
        st.markdown(
            f"""
### 🔌 Units & Duration
- **Total Units Used:** `{total_units:.2f} units`  
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
    st.header("📊 Energy Usage Visualization")
    if st.session_state.prediction is None:
        st.info("🔹 Please predict energy usage first.")
    else:
        plot_df = pd.DataFrame({
            "Category": list(st.session_state.feature_values.keys()),
            "Power (kW)": list(st.session_state.feature_values.values())
        })

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(plot_df["Category"], plot_df["Power (kW)"], color="#38bdf8")
        ax.set_title("Power Distribution by Feature")
        ax.set_ylabel("Power (kW)")
        st.pyplot(fig)

        st.info(
            f"""
🔋 **Total Units Used:** {st.session_state.prediction:.2f} units  
⏱️ **Duration:** {st.session_state.duration} hours  
💰 **Estimated Cost:** {st.session_state.prediction*PRICE_PER_UNIT:,.0f} TZS
"""
        )
