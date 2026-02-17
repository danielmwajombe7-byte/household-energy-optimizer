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

PRICE_PER_UNIT = 350  # TZS per kWh

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("tanzania_power_data.csv")
    except:
        return pd.DataFrame()

df = load_data()

# ===============================
# APPLICATION FEATURES
# ===============================
APPLICATION_FEATURES = {
    "Home": ["Kitchen", "Laundry", "Lighting", "Other Appliances", "Extra Loss"],
    "Industry": ["Machine Power", "HVAC", "Lighting", "Extra Loss"],
    "Hospital": ["ICU", "Laboratory", "Lighting", "Extra Loss"],
    "School": ["Classrooms", "ICT Labs", "Lighting", "Extra Loss"],
    "Mining": ["Excavators", "Conveyors", "Lighting", "Extra Loss"]
}

ALL_FEATURES = sum(APPLICATION_FEATURES.values(), [])
TARGET = "Total_Power"

for col in ALL_FEATURES + [TARGET]:
    if col not in df.columns:
        df[col] = 0.0

# ===============================
# TRAIN MODEL (DEMO)
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
# SMART ADVICE FUNCTION
# ===============================
def generate_energy_advice(application, feature_usage):
    total = sum(feature_usage.values())
    percentages = {k: (v / total) * 100 for k, v in feature_usage.items()}

    max_feature = max(percentages, key=percentages.get)
    max_value = percentages[max_feature]
    high_features = [k for k, v in percentages.items() if v >= 25]

    context = {
        "Home": {
            "balanced": "This shows responsible household energy behavior.",
            "dominant": "Household appliances running for long hours often increase consumption.",
            "suggestion": "Use energy-efficient appliances, switch off idle devices, and stagger usage."
        },
        "Hospital": {
            "balanced": "This indicates efficient hospital energy management.",
            "dominant": "Critical medical equipment requires continuous power.",
            "suggestion": "Optimize non-critical systems and maintain energy-efficient medical equipment."
        },
        "Mining": {
            "balanced": "Energy usage is well distributed across mining operations.",
            "dominant": "Heavy machinery consumes large continuous power.",
            "suggestion": "Apply machinery scheduling, load balancing, and preventive maintenance."
        },
        "School": {
            "balanced": "This reflects effective energy use in learning facilities.",
            "dominant": "Lighting and ICT labs significantly affect school energy usage.",
            "suggestion": "Automate lighting, limit idle ICT use, and promote energy awareness."
        },
        "Industry": {
            "balanced": "Energy demand is evenly managed across industrial systems.",
            "dominant": "Industrial machines require high power during peak operation.",
            "suggestion": "Schedule heavy machinery and invest in efficient industrial equipment."
        }
    }

    ctx = context.get(application, context["Home"])

    if max_value < 40:
        return (
            "✅ WELL-BALANCED ENERGY USAGE\n\n"
            f"{ctx['balanced']} Energy is evenly distributed, reducing system overload "
            "and lowering operational costs.\n\n"
            f"💡 Recommendation: {ctx['suggestion']}"
        )

    if max_value >= 45:
        return (
            f"⚠️ HIGH ENERGY CONSUMPTION: {max_feature}\n\n"
            f"{ctx['dominant']} This increases total cost and energy strain.\n\n"
            f"🔧 Action: Moderate usage, not elimination. {ctx['suggestion']}"
        )

    if len(high_features) >= 2:
        return (
            "⚖️ MULTIPLE HIGH ENERGY CONSUMERS\n\n"
            f"The following features consume high energy together: {', '.join(high_features)}.\n\n"
            f"📊 Recommendation: Avoid simultaneous usage and apply moderation. {ctx['suggestion']}"
        )

    return "ℹ️ Energy analysis completed. Continue monitoring usage."

# ===============================
# SESSION STATE INIT
# ===============================
defaults = {
    "logged_in": False,
    "username": "",
    "application": "",
    "page": "Login",
    "feature_values": {},
    "duration": 1.0,
    "prediction": None,
    "advice": ""
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===============================
# SIDEBAR NAVIGATION
# ===============================
if st.session_state.logged_in:
    st.sidebar.title("📌 Navigation")
    st.session_state.page = st.sidebar.radio(
        "Go to",
        ["Prediction", "Visualization"],
        index=0 if st.session_state.page == "Prediction" else 1
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Logout / Change Application"):
        for k, v in defaults.items():
            st.session_state[k] = v
else:
    st.session_state.page = "Login"

# ===============================
# LOGIN PAGE
# ===============================
if st.session_state.page == "Login":
    st.markdown("""
    <div style="background:#1e293b;padding:20px;border-radius:12px;color:white;text-align:center;">
        <h1>⚡ Enter Your Details to Access Prediction</h1>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.username = st.text_input("👤 Your Name", st.session_state.username)
    st.session_state.application = st.selectbox(
        "🏭 Select Application",
        ["Select"] + list(APPLICATION_FEATURES.keys())
    )

    if st.button("➡️ Enter"):
        if st.session_state.username == "" or st.session_state.application == "Select":
            st.warning("⚠️ Please enter name and select application.")
        else:
            st.session_state.logged_in = True
            st.session_state.page = "Prediction"

# ===============================
# PREDICTION PAGE
# ===============================
elif st.session_state.page == "Prediction":
    st.markdown(f"""
    <div style="background:#0f766e;padding:15px;border-radius:12px;color:white;text-align:center;">
        <h2>⚡ Energy Prediction Form</h2>
        <p>User: <b>{st.session_state.username}</b> | Application: <b>{st.session_state.application}</b></p>
    </div>
    """, unsafe_allow_html=True)

    features = APPLICATION_FEATURES[st.session_state.application]
    cols = st.columns(2)
    values = {}

    for i, feat in enumerate(features):
        values[feat] = cols[i % 2].number_input(f"{feat} Power (kW)", 0.0, value=5.0)

    st.session_state.duration = st.slider("⏱️ Duration (Hours)", 0.5, 24.0, 5.0, 0.5)

    if st.button("🚀 Predict Energy Used", use_container_width=True):
        total_units = sum(values.values()) * st.session_state.duration
        cost = total_units * PRICE_PER_UNIT

        st.session_state.feature_values = values
        st.session_state.prediction = total_units
        st.session_state.advice = generate_energy_advice(
            st.session_state.application,
            {k: v * st.session_state.duration for k, v in values.items()}
        )

        st.success("⚡ Prediction Completed")

        st.markdown(f"""
### 🔌 Results
- **Total Energy Used:** `{total_units:.2f} kWh`
- **Estimated Cost:** `{cost:,.0f} TZS`

---

### 🧠 Smart Advice
{st.session_state.advice}
""")

# ===============================
# VISUALIZATION PAGE
# ===============================
elif st.session_state.page == "Visualization":
    if st.session_state.prediction is None:
        st.warning("⚠️ Please make a prediction first.")
    else:
        st.subheader("📊 Energy Distribution")

        plot_df = pd.DataFrame({
            "Category": st.session_state.feature_values.keys(),
            "Power (kW)": st.session_state.feature_values.values()
        })

        fig, ax = plt.subplots()
        ax.bar(plot_df["Category"], plot_df["Power (kW)"])
        ax.set_ylabel("Power (kW)")
        plt.xticks(rotation=30)

        st.pyplot(fig)

        st.info(f"""
🔋 Total Energy: {st.session_state.prediction:.2f} kWh  
💰 Cost: {st.session_state.prediction * PRICE_PER_UNIT:,.0f} TZS
""")
