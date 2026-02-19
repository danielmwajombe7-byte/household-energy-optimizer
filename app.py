import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import base64

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="⚡ Smart Energy Consumption Dashboard",
    layout="wide"
)

PRICE_PER_UNIT = 350  # TZS per kWh

# ===============================
# CLOUD-SAFE BACKGROUND (Prediction page only)
# ===============================
def add_bg_prediction():
    try:
        with open("energy.jpg", "rb") as img:
            encoded = base64.b64encode(img.read()).decode()
        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("⚠️ Background image 'energy.jpg' not found. Place it in the project folder.")

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
    if total == 0:
        return "ℹ️ No energy usage detected."

    percentages = {k: (v / total) * 100 for k, v in feature_usage.items()}
    max_feature = max(percentages, key=percentages.get)
    max_value = percentages[max_feature]
    high_features = [k for k, v in percentages.items() if v >= 25]

    context = {
        "Home": {
            "balanced": "This shows responsible household energy behavior.",
            "dominant": "Household appliances running for long hours increase consumption.",
            "suggestion": "Use energy-efficient appliances and avoid idle usage."
        },
        "Hospital": {
            "balanced": "Efficient hospital energy management is observed.",
            "dominant": "Critical medical equipment requires continuous power.",
            "suggestion": "Optimize non-critical systems where possible."
        },
        "Mining": {
            "balanced": "Mining energy usage is well distributed.",
            "dominant": "Heavy machinery consumes large continuous power.",
            "suggestion": "Apply machinery scheduling and preventive maintenance."
        },
        "School": {
            "balanced": "Effective energy usage in learning facilities.",
            "dominant": "Lighting and ICT labs significantly affect consumption.",
            "suggestion": "Automate lighting and limit idle ICT usage."
        },
        "Industry": {
            "balanced": "Industrial energy demand is well managed.",
            "dominant": "Industrial machines consume high power during operation.",
            "suggestion": "Schedule heavy machinery during off-peak hours."
        }
    }

    ctx = context.get(application, context["Home"])

    if max_value < 40:
        return f"✅ WELL-BALANCED USAGE\n\n{ctx['balanced']}\n\n💡 {ctx['suggestion']}"

    if len(high_features) >= 2:
        return (
            "⚖️ MULTIPLE HIGH ENERGY USERS\n\n"
            f"{', '.join(high_features)} consume high energy together.\n\n"
            f"📊 Recommendation: {ctx['suggestion']}"
        )

    return (
        f"⚠️ HIGH CONSUMPTION: {max_feature}\n\n"
        f"{ctx['dominant']}\n\n"
        f"🔧 Action: {ctx['suggestion']}"
    )

# ===============================
# SESSION STATE
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
        ["Prediction", "Visualization", "Summary"]
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Logout"):
        for k, v in defaults.items():
            st.session_state[k] = v
else:
    st.session_state.page = "Login"

# ===============================
# LOGIN PAGE
# ===============================
if st.session_state.page == "Login":
    st.markdown("""
    <div style="background:rgba(30,41,59,0.85);padding:25px;border-radius:14px;color:white;text-align:center;">
        <h1>⚡ Smart Energy Consumption System</h1>
        <p>Login to access predictions</p>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.username = st.text_input("👤 Your Name")
    st.session_state.application = st.selectbox(
        "🏭 Select Application",
        ["Select"] + list(APPLICATION_FEATURES.keys())
    )

    if st.button("➡️ Enter"):
        if st.session_state.username == "" or st.session_state.application == "Select":
            st.warning("⚠️ Fill all fields")
        else:
            st.session_state.logged_in = True
            st.session_state.page = "Prediction"

# ===============================
# PREDICTION PAGE
# ===============================
elif st.session_state.page == "Prediction":

    add_bg_prediction()  # background only here

    st.markdown(f"""
    <div style="background:rgba(15,118,110,0.85);padding:20px;border-radius:14px;color:white;text-align:center;">
        <h2>⚡ Energy Prediction</h2>
        <p>User: <b>{st.session_state.username}</b> | Application: <b>{st.session_state.application}</b></p>
    </div>
    """, unsafe_allow_html=True)

    features = APPLICATION_FEATURES[st.session_state.application]
    cols = st.columns(2)
    values = {}

    for i, feat in enumerate(features):
        values[feat] = cols[i % 2].number_input(f"{feat} Power (kW)", 0.0, value=5.0)

    st.session_state.duration = st.slider("⏱️ Duration (Hours)", 0.5, 24.0, 5.0, 0.5)

    if st.button("🚀 Predict", use_container_width=True):
        total_units = sum(values.values()) * st.session_state.duration
        cost = total_units * PRICE_PER_UNIT

        st.session_state.feature_values = values
        st.session_state.prediction = total_units
        st.session_state.advice = generate_energy_advice(
            st.session_state.application,
            {k: v * st.session_state.duration for k, v in values.items()}
        )

        st.success("✅ Prediction completed")

        st.markdown(f"""
### 🔌 Results
- **Energy Used:** `{total_units:.2f} kWh`
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
        st.warning("⚠️ Predict first")
    else:
        st.subheader("📊 Energy Distribution")

        plot_df = pd.DataFrame({
            "Category": st.session_state.feature_values.keys(),
            "Power (kW)": st.session_state.feature_values.values()
        })

        fig, ax = plt.subplots()
        ax.bar(plot_df["Category"], plot_df["Power (kW)"], color='teal')
        ax.set_ylabel("Power (kW)")
        plt.xticks(rotation=30)

        st.pyplot(fig)

        st.info(
            f"🔋 Total Energy: {st.session_state.prediction:.2f} kWh\n"
            f"💰 Cost: {st.session_state.prediction * PRICE_PER_UNIT:,.0f} TZS"
        )

# ===============================
# SUMMARY PAGE
# ===============================
elif st.session_state.page == "Summary":
    st.markdown("""
    <div style="background:rgba(100,100,120,0.85);padding:20px;border-radius:14px;color:white;text-align:center;">
        <h2>📄 Feature Usage Summary</h2>
        <p>Typical Power Ratings (kW) for Various Applications</p>
    </div>
    """ , unsafe_allow_html=True)

    # Typical power table
    typical_power_data = [
        ["Home", "Kitchen (oven, stove, fridge)", "1.0 -- 3.0", "Oven/stove high, fridge low (~0.1--0.2 kW)"],
        ["Home", "Laundry (washing machine)", "0.5 -- 2.0", "Depends on cycle & heater usage"],
        ["Home", "Lighting", "0.05 -- 0.5", "LED ~0.05 kW, CFL ~0.1--0.2 kW"],
        ["Home", "Other Appliances", "0.1 -- 1.0", "TV, microwave, fan, iron, etc."],
        ["Home", "Extra Loss", "0.05 -- 0.2", "Vampire power of devices"],

        ["School", "Classrooms (lighting & fans)", "0.2 -- 1.0", "Depends on number of lights & fans"],
        ["School", "ICT Labs", "1.0 -- 5.0", "Computers, servers, printers"],
        ["School", "Lighting", "0.2 -- 0.5", "Hallways, classrooms"],
        ["School", "Extra Loss", "0.1 -- 0.3", "Miscellaneous standby devices"],

        ["Industry", "Machine Power", "5 -- 50+", "Heavy machinery, motors, CNC"],
        ["Industry", "HVAC", "5 -- 30", "Air conditioning & ventilation"],
        ["Industry", "Lighting", "1 -- 5", "Factory floor lighting"],
        ["Industry", "Extra Loss", "0.5 -- 2", "Office equipment, standby"],

        ["Mining", "Excavators", "50 -- 200", "Large machines, continuous operation"],
        ["Mining", "Conveyors", "10 -- 50", "Transport of ore, motors"],
        ["Mining", "Lighting", "1 -- 5", "Underground or outdoor lighting"],
        ["Mining", "Extra Loss", "1 -- 3", "Pumps, control systems, standby"]
    ]

    typical_df = pd.DataFrame(typical_power_data, columns=["Application", "Feature", "Power (kW)", "Notes"])
    st.dataframe(typical_df, use_container_width=True)

    # Show user prediction summary if available
    if st.session_state.feature_values and st.session_state.prediction is not None:
        duration = st.session_state.duration
        summary_df = pd.DataFrame({
            "Feature": list(st.session_state.feature_values.keys()),
            "Power (kW)": list(st.session_state.feature_values.values()),
            "Duration (h)": [duration]*len(st.session_state.feature_values),
            "Energy (kWh)": [v*duration for v in st.session_state.feature_values.values()]
        })
        summary_df["% of Total"] = (summary_df["Energy (kWh)"]/st.session_state.prediction*100).round(2)

        st.markdown("### 🔹 Your Prediction Summary")
        st.dataframe(summary_df, use_container_width=True)

        fig, ax = plt.subplots()
        ax.bar(summary_df["Feature"], summary_df["Energy (kWh)"], color='teal')
        ax.set_ylabel("Energy (kWh)")
        ax.set_title("Feature Energy Usage")
        plt.xticks(rotation=30)
        st.pyplot(fig)
