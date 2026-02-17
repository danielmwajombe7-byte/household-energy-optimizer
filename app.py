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
# ELECTRICITY PRICE
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

ALL_FEATURES = list(set(sum(APPLICATION_FEATURES.values(), [])))
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
# SESSION STATE INIT
# ===============================
defaults = {
    "logged_in": False,
    "username": "",
    "application": "",
    "prediction": None,
    "feature_values": {},
    "duration": 1.0
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===============================
# SIDEBAR NAVIGATION + LOGOUT
# ===============================
if st.session_state.logged_in:
    page = st.sidebar.radio("📌 Navigation", ["Prediction", "Visualization"])

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Change Application / Logout"):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()
else:
    page = "Login"

# ===============================
# LOGIN PAGE
# ===============================
if page == "Login":
    st.markdown("""
    <div style="background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);
    padding:25px;border-radius:15px;color:white;text-align:center;">
        <h1>⚡ Smart Energy Consumption Dashboard</h1>
        <p>Login to access energy prediction</p>
    </div>
    """, unsafe_allow_html=True)

    username = st.text_input("👤 Enter Your Name")
    application = st.selectbox(
        "🏭 Select Application Type",
        ["Select"] + list(APPLICATION_FEATURES.keys())
    )

    if st.button("➡️ Enter"):
        if username.strip() == "" or application == "Select":
            st.warning("⚠️ Verbal Warning. Please enter your Name and Application Type!")
        else:
            st.session_state.username = username.strip()
            st.session_state.application = application
            st.session_state.logged_in = True
            st.rerun()

# ===============================
# PREDICTION PAGE
# ===============================
elif page == "Prediction":
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#203a43,#2c5364);
    padding:18px;border-radius:15px;color:white;text-align:center;">
        <h2>⚡ Energy Prediction</h2>
        <p><b>{st.session_state.username}</b> | {st.session_state.application}</p>
    </div>
    """, unsafe_allow_html=True)

    features = APPLICATION_FEATURES[st.session_state.application]
    cols = st.columns(2)
    feature_values = {}

    for i, feat in enumerate(features):
        with cols[i % 2]:
            feature_values[feat] = st.number_input(
                f"{feat} Power (kW)", min_value=0.0, value=5.0
            )

    duration = st.slider(
        "⏱️ Duration of Usage (Hours)", 0.5, 24.0, 5.0, 0.5
    )

    if st.button("🚀 Predict Energy Used", use_container_width=True):
        st.session_state.feature_values = feature_values
        st.session_state.duration = duration

        total_units = sum(feature_values.values()) * duration
        total_cost = total_units * PRICE_PER_UNIT
        st.session_state.prediction = total_units

        advice = []
        for feat, val in feature_values.items():
            if total_units > 0 and (val * duration / total_units) * 100 > 30:
                advice.append(f"⚠️ {feat} consumes a large share of energy.")

        if not advice:
            advice.append("✅ Energy usage is well balanced.")

        st.success("⚡ Prediction Complete")
        st.markdown(f"""
### 🔌 Units & Duration
- **Total Units Used:** `{total_units:.2f}`  
- **Duration:** `{duration} hours`

---

### 💰 Estimated Cost
- **Total Cost:** `{total_cost:,.0f} TZS`

---

### 🧠 Advice
{"<br>".join(advice)}
""", unsafe_allow_html=True)

# ===============================
# VISUALIZATION PAGE
# ===============================
elif page == "Visualization":
    if st.session_state.prediction is None:
        st.warning("⚠️ Please predict energy first.")
    else:
        st.subheader(f"📊 {st.session_state.application} Power Distribution")

        plot_df = pd.DataFrame({
            "Category": st.session_state.feature_values.keys(),
            "Power (kW)": st.session_state.feature_values.values()
        })

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(plot_df["Category"], plot_df["Power (kW)"])
        ax.set_ylabel("Power (kW)")
        st.pyplot(fig)

        st.info(f"""
🔋 **Total Units Used:** {st.session_state.prediction:.2f}  
⏱️ **Duration:** {st.session_state.duration} hours  
💰 **Estimated Cost:** {st.session_state.prediction * PRICE_PER_UNIT:,.0f} TZS
""")
