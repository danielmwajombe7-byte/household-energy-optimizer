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
# FEATURES (FOR ML – DEMO)
# ===============================
FEATURES = ["Kitchen_Power","Laundry_Power","Other_Use","Extra_Loss","Voltage","Current"]
TARGET = "Total_Power"

for col in FEATURES + [TARGET]:
    if col not in df.columns:
        df[col] = 0.0

# ===============================
# TRAIN MODEL
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
session_defaults = {
    "user_name": "",
    "application_type": "Select",
    "authenticated": False,
    "prediction": None,
    "duration": 1.0,
    "kitchen": 0.0,
    "laundry": 0.0,
    "other": 0.0,
    "extra": 0.0,
    "show_prediction_form": False
}

for k, v in session_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===============================
# LOGIN PAGE
# ===============================
if not st.session_state.authenticated:
    st.markdown("""
    <div style="background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);
    padding:20px;border-radius:15px;color:white;text-align:center;">
        <div style="font-size:45px;">💡</div>
        <h1>Smart Energy Consumption Dashboard</h1>
        <p>Please login to access prediction</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        st.subheader("Enter Your Details")
        user_name = st.text_input("👤 Your Name")
        application_type = st.selectbox("🏭 Application Type", ["Select", "Hospital", "Industry", "Other"])
        submitted = st.form_submit_button("Continue")

        if submitted:
            if user_name.strip() == "" or application_type == "Select":
                st.warning("⚠️ Verbal Warning. Please enter your Name and select Application Type.")
            else:
                # Save user info and show prediction form
                st.session_state.user_name = user_name
                st.session_state.application_type = application_type
                st.session_state.authenticated = True
                st.session_state.show_prediction_form = True

# ===============================
# PREDICTION PAGE
# ===============================
if st.session_state.authenticated and st.session_state.show_prediction_form:
    st.header("🧮 Predict Energy Usage")
    st.write(f"User: **{st.session_state.user_name}** | Application: **{st.session_state.application_type}**")
    
    col1, col2 = st.columns(2)

    with col1:
        kitchen = st.number_input("🍳 Kitchen Power (kW)", 0.0, value=4.5)
        laundry = st.number_input("🧺 Laundry Power (kW)", 0.0, value=6.0)
        other = st.number_input("💡 Other Usage (kW)", 0.0, value=3.0)

    with col2:
        extra = st.number_input("🔥 Extra Loss (kW)", 0.0, value=2.0)
        voltage = st.number_input("⚡ Voltage (V)", 0.0, value=220.0)
        current = st.number_input("🔁 Current (A)", 0.0, value=4.5)

    duration = st.slider("⏱️ Duration of Usage (Hours)", 0.5, 24.0, 5.0, 0.5)

    if st.button("⚡ Predict Energy Used"):
        st.session_state.kitchen = kitchen
        st.session_state.laundry = laundry
        st.session_state.other = other
        st.session_state.extra = extra
        st.session_state.duration = duration

        # Feature energy contribution
        feature_energy = {
            "Kitchen": kitchen * duration,
            "Laundry": laundry * duration,
            "Other Use": other * duration,
            "Extra Loss": extra * duration
        }
        total_units = sum(feature_energy.values())
        total_cost = total_units * PRICE_PER_UNIT

        advice_list = []
        for feat, energy in feature_energy.items():
            if (energy / total_units) * 100 > 30:
                if feat == "Kitchen":
                    advice_list.append("⚠️ Kitchen high! Cook efficiently, batch cook, use lids.")
                elif feat == "Laundry":
                    advice_list.append("⚠️ Laundry high! Wash full loads, avoid peak hours.")
                elif feat == "Other Use":
                    advice_list.append("⚠️ Other high! Turn off devices when idle.")
                elif feat == "Extra Loss":
                    advice_list.append("⚠️ Extra losses high! Check wiring & appliances.")

        if not advice_list:
            advice_list.append("✅ Energy usage is balanced across appliances.")

        final_advice = "\n".join(advice_list)
        st.session_state.prediction = total_units

        st.success("⚡ Prediction Result")
        st.markdown(
            f"""
### 🔌 Units & Duration
- **Total Units Used:** `{total_units:.2f} units`  
- **Duration:** `{duration} hours`  

---

### 💰 Estimated Cost
- **Total Cost:** `{total_cost:,.0f} TZS`

---

### 🧠 Advice Based on High Consumption
{final_advice}
"""
        )

# ===============================
# VISUALIZATION PAGE
# ===============================
if st.session_state.authenticated and st.session_state.prediction is not None:
    st.subheader("📊 Energy Usage Visualization")
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
    ax.set_title(f"{st.session_state.application_type} Power Distribution")
    ax.set_ylabel("Power (kWh)")
    st.pyplot(fig)

    total_cost = st.session_state.prediction * PRICE_PER_UNIT
    st.info(
        f"""
🔋 **Total Units Used:** {st.session_state.prediction:.2f} units  
⏱️ **Duration:** {st.session_state.duration} hours  
💰 **Total Cost:** {total_cost:,.0f} TZS
"""
    )
