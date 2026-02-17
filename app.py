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
# FEATURES (FOR ML)
# ===============================
FEATURES = ["Kitchen_Power","Laundry_Power","Other_Use","Extra_Loss","Voltage","Current"]
TARGET = "Total_Power"

# Ensure required columns exist
for col in FEATURES + [TARGET]:
    if col not in df.columns:
        df[col] = 0.0

# ===============================
# TRAIN MODEL (DEMO)
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
for key in ["authenticated","prediction","duration","kitchen","laundry","other","extra","username","application_type"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in ["prediction","username","application_type"] else 0.0
st.session_state.authenticated = False if st.session_state.authenticated is None else st.session_state.authenticated

# ===============================
# SIDEBAR NAVIGATION
# ===============================
if st.session_state.authenticated:
    page = st.sidebar.radio("📌 Navigation", ["Prediction", "Visualization"])
else:
    page = "Login"

# ===============================
# LOGIN PAGE
# ===============================
if page == "Login":
    st.markdown("""
    <div style="background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);
    padding:20px;border-radius:12px;color:white;text-align:center;">
        <h1>⚡ Enter Your Details to Access Prediction</h1>
    </div>
    """, unsafe_allow_html=True)
    
    username = st.text_input("👤 Enter Your Name")
    application_type = st.selectbox("🏭 Select Application Type", ["Select","Household","Industry","Hospital","Other"])
    
    if st.button("➡️ Enter"):
        if username.strip() == "" or application_type == "Select":
            st.warning("⚠️ Verbal Warning. Please enter your Name and Application Type!")
        else:
            st.session_state.username = username.strip()
            st.session_state.application_type = application_type
            st.session_state.authenticated = True
            st.experimental_rerun()  # go to Prediction page

# ===============================
# PREDICTION PAGE
# ===============================
elif page == "Prediction" and st.session_state.authenticated:
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#203a43,#2c5364);
    padding:15px;border-radius:12px;color:white;text-align:center;">
        <h2>⚡ Energy Prediction Form</h2>
        <p>User: <b>{st.session_state.username}</b> | Application: <b>{st.session_state.application_type}</b></p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    if st.button("🚀 Predict Energy Used", use_container_width=True):
        # Store input
        st.session_state.kitchen = kitchen
        st.session_state.laundry = laundry
        st.session_state.other = other
        st.session_state.extra = extra
        st.session_state.duration = duration
        
        # Compute feature energy
        feature_energy = {
            "Kitchen": kitchen * duration,
            "Laundry": laundry * duration,
            "Other Use": other * duration,
            "Extra Loss": extra * duration
        }
        total_units = sum(feature_energy.values())
        total_cost = total_units * PRICE_PER_UNIT
        
        # Advice based on top consuming features (>30%)
        advice_list = []
        for feat, energy in feature_energy.items():
            contribution = (energy / total_units) * 100
            if contribution > 30:
                if feat == "Kitchen":
                    advice_list.append("⚠️ Kitchen: Use lids, batch cooking to save energy.")
                elif feat == "Laundry":
                    advice_list.append("⚠️ Laundry: Wash full loads, avoid peak hours.")
                elif feat == "Other Use":
                    advice_list.append("⚠️ Other Usage: Turn off idle devices.")
                elif feat == "Extra Loss":
                    advice_list.append("⚠️ Extra Loss: Check wiring and appliances.")
        if not advice_list:
            advice_list.append("✅ Energy usage is balanced across appliances.")
        
        st.session_state.prediction = total_units
        final_advice = "\n".join(advice_list)
        
        # Display results
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
{final_advice}
""")

# ===============================
# VISUALIZATION PAGE
# ===============================
elif page == "Visualization" and st.session_state.authenticated:
    if st.session_state.prediction is None:
        st.warning("⚠️ Please predict energy first on the Prediction page.")
    else:
        st.subheader(f"📊 {st.session_state.application_type} Power Distribution")
        plot_df = pd.DataFrame({
            "Category":["Kitchen","Laundry","Other Use","Extra Loss"],
            "Power (kW)":[st.session_state.kitchen,st.session_state.laundry,st.session_state.other,st.session_state.extra]
        })
        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(plot_df["Category"], plot_df["Power (kW)"], color="#38bdf8")
        ax.set_ylabel("Power (kWh)")
        st.pyplot(fig)
        
        total_cost = st.session_state.prediction * PRICE_PER_UNIT
        st.info(f"""
🔋 **Total Units Used:** {st.session_state.prediction:.2f}  
⏱️ **Duration:** {st.session_state.duration} hours  
💰 **Total Cost:** {total_cost:,.0f} TZS
""")
