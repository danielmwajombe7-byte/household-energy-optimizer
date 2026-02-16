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
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("tanzania_power_data.csv")
    return df

df = load_data()

# ===============================
# FEATURES & TARGET
# ===============================
FEATURES = [
    "Kitchen_Power",
    "Laundry_Power",
    "Other_Power",
    "Extra_Loss",
    "Voltage",
    "Current"
]

TARGET = "Total_Power"

# ===============================
# ENSURE COLUMNS EXIST
# ===============================
for col in FEATURES:
    if col not in df.columns:
        df[col] = 0.0

# Calculate TOTAL POWER correctly
df["Total_Power"] = (
    df["Kitchen_Power"]
    + df["Laundry_Power"]
    + df["Other_Power"]
    + df["Extra_Loss"]
)

# ===============================
# TRAIN MODEL
# ===============================
@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]

    model = DecisionTreeRegressor(
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)
    return model

model = train_model(df)

# ===============================
# SESSION STATE
# ===============================
if "prediction" not in st.session_state:
    st.session_state.prediction = None

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
    st.title("⚡ Smart Energy Consumption Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Records", len(df))
    c2.metric("📊 Features", len(FEATURES))
    c3.metric("🎯 Target", "Total Energy Used")
    c4.metric("🤖 Model", "Decision Tree")

# ===============================
# PREDICTION PAGE
# ===============================
elif page == "Prediction":
    st.header("🧮 Enter Household Energy Usage")

    col1, col2 = st.columns(2)

    with col1:
        kitchen = st.number_input("🍳 Kitchen Power (kW)", 0.0, value=4.5)
        laundry = st.number_input("🧺 Laundry Power (kW)", 0.0, value=6.0)
        other = st.number_input(
            "💡 Other Uses (TV, Bulbs, Phone Charging) (kW)",
            0.0,
            value=3.0
        )

    with col2:
        extra = st.number_input("⚠️ Extra Power Loss (kW)", 0.0, value=2.0)
        voltage = st.number_input("🔌 Voltage (V)", 0.0, value=220.0)
        current = st.number_input("⚡ Current (A)", 0.0, value=4.5)

    if st.button("🚀 Predict Total Energy Used", use_container_width=True):
        input_df = pd.DataFrame([{
            "Kitchen_Power": kitchen,
            "Laundry_Power": laundry,
            "Other_Power": other,
            "Extra_Loss": extra,
            "Voltage": voltage,
            "Current": current
        }])

        prediction = model.predict(input_df)[0]
        st.session_state.prediction = prediction

        st.success(
            f"✅ Predicted Total Energy Used: **{prediction:.2f} kW**"
        )

# ===============================
# VISUALIZATION
# ===============================
elif page == "Visualization":
    st.header("📊 Power Usage Breakdown")

    if st.session_state.prediction is None:
        st.warning("⚠️ Fanya prediction kwanza.")
    else:
        plot_df = pd.DataFrame({
            "Category": [
                "Kitchen",
                "Laundry",
                "Other Uses",
                "Extra Loss"
            ],
            "Power (kW)": [
                kitchen,
                laundry,
                other,
                extra
            ]
        })

        fig, ax = plt.subplots()
        ax.bar(plot_df["Category"], plot_df["Power (kW)"])
        ax.set_ylabel("Power (kW)")
        ax.set_title("Household Energy Distribution")

        st.pyplot(fig)

        st.info(
            f"🔮 Total Energy Used: **{st.session_state.prediction:.2f} kW**"
        )
