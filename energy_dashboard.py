# =====================================
# INSTALL (IF NOT INSTALLED)
# pip install streamlit pandas numpy matplotlib scikit-learn openpyxl
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Smart Energy Usage Control",
    page_icon="⚡",
    layout="wide"
)

# =====================================
# HEADER
# =====================================
st.markdown("""
<div style="
    background: linear-gradient(90deg,#0f172a,#1e293b,#020617);
    padding:30px;
    border-radius:16px;
    text-align:center;
    margin-bottom:25px;">
    <div style="font-size:55px;">💡⚡</div>
    <h1 style="color:#facc15;margin:0;">Smart Energy Usage Control Dashboard</h1>
    <p style="color:#e5e7eb;">
        AI-Powered Prediction • Monitoring • Optimization
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("📌 Navigation")
st.sidebar.info("Upload data, train model, predict and visualize energy usage")

uploaded_file = st.sidebar.file_uploader(
    "📤 Upload Excel Dataset",
    type=["xlsx"]
)

# =====================================
# MAIN LOGIC
# =====================================
if uploaded_file is None:
    st.warning("⚠️ Please upload an Excel dataset to continue.")
    st.stop()

# =====================================
# LOAD & PREPROCESS DATA
# =====================================
df = pd.read_excel(uploaded_file)

st.subheader("📄 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# Combine Date & Time safely
df["datetime"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    dayfirst=True,
    errors="coerce"
)

numeric_cols = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Feature engineering
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
df["rolling_avg_3h"] = df["Global_active_power"].rolling(3).mean()

df_clean = df.dropna()

# =====================================
# MODEL TRAINING
# =====================================
X = df_clean[["hour", "day_of_week", "month", "rolling_avg_3h"]]
y = df_clean["Global_active_power"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# =====================================
# DASHBOARD METRICS
# =====================================
st.subheader("📊 Model Performance Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("📁 Records", df_clean.shape[0])
c2.metric("🧠 Model", "Random Forest")
c3.metric("📉 RMSE", f"{rmse:.3f}")
c4.metric("⚡ Avg Usage (kW)", f"{y.mean():.2f}")

# =====================================
# HIGH USAGE ANALYSIS
# =====================================
st.subheader("⚠️ High Energy Usage Prediction")

threshold = y_train.mean() + y_train.std()
high_usage = X_test[y_pred > threshold]

st.write("Predicted **high-energy usage periods**:")
st.dataframe(high_usage.head(), use_container_width=True)

# =====================================
# SMART RECOMMENDATIONS
# =====================================
st.subheader("🧠 AI Energy Recommendations")

def recommend_action(pred, threshold):
    if pred > threshold:
        return "⚡ High usage predicted — switch off non-essential appliances or delay heavy usage."
    else:
        return "✅ Usage within normal range — keep up efficient habits."

recent_preds = y_pred[-10:]
for i, p in enumerate(recent_preds, 1):
    st.write(f"{i}. {recommend_action(p, threshold)}")

# =====================================
# VISUALIZATION SECTION
# =====================================
st.subheader("📈 Energy Usage Visualization")

show_plot = st.checkbox("Show Energy Usage Graph")

if show_plot:
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(
        df_clean["datetime"][:1000],
        df_clean["Global_active_power"][:1000],
        color="orange"
    )
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Global Active Power (kW)")
    ax.set_title("Energy Consumption Over Time")
    ax.grid(alpha=0.3)

    st.pyplot(fig)

# =====================================
# FOOTER
# =====================================
st.markdown("""
<hr>
<p style="text-align:center;color:gray;">
⚡ Smart Energy AI Application | Machine Learning Project
</p>
""", unsafe_allow_html=True)
