import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="⚡ Smart Energy Prediction Dashboard",
    page_icon="⚡",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================
st.markdown("""
<h1 style='text-align:center;'>⚡ Smart Energy Consumption Dashboard</h1>
<p style='text-align:center;color:gray;'>
Predict • Visualize • Understand Your Power Usage
</p>
""", unsafe_allow_html=True)

st.divider()

# =====================================================
# LOAD DATA (DIRECT)
# =====================================================
DATA_PATH = "tanzania_power_data.csv"

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset haipo. Hakikisha 'tanzania_power_data.csv' ipo folder moja na app.py")
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_PATH,
        sep=None,
        engine="python",
        on_bad_lines="skip"
    )

df = load_data()

# =====================================================
# CLEAN DATA – Ensure numeric columns exist
# =====================================================
df_numeric = df.select_dtypes(include="number")

if df_numeric.shape[1] == 0:
    st.error("❌ Hakuna numeric columns kwenye dataset. Angalia CSV yako!")
    st.stop()

# Set target (last numeric column)
target_column = df_numeric.columns[-1]
feature_columns = df_numeric.columns.drop(target_column)

X = df_numeric[feature_columns]
y = df_numeric[target_column]

# =====================================================
# TRAIN MODEL (directly inside app.py)
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=80,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

# =====================================================
# DASHBOARD METRICS
# =====================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("📄 Records", f"{len(df):,}")
c2.metric("📊 Features", len(feature_columns))
c3.metric("🎯 Target", target_column)
c4.metric("📉 RMSE", f"{rmse:.3f}")

st.divider()

# =====================================================
# TWO COLUMN DASHBOARD
# =====================================================
left, right = st.columns([1, 1.4])

# =====================================================
# LEFT SIDE – USER INPUT
# =====================================================
with left:
    st.subheader("🧮 Enter Values for Prediction")

    user_input = {}
    for col in feature_columns:
        user_input[col] = st.slider(
            col,
            float(df_numeric[col].min()),
            float(df_numeric[col].max()),
            float(df_numeric[col].mean())
        )

    predict_btn = st.button("🚀 Predict Energy Consumption", use_container_width=True)

# =====================================================
# RIGHT SIDE – RESULTS + ZOOM EFFECT
# =====================================================
with right:
    st.subheader("📈 Prediction Results")

    if predict_btn:
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]

        st.markdown(f"""
        <h2 style='color:#ff7f0e;'>⚡ Predicted Consumption</h2>
        <h1>{prediction:.3f}</h1>
        """, unsafe_allow_html=True)

        if prediction > y.mean():
            st.warning("⚠️ High energy usage detected. Consider reducing heavy appliances.")
        else:
            st.success("✅ Energy usage is within normal range.")

        # ===============================
        # ZOOM IN → OUT VISUALIZATION
        # ===============================
        zoom_df = pd.DataFrame({
            "Stage": ["Low", "Average", "Your Prediction", "High"],
            "Consumption": [
                y.min(),
                y.mean(),
                prediction,
                y.max()
            ]
        })

        fig = px.line(
            zoom_df,
            x="Stage",
            y="Consumption",
            markers=True,
            title="🔍 Zoom View of Your Energy Consumption",
            line_shape="spline"
        )

        fig.update_traces(
            marker=dict(size=14),
            line=dict(width=4)
        )

        fig.update_layout(
            transition_duration=1200,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 Enter values and click **Predict** to see results")

# =====================================================
# DATA PREVIEW
# =====================================================
st.divider()
with st.expander("🔎 View Dataset Preview"):
    st.dataframe(df.head(100))
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True,
    errors="coerce"
)

