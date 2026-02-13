import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px
import os

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="⚡ Power Consumption Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Tanzania Power Consumption Dashboard")
st.caption("Smart Energy Prediction & Optimization")

# ============================================
# LOAD DATA (DIRECT – NO UPLOAD)
# ============================================
DATA_PATH = "tanzania_power_data.csv"

if not os.path.exists(DATA_PATH):
    st.error(f"❌ File '{DATA_PATH}' haipo. Hakikisha ipo folder moja na app.py")
    st.stop()

@st.cache_data
def load_data(path):
    df = pd.read_csv(
        path,
        sep=None,
        engine="python",
        on_bad_lines="skip"
    )
    return df

df = load_data(DATA_PATH)

st.success("✅ Dataset loaded successfully")

# ============================================
# DATA CLEANING
# ============================================
df_numeric = df.select_dtypes(include=["number"])

if df_numeric.shape[1] < 2:
    st.error("❌ Dataset haina numeric columns za kutosha")
    st.stop()

target_column = df_numeric.columns[-1]
feature_columns = df_numeric.columns[:-1]

X = df_numeric[feature_columns]
y = df_numeric[target_column]

# ============================================
# METRICS
# ============================================
c1, c2, c3 = st.columns(3)
c1.metric("Records", f"{len(df):,}")
c2.metric("Features", len(feature_columns))
c3.metric("Target", target_column)

# ============================================
# TRAIN MODEL
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
st.success(f"✅ Model trained | RMSE: {rmse:.3f}")

# ============================================
# VISUALIZATION
# ============================================
st.subheader("📊 Power Consumption Distribution")

fig = px.histogram(
    df_numeric,
    x=target_column,
    nbins=50,
    title="Target Distribution"
)
st.plotly_chart(fig, use_container_width=True)

# ============================================
# USER INPUT FOR PREDICTION
# ============================================
st.subheader("🔮 Predict Power Consumption")

input_data = {}
for col in feature_columns:
    input_data[col] = st.number_input(
        f"Enter {col}",
        value=float(df_numeric[col].mean())
    )

if st.button("🚀 Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    st.success(f"⚡ Predicted Consumption: **{prediction:.3f}**")

    if prediction > y.mean():
        st.warning("⚠️ High consumption predicted – reduce heavy appliance usage")
    else:
        st.info("✅ Consumption level is normal")

# ============================================
# DATA PREVIEW
# ============================================
with st.expander("🔍 View Dataset"):
    st.dataframe(df.head(100))
