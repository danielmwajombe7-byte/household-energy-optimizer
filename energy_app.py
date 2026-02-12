import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(
    page_title="⚡ Energy Usage Dashboard",
    layout="wide",
    page_icon="⚡"
)

st.title("⚡ Household Energy Consumption & Prediction Dashboard")

# ------------------------
# Load dataset (built‑in)
# ------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("individual+household+electric+power+consumption.csv",
                     sep=";", na_values="?")
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                    dayfirst=True, errors='coerce')
    numeric_cols = [
        'Global_active_power','Global_reactive_power','Voltage',
        'Sub_metering_1','Sub_metering_2','Sub_metering_3'
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['rolling_avg_3h'] = df['Global_active_power'].rolling(3).mean().bfill()
    df_clean = df.dropna(subset=['Global_active_power','rolling_avg_3h'])
    return df_clean

df = load_data()

# ------------------------
# Overview metrics
# ------------------------

st.markdown("## 📊 Data Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Avg Active Power (kW)", f"{df['Global_active_power'].mean():.2f}")
col3.metric("Peak Active Power (kW)", f"{df['Global_active_power'].max():.2f}")

# ------------------------
# Interactive Filters
# ------------------------

st.markdown("## 🔍 Filters & Visualizations")

with st.expander("Select time range / filters"):
    min_date, max_date = st.slider("Select datetime range",
                                   value=(df['datetime'].min(), df['datetime'].max()),
                                   format="YYYY-MM-DD HH:mm")
    sel_days = st.multiselect("Days of Week (0=Mon)", sorted(df['day_of_week'].unique()),
                              default=sorted(df['day_of_week'].unique()))
    sel_hours = st.slider("Hour range", 0, 23, (0,23))

df_filtered = df[
    (df['datetime'] >= min_date) &
    (df['datetime'] <= max_date) &
    (df['day_of_week'].isin(sel_days)) &
    (df['hour'] >= sel_hours[0]) &
    (df['hour'] <= sel_hours[1])
]

st.write(f"Filtered dataset: {len(df_filtered):,} rows")

# ------------------------
# Plots
# ------------------------

st.markdown("### 📈 Energy Usage Over Time")
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df_filtered['datetime'], df_filtered['Global_active_power'], alpha=0.6)
ax.set_xlabel("Datetime")
ax.set_ylabel("Global Active Power (kW)")
st.pyplot(fig)

st.markdown("### 📊 Hourly Average Power")
hourly = df_filtered.groupby('hour')['Global_active_power'].mean()
st.bar_chart(hourly)

# ------------------------
# Train ML Model
# ------------------------

X = df[['hour','day_of_week','month','rolling_avg_3h']]
y = df['Global_active_power']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=80, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.markdown("## 🤖 Prediction Model Performance")
st.write(f"**RMSE:** {rmse:.3f} kW")

# ------------------------
# Interactive Prediction Panel
# ------------------------

st.markdown("## 🧠 Custom Input Prediction")

colA, colB, colC, colD = st.columns(4)
input_hour = colA.number_input("Hour (0–23)", min_value=0, max_value=23, value=12)
input_day = colB.number_input("Day of Week (0=Mon…6=Sun)", min_value=0, max_value=6, value=2)
input_month = colC.number_input("Month (1–12)", min_value=1, max_value=12, value=6)
input_rolling = colD.slider("Rolling Avg 3h (approx)", 0.0, float(df['rolling_avg_3h'].max()), float(df['rolling_avg_3h'].mean()))

if st.button("🔮 Predict Energy Usage"):
    user_input = np.array([[input_hour, input_day, input_month, input_rolling]])
    prediction = model.predict(user_input)[0]
    st.success(f"📌 Predicted Global Active Power: **{prediction:.3f} kW**")

    threshold = y_train.mean() + y_train.std()
    if prediction > threshold:
        st.warning("⚡ High usage predicted! Consider reducing load.")
    else:
        st.info("✅ Normal usage predicted.")

# ------------------------
# Download filtered data
# ------------------------

def to_excel_download(df_df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df_df.to_excel(writer, index=False, sheet_name="FilteredData")
    writer.save()
    return output.getvalue()

st.markdown("## 📥 Download Data")
download_data = to_excel_download(df_filtered)
st.download_button(
    label="Download Filtered Data (Excel)",
    data=download_data,
    file_name="filtered_energy_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
