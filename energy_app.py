import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

st.set_page_config(page_title="⚡ Household Energy Optimizer", layout="wide", page_icon="⚡")

st.markdown("<h1 style='text-align: center; color: #ff6600;'>⚡ Household Energy Optimizer Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload your ZIP dataset", type="zip")
sample_frac = st.sidebar.slider("Sample Fraction (%)", 5, 100, 20)

if uploaded_file is not None:
    with zipfile.ZipFile(uploaded_file) as z:
        file_name = z.namelist()[0]
        with z.open(file_name) as f:
            df = pd.read_csv(f, sep=';', na_values='?')

    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    numeric_cols = ['Global_active_power','Global_reactive_power','Voltage',
                    'Sub_metering_1','Sub_metering_2','Sub_metering_3']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    df = df.sample(frac=sample_frac/100, random_state=42).sort_values('datetime')

    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['rolling_avg_3h'] = df['Global_active_power'].rolling(window=3).mean().bfill()
    df_clean = df.dropna(subset=['Global_active_power','rolling_avg_3h','hour','day_of_week','month'])

    scaler = MinMaxScaler()
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

    X = df_clean[['hour','day_of_week','month','rolling_avg_3h']]
    y = df_clean['Global_active_power']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    threshold = y_train.mean() + y_train.std()
    high_usage_hours = X_test.loc[y_pred > threshold]

    # Metrics cards
    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:.4f}")
    col2.metric("High Usage Threshold", f"{threshold:.4f}")

    # High usage table
    st.markdown("### ⚡ High Energy Usage Predicted")
    st.dataframe(high_usage_hours.head(10), use_container_width=True)

    # Interactive Plotly line chart
    st.markdown("### 📈 Energy Usage Sample Plot")
    fig = px.line(df_clean.head(1000), x='datetime', y='Global_active_power', 
                  labels={"datetime":"Datetime", "Global_active_power":"Global Active Power (scaled)"},
                  title="Energy Usage Sample")
    st.plotly_chart(fig, use_container_width=True)

    # Prediction section
    st.markdown("### 🔮 Predict Energy Usage for Custom Time")
    with st.expander("Enter input values"):
        input_hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)
        input_day = st.number_input("Day of Week (0=Monday, 6=Sunday)", min_value=0, max_value=6, value=2)
        input_month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=6)
        input_rolling = st.number_input("Rolling Avg 3h (scaled 0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

        if st.button("Predict"):
            custom_input = np.array([[input_hour, input_day, input_month, input_rolling]])
            prediction = model.predict(custom_input)[0]
            st.write(f"⚡ Predicted Global Active Power (scaled): {prediction:.4f}")
            if prediction > threshold:
                st.warning("High energy usage predicted! ⚡ Consider reducing load.")
            else:
                st.success("Energy usage normal ✅")
