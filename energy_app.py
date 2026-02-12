import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px

st.set_page_config(
    page_title="Household Energy Optimizer",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Household Energy Consumption Optimizer")
st.markdown("Fast • Optimized • Streamlit Ready")

@st.cache_data
def load_and_prepare_data(uploaded_file):
    with zipfile.ZipFile(uploaded_file) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f, sep=';', na_values='?')

    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        dayfirst=True,
        errors='coerce'
    )

    cols = [
        'Global_active_power','Global_reactive_power',
        'Voltage','Sub_metering_1',
        'Sub_metering_2','Sub_metering_3'
    ]
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    df = df.sample(frac=0.05, random_state=42).sort_values('datetime')

    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['rolling_avg'] = df['Global_active_power'].rolling(3).mean().bfill()

    return df.dropna()

uploaded_file = st.file_uploader(
    "📂 Upload household energy ZIP dataset",
    type="zip"
)

if uploaded_file:
    df = load_and_prepare_data(uploaded_file)

    col1, col2, col3 = st.columns(3)
    col1.metric("Records Used", f"{len(df):,}")
    col2.metric("Avg Power (kW)", f"{df['Global_active_power'].mean():.2f}")
    col3.metric("Peak Power (kW)", f"{df['Global_active_power'].max():.2f}")

    X = df[['hour','day','month','rolling_avg']]
    y = df['Global_active_power']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if st.button("🚀 Train Energy Model"):
        model = RandomForestRegressor(
            n_estimators=30,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        st.success(f"Model trained successfully | RMSE: {rmse:.3f}")

        threshold = y_train.mean() + y_train.std()
        high_usage = X_test[preds > threshold]

        st.subheader("⚠️ Predicted High Energy Usage Periods")
        st.dataframe(high_usage.head(10))

        st.subheader("📈 Energy Consumption Trend")
        fig = px.line(
            df.head(500),
            x="datetime",
            y="Global_active_power",
            title="Energy Usage Sample"
        )
        st.plotly_chart(fig, use_container_width=True)
