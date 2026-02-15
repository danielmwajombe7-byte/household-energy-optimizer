import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="⚡ Smart Energy Consumption AI",
    page_icon="💡",
    layout="wide"
)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ===============================
# SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("⚡ Energy AI App")
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Dashboard", "🤖 Prediction", "📈 Visualization", "ℹ️ About"]
)

# ===============================
# HOME PAGE
# ===============================
if menu == "🏠 Home":
    st.title("⚡ Smart Energy Consumption AI System")
    st.markdown("""
    💡 **AI-powered system for predicting electricity consumption**  

    ### 🎯 Objectives
    - Predict energy consumption based on user inputs  
    - Compare Machine Learning models  
    - Visualize power usage patterns  
    - Support smart energy decision making  

    👉 Use the sidebar to navigate.
    """)

# ===============================
# DASHBOARD
# ===============================
elif menu == "📊 Dashboard":
    st.title("📊 Energy Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("⚡ Voltage (V)", "220 V")
    col2.metric("🔌 Average Power", "3.5 kW")
    col3.metric("💡 Status", "Normal Usage")

    st.info("Dashboard overview showing general electricity indicators.")

# ===============================
# PREDICTION PAGE
# ===============================
elif menu == "🤖 Prediction":
    st.title("🤖 Energy Consumption Prediction")

    st.subheader("🔢 Enter Input Values")

    voltage = st.number_input("Electric Voltage (V)", min_value=180, max_value=260, value=220)
    current = st.number_input("Current Intensity (A)", min_value=0.1, value=5.0)
    kitchen = st.number_input("Kitchen Power Usage (kW)", min_value=0.0, value=1.2)
    laundry = st.number_input("Laundry Power Usage (kW)", min_value=0.0, value=0.8)
    extra_loss = st.number_input("Extra Power Loss (kW)", min_value=0.0, value=0.3)

    if st.button("⚡ Predict Energy Consumption"):
        input_data = np.array([[voltage, current, kitchen, laundry, extra_loss]])
        prediction = model.predict(input_data)

        st.success(f"🔮 **Predicted Total Power Used:** {prediction[0]:.2f} kW")

# ===============================
# VISUALIZATION PAGE
# ===============================
elif menu == "📈 Visualization":
    st.title("📈 Energy Consumption Visualization")

    graph_type = st.selectbox(
        "Select Graph Type",
        ["Bar Chart", "Line Graph", "Scatter Plot"]
    )

    sample_data = pd.DataFrame({
        "Category": ["Kitchen", "Laundry", "Extra Loss"],
        "Power (kW)": [1.2, 0.8, 0.3]
    })

    fig, ax = plt.subplots()

    if graph_type == "Bar Chart":
        ax.bar(sample_data["Category"], sample_data["Power (kW)"])
        ax.set_title("Power Usage by Category")

    elif graph_type == "Line Graph":
        ax.plot(sample_data["Category"], sample_data["Power (kW)"], marker="o")
        ax.set_title("Energy Consumption Trend")

    elif graph_type == "Scatter Plot":
        ax.scatter(sample_data["Category"], sample_data["Power (kW)"])
        ax.set_title("Power Distribution")

    ax.set_ylabel("Power (kW)")
    st.pyplot(fig)

# ===============================
# ABOUT PAGE
# ===============================
elif menu == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown("""
    **Course:** Machine Learning – Project Work (Test 2)  

    **Project Title:**  
    Development and Deployment of a Machine Learning–based AI Application  

    **Models Used:**  
    - Linear Regression  
    - Decision Tree  

    **Deployment:**  
    - Streamlit Cloud  

    ⚡ This project demonstrates how Machine Learning can support smart electricity usage in real-world Tanzanian contexts.
    """)
