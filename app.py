# ===============================
# PREDICTION PAGE
# ===============================
elif page == "Prediction":
    st.header("🧮 Energy Prediction Input")

    col1, col2 = st.columns(2)

    with col1:
        kitchen = st.number_input("🍳 Kitchen Power (kW)", 0.0, value=4.5)
        laundry = st.number_input("🧺 Laundry Power (kW)", 0.0, value=6.0)
        other = st.number_input("💡 Other Usage (kW)", 0.0, value=3.0)

    with col2:
        extra = st.number_input("🔥 Extra Loss (kW)", 0.0, value=2.0)
        voltage = st.number_input("⚡ Voltage (V)", 0.0, value=220.0)
        current = st.number_input("🔁 Current (A)", 0.0, value=4.5)

    st.markdown("---")

    duration = st.slider(
        "⏱️ Duration of Usage (Hours)",
        min_value=0.5,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    if st.button("🚀 Calculate Energy Consumption", use_container_width=True):
        # ✅ FIXED: store inputs correctly in session_state
        st.session_state.kitchen = kitchen
        st.session_state.laundry = laundry
        st.session_state.other = other
        st.session_state.extra = extra
        st.session_state.duration = duration

        # Calculate total power and total units
        total_power = kitchen + laundry + other + extra
        total_units = total_power * duration  #*
