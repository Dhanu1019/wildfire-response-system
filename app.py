import streamlit as st
import joblib
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------- Page ----------
st.set_page_config(page_title="Wildfire Response System", page_icon="🔥")

st.title("🔥 AI Wildfire Response System")

# =====================================================
# PART 1 — IMAGE-BASED DETECTION
# =====================================================

st.header("🛰️ Wildfire Detection from Satellite Image")

image_model = joblib.load("wildfire_model.pkl")

base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(224,224,3)
)

uploaded_file = st.file_uploader("Upload Satellite Image",
                                 type=["jpg","png","jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB").resize((224,224))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    features = base_model.predict(img_array)
    prediction = image_model.predict(features)

    if prediction[0] == 1:
        st.error("🔥 WILDFIRE DETECTED")
    else:
        st.success("✅ No Wildfire Detected")

st.divider()

# =====================================================
# PART 2 — WEATHER-BASED PREDICTION
# =====================================================

st.header("🌍 Wildfire Risk Warning System (Weather-Based)")

st.markdown("Enter current environmental conditions:")

temp = st.number_input("🌡 Temperature (°C)", 0.0, 60.0, 35.0)
humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 30.0)
wind = st.number_input("🌬 Wind Speed (km/h)", 0.0, 150.0, 20.0)
rain = st.number_input("🌧 Rainfall (mm)", 0.0, 50.0, 0.0)

if st.button("🚨 Analyze Wildfire Risk"):

    # 🔥 Advanced risk formula
    risk_score = (temp * 0.6) + (wind * 0.7) + ((100 - humidity) * 0.6) - (rain * 2)

    # Normalize score to 0–100
    risk_score = max(0, min(100, risk_score))

    st.subheader("🔥 Fire Danger Index")

    # 📊 Risk Gauge (Progress Bar)
    st.progress(int(risk_score))

    st.write(f"Risk Score: {risk_score:.1f} / 100")

    # =========================
    # RISK LEVEL OUTPUT
    # =========================

    if risk_score >= 70:

        st.error("🔥 EXTREME WILDFIRE RISK")

        st.markdown("""
        ### 🚨 Immediate Action Required
        - 🚒 Deploy firefighting resources
        - 🏃 Initiate evacuation readiness
        - 📢 Alert emergency authorities
        - 🚫 Ban open fires and outdoor burning
        """)

    elif risk_score >= 45:

        st.warning("⚠️ HIGH WILDFIRE RISK")

        st.markdown("""
        ### ⚠️ Dangerous Conditions
        - Avoid campfires and sparks
        - Monitor local alerts
        - Prepare emergency kits
        """)

    elif risk_score >= 25:

        st.info("🟠 MODERATE RISK")

        st.markdown("""
        ### 🟡 Conditions Favor Fire Spread
        - Stay cautious outdoors
        - Monitor weather updates
        """)

    else:

        st.success("✅ LOW WILDFIRE RISK")

        st.markdown("""
        ### 🌲 Conditions Stable
        - Routine monitoring sufficient
        - Low likelihood of wildfire ignition
        """)