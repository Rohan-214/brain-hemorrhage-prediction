import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Brain Hemorrhage Prediction", layout="centered")

# Load model
model = joblib.load("hemorrhage_model.pkl")

# Title
st.title("🧠 Early Brain Hemorrhage Risk Prediction System")

st.markdown("### Enter Patient Clinical Details")

# Layout in two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=40)
    systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=250, value=120)
    gcs_total = st.number_input("GCS Total Score", min_value=3, max_value=15, value=15)

with col2:
    seizure = st.selectbox("Seizure Present?", ["No", "Yes"])
    vomiting = st.selectbox("Vomiting Present?", ["No", "Yes"])
    afib = st.selectbox("Atrial Fibrillation?", ["No", "Yes"])

# Convert Yes/No to 0/1
seizure = 1 if seizure == "Yes" else 0
vomiting = 1 if vomiting == "Yes" else 0
afib = 1 if afib == "Yes" else 0

st.markdown("---")

if st.button("🔍 Predict Risk"):

    input_data = np.array([[age, systolic_bp, gcs_total, seizure, vomiting, afib]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    risk_percent = probability * 100

    st.markdown("## Prediction Result")

    # Risk Meter
    st.progress(int(risk_percent))

    if prediction == 1:
        st.error(f"🔴 High Hemorrhage Risk Detected")
        st.markdown(f"### Probability: **{risk_percent:.2f}%**")
    else:
        st.success(f"🟢 Low Hemorrhage Risk")
        st.markdown(f"### Probability: **{risk_percent:.2f}%**")

    # Risk Interpretation
    if risk_percent < 30:
        st.info("Risk Level: Mild")
    elif risk_percent < 70:
        st.warning("Risk Level: Moderate")
    else:
        st.error("Risk Level: Severe")

