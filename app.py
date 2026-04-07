import streamlit as st
import joblib
import numpy as np

# Page Config
st.set_page_config(
    page_title="Brain Hemorrhage Prediction",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🧠 Early Brain Hemorrhage Risk Prediction System")

st.markdown(
    """
    <div style='background-color:#f8f9fa;padding:15px;border-radius:10px'>
    ⚠️ <b>Disclaimer:</b> This tool is for educational purposes only and should NOT replace medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)

# Load model safely
@st.cache_resource
def load_model():
    return joblib.load("hemorrhage_model.pkl")

model = load_model()

st.markdown("### 📝 Enter Patient Clinical Details")

# Columns layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 40)
    systolic_bp = st.number_input("Systolic BP (mmHg)", 80, 250, 120)
    gcs_total = st.number_input("GCS Score", 3, 15, 15)

with col2:
    seizure = st.selectbox("Seizure", ["No", "Yes"])
    vomiting = st.selectbox("Vomiting", ["No", "Yes"])
    afib = st.selectbox("Atrial Fibrillation", ["No", "Yes"])

# Convert categorical to numeric
seizure = 1 if seizure == "Yes" else 0
vomiting = 1 if vomiting == "Yes" else 0
afib = 1 if afib == "Yes" else 0

st.markdown("---")

# Prediction Button
if st.button("🔍 Predict Risk"):

    # Validation check
    if gcs_total < 3 or gcs_total > 15:
        st.error("Invalid GCS Score")
    else:
        input_data = np.array([[age, systolic_bp, gcs_total, seizure, vomiting, afib]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        risk_percent = probability * 100

        st.markdown("## 📊 Prediction Result")

        # Progress bar
        st.progress(int(risk_percent))

        # Risk Color Logic
        if risk_percent < 30:
            color = "green"
            level = "Mild"
        elif risk_percent < 70:
            color = "orange"
            level = "Moderate"
        else:
            color = "red"
            level = "Severe"

        # Result Display
        st.markdown(
            f"""
            <div style='padding:20px;border-radius:10px;background-color:#f1f3f5'>
                <h3 style='color:{color}'>Risk Level: {level}</h3>
                <h4>Probability: {risk_percent:.2f}%</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        if prediction == 1:
            st.error("🔴 High Hemorrhage Risk Detected")
        else:
            st.success("🟢 Low Hemorrhage Risk")

        # Recommendations
        st.markdown("### 🩺 Recommendations")

        if level == "Mild":
            st.info("Maintain healthy lifestyle and monitor regularly.")
        elif level == "Moderate":
            st.warning("Consult a doctor and monitor symptoms closely.")
        else:
            st.error("Seek immediate medical attention!")
