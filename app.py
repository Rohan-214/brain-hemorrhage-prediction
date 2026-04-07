import streamlit as st
import joblib
import numpy as np

# Page Config
st.set_page_config(
    page_title="Hemorrhage Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("hemorrhage_model.pkl")

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧾 Patient Info")

age = st.sidebar.number_input("Age", 20, 100, 40)
systolic_bp = st.sidebar.number_input("Systolic BP", 80, 250, 120)
gcs_total = st.sidebar.number_input("GCS Score", 3, 15, 15)

seizure = st.sidebar.selectbox("Seizure", ["No", "Yes"])
vomiting = st.sidebar.selectbox("Vomiting", ["No", "Yes"])
afib = st.sidebar.selectbox("Atrial Fibrillation", ["No", "Yes"])

# Convert values
seizure = 1 if seizure == "Yes" else 0
vomiting = 1 if vomiting == "Yes" else 0
afib = 1 if afib == "Yes" else 0

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align:center;'>🏥 Brain Hemorrhage Risk Dashboard</h1>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------- KPI CARDS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("🧓 Age", age)
col2.metric("💓 BP", f"{systolic_bp} mmHg")
col3.metric("🧠 GCS", gcs_total)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🔍 Run Diagnosis"):

    input_data = np.array([[age, systolic_bp, gcs_total, seizure, vomiting, afib]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    risk_percent = probability * 100

    # Risk Level
    if risk_percent < 30:
        color = "#28a745"
        level = "Mild"
    elif risk_percent < 70:
        color = "#ffc107"
        level = "Moderate"
    else:
        color = "#dc3545"
        level = "Severe"

    # ---------------- MAIN DASHBOARD ----------------
    left, right = st.columns([2, 1])

    # LEFT PANEL (RESULT)
    with left:
        st.markdown("## 📊 Risk Analysis")

        st.progress(int(risk_percent))

        st.markdown(
            f"""
            <div style='padding:25px;border-radius:15px;background-color:#f8f9fa'>
                <h2 style='color:{color}'>Risk Level: {level}</h2>
                <h3>Probability: {risk_percent:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        if prediction == 1:
            st.error("🔴 High Risk Detected")
        else:
            st.success("🟢 Low Risk")

    # RIGHT PANEL (DETAILS)
    with right:
        st.markdown("## 🧾 Clinical Flags")

        st.write(f"Seizure: {'Yes' if seizure else 'No'}")
        st.write(f"Vomiting: {'Yes' if vomiting else 'No'}")
        st.write(f"Atrial Fibrillation: {'Yes' if afib else 'No'}")

        st.markdown("---")

        st.markdown("## 🩺 Recommendation")

        if level == "Mild":
            st.success("Routine monitoring recommended.")
        elif level == "Moderate":
            st.warning("Consult neurologist.")
        else:
            st.error("Immediate hospitalization required!")

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <center>⚠️ This system is AI-assisted and not a substitute for medical professionals.</center>
    """,
    unsafe_allow_html=True
)
