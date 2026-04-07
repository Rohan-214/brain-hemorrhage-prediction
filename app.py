import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Brain Hemorrhage Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("hemorrhage_model.pkl")

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧾 Patient Details")

age = st.sidebar.number_input("Age", 20, 100, 40)
systolic_bp = st.sidebar.number_input("Systolic BP (mmHg)", 80, 250, 120)

# ----------- GCS INPUT (NEW) -----------
st.sidebar.markdown("### 🧠 Glasgow Coma Scale (GCS)")

eye = st.sidebar.selectbox(
    "👁️ Eye Opening",
    [
        ("4 - Spontaneous", 4),
        ("3 - To Speech", 3),
        ("2 - To Pain", 2),
        ("1 - No Response", 1)
    ],
    format_func=lambda x: x[0]
)[1]

verbal = st.sidebar.selectbox(
    "🗣️ Verbal Response",
    [
        ("5 - Oriented", 5),
        ("4 - Confused", 4),
        ("3 - Inappropriate Words", 3),
        ("2 - Incomprehensible Sounds", 2),
        ("1 - No Response", 1)
    ],
    format_func=lambda x: x[0]
)[1]

motor = st.sidebar.selectbox(
    "💪 Motor Response",
    [
        ("6 - Obeys Commands", 6),
        ("5 - Localizes Pain", 5),
        ("4 - Withdraws from Pain", 4),
        ("3 - Abnormal Flexion", 3),
        ("2 - Abnormal Extension", 2),
        ("1 - No Response", 1)
    ],
    format_func=lambda x: x[0]
)[1]

# Calculate total GCS
gcs_total = eye + verbal + motor

# ----------- OTHER INPUTS -----------
seizure = st.sidebar.selectbox("Seizure", ["No", "Yes"])
vomiting = st.sidebar.selectbox("Vomiting", ["No", "Yes"])
afib = st.sidebar.selectbox("Atrial Fibrillation", ["No", "Yes"])

# Convert to numeric
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
col1, col2, col3, col4 = st.columns(4)

col1.metric("🧓 Age", age)
col2.metric("💓 BP", f"{systolic_bp} mmHg")
col3.metric("🧠 GCS", gcs_total)
col4.metric("⚠️ Status", "Stable" if gcs_total >= 13 else "Critical")

# GCS Breakdown
st.markdown("### 🧠 GCS Breakdown")
g1, g2, g3 = st.columns(3)
g1.metric("👁️ Eye", eye)
g2.metric("🗣️ Verbal", verbal)
g3.metric("💪 Motor", motor)

# GCS Interpretation
if gcs_total >= 13:
    st.success("🟢 Mild Brain Injury")
elif gcs_total >= 9:
    st.warning("🟡 Moderate Brain Injury")
else:
    st.error("🔴 Severe Brain Injury")

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🔍 Run Diagnosis"):

    input_data = np.array([[age, systolic_bp, gcs_total, seizure, vomiting, afib]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    risk_percent = probability * 100

    # Risk Level Logic
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

    # LEFT PANEL
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
            st.error("🔴 High Hemorrhage Risk Detected")
        else:
            st.success("🟢 Low Hemorrhage Risk")

    # RIGHT PANEL
    with right:
        st.markdown("## 🧾 Clinical Summary")

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
