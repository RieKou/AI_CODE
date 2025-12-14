# app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="TB Delay Prediction", layout="centered")

# Title
st.title("Tuberculosis Detection Delay Prediction - MVP")
st.write("Minimum Viable Product for Early Triage and Screening")

# Load model
try:
    model = joblib.load("model_pipeline.joblib")
except:
    st.error("Model 'model_pipeline.joblib' not found. Run train.py first.")
    st.stop()

# Form
st.header("Patient Information")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        sex = st.selectbox("Sex", ["M", "F"])
        education = st.selectbox("Education Level", ["none", "primary", "secondary", "university"])
        socio = st.selectbox("Socioeconomic Proxy", ["BPJS", "private", "none"])
        smoke = st.selectbox("Smoking Status", ["never", "former", "current"])

    with col2:
        cough_days = st.number_input("Cough Duration (days)", min_value=0, max_value=365, value=14)
        hemop = st.selectbox("Hemoptysis (batuk darah)", [0, 1])
        weight_loss = st.selectbox("Weight Loss", [0, 1])
        fever = st.selectbox("Night Sweats / Fever", [0, 1])
        contact = st.selectbox("Contact With TB Case", [0, 1])
        xray = st.selectbox("X-ray Findings", ["normal", "suspicious", "typical"])
        distance = st.number_input("Distance to Healthcare (km)", min_value=0.0, max_value=100.0, value=5.0)
        diabetes = st.selectbox("Diabetes", [0, 1])
        hiv = st.selectbox("HIV", [0, 1])

    submitted = st.form_submit_button("Predict Risk")

# Prediction
if submitted:
    X = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "education_level": education,
        "socioeconomic_proxy": socio,
        "cough_duration_days": cough_days,
        "hemoptysis": hemop,
        "weight_loss": weight_loss,
        "fever_night_sweats": fever,
        "smoking_status": smoke,
        "contact_with_TB_case": contact,
        "comorbidity_diabetes": diabetes,
        "comorbidity_HIV": hiv,
        "xray_findings": xray,
        "distance_to_healthcare_km": distance
    }])

    prob = model.predict_proba(X)[:, 1][0]
    risk_class = "HIGH RISK" if prob > 0.5 else "LOW / MEDIUM RISK"

    st.header("Prediction Result")
    st.subheader(f"Risk Category: **{risk_class}**")
    st.write(f"Probability of Long Delay: **{prob:.3f}**")

    # Recommendations
    st.header("Triage Recommendation")
    if cough_days > 14:
        st.write("- Cough >14 days detected → Recommend chest X-ray.")
    if xray == "suspicious":
        st.write("- Chest X-ray suspicious → Suggest GeneXpert test.")
    if contact == 1:
        st.write("- Close contact with TB case → Prioritize early screening.")
    if diabetes == 1 or hiv == 1:
        st.write("- Patient has comorbidity → Faster diagnosis recommended.")

    st.info("This MVP is for educational purposes and not a medical diagnostic tool.")
    