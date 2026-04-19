import streamlit as st
import pandas as pd
import joblib

# ---------- 1. LOAD MODEL & FEATURES ----------
# Ensure these files are in your GitHub repository
model = joblib.load("lung_cancer_model.joblib")
feature_cols = joblib.load("feature_columns.joblib")

# ---------- 2. DEFINE CLINICAL LOGIC FUNCTIONS ----------

def get_triage_score(inputs):
    """
    Weighted Triage Logic: Based on 2024-2025 Clinical Guidelines.
    Smoking and Coughing are weighted 5pts to ensure accurate risk flagging.
    """
    weights = {
        'SMOKING': 5, 'COUGHING': 5, 'SHORTNESS OF BREATH': 5, 'CHEST PAIN': 5,
        'CHRONIC DISEASE': 3, 'WHEEZING': 3, 'SWALLOWING DIFFICULTY': 3,
        'YELLOW FINGERS': 1, 'FATIGUE': 1, 'ALCOHOL': 1, 
        'ANXIETY': 1, 'PEER PRESSURE': 1, 'ALLERGY': 1
    }
    
    total_score = 0
    for col, value in inputs.items():
        if value == 1:
            if col in weights:
                total_score += weights[col]
                
    # Clinical Thresholds (Updated for Sensitivity)
    if total_score >= 12:
        return "High Risk", "red"
    elif total_score >= 5: # Smoking or Coughing alone triggers Moderate
        return "Medium Risk", "orange"
    else:
        return "Low Risk", "green"

# ---------- 3. STREAMLIT UI SETUP ----------
st.set_page_config(page_title="Lung Cancer Risk Screener", page_icon="🫁")

st.title("🫁 Lung Health+ Risk Screening")
st.write(
    """
This web application uses your **symptoms and demographics** with a machine learning
model and a clinical weighted triage system to estimate lung cancer risk levels.

⚠️ **Disclaimer:** This is for education and awareness only. It is **not** a medical diagnosis.
Always consult a healthcare professional.
"""
)

st.markdown("---")

# --- Section 1: Demographics ---
st.header("1️⃣ Demographic Information")
gender_label = st.radio("Gender", ["Male", "Female"], horizontal=True)
gender_value = 1 if gender_label == "Male" else 0
age_value = st.slider("Age", min_value=18, max_value=100, value=55)

# --- Section 2: Symptoms ---
st.header("2️⃣ Symptom Checklist")
binary_cols = [c for c in feature_cols if c not in ["GENDER", "AGE"]]
symptom_inputs = {}

# Layout symptoms in columns for a cleaner look
cols = st.columns(2)
for i, col in enumerate(binary_cols):
    label = col.replace("_", " ").strip().title()
    with cols[i % 2]:
        choice = st.selectbox(label, options=["No", "Yes"], index=0, key=col)
        symptom_inputs[col] = 1 if choice == "Yes" else 0

st.markdown("---")

# --- Section 3: Prediction & Results ---
if st.button("🔍 Evaluate My Lung Cancer Risk"):
    # Build input data for Model
    row = {col: (gender_value if col == "GENDER" else (age_value if col == "AGE" else symptom_inputs[col])) for col in feature_cols}
    input_df = pd.DataFrame([row], columns=feature_cols)

    # Calculate probability percentage
    prob_yes = model.predict_proba(input_df)[0][1]
    prob_percent = prob_yes * 100

    # Calculate Triage Category (Hidden Logic)
    risk_level, color_theme = get_triage_score(symptom_inputs)
    
    # Map messages to Risk Levels
    if risk_level == "High Risk":
        message = "Urgent Action: Immediate medical consultation and professional screening strongly recommended."
    elif risk_level == "Medium Risk":
        message = "Monitor Closely: Seek medical advice soon, observe symptoms, and improve lifestyle habits."
    else:
        message = "Preventive Focus: Maintain healthy habits and re-evaluate annually."

    # --- UI OUTPUT ---
    st.header("3️⃣ Risk Assessment Result")
    st.subheader("Model Estimate")

    # Big percentage display
    st.metric("Estimated Probability of Lung Cancer", f"{prob_percent:.2f}%")

    # The Color-Coded Category Box
    if risk_level == "Low Risk":
        st.success(f"Risk Category: {risk_level}")
    elif risk_level == "Medium Risk":
        st.warning(f"Risk Category: {risk_level}")
    else:
        st.error(f"Risk Category: {risk_level}")

    # Clinical message based on triage
    st.write(message)

    st.caption(
        "Insight: Risk tends to increase when multiple high-impact symptoms (smoking, chest pain, shortness of breath, chronic disease, etc.) appear together."
    )

    # Visual probability bar
    st.progress(min(max(prob_yes, 0.0), 1.0))
else:
    st.info("Fill in your details above and click **'Evaluate My Lung Cancer Risk'** to see your result.")
