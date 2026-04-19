import streamlit as st
import pandas as pd
import joblib

# ---------- LOAD MODEL & FEATURES ----------
model = joblib.load("lung_cancer_model.joblib")
feature_cols = joblib.load("feature_columns.joblib")

def classify_risk(probability: float):
    """
    Map model probability to Low / Moderate / High risk.
    """
    if probability < 0.33:
        return (
            "Low Risk",
            "Preventive Focus: Maintain healthy habits and re-evaluate annually."
        )
    elif probability < 0.66:
        return (
            "Moderate Risk",
            "Monitor Closely: Seek medical advice soon, observe symptoms, and improve lifestyle habits."
        )
    else:
        return (
            "High Risk",
            "Urgent Action: Immediate medical consultation and professional screening strongly recommended."
        )

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Lung Cancer Risk Screener", page_icon="🫁")

st.title("🫁 Lung Cancer Risk Screening")
st.write(
    """
This web application uses your **symptoms and demographics** with a machine learning
model trained on the *survey lung cancer* dataset (309 respondents) to estimate
the probability of lung cancer.

⚠️ **Disclaimer:** This is for education and awareness only. It is **not** a medical diagnosis.
Always consult a healthcare professional.
"""
)

st.markdown("---")

# 1️⃣ Demographic inputs
st.header("1️⃣ Demographic Information")

gender_label = st.radio("Gender", ["Male", "Female"], horizontal=True)
gender_value = 1 if gender_label == "Male" else 0

age_value = st.slider("Age", min_value=18, max_value=100, value=55)

# 2️⃣ Symptom inputs
st.header("2️⃣ Symptom Checklist")

binary_cols = [c for c in feature_cols if c not in ["GENDER", "AGE"]]

symptom_inputs = {}
for col in binary_cols:
    label = col.replace("_", " ").strip().title()  # prettier label
    choice = st.selectbox(
        label,
        options=["No", "Yes"],
        index=0,
        key=col
    )
    symptom_inputs[col] = 1 if choice == "Yes" else 0

st.markdown("---")

# 3️⃣ Prediction
if st.button("🔍 Evaluate My Lung Cancer Risk"):
    # Build input data in correct column order
    row = {}
    for col in feature_cols:
        if col == "GENDER":
            row[col] = gender_value
        elif col == "AGE":
            row[col] = age_value
        else:
            row[col] = symptom_inputs[col]

    input_df = pd.DataFrame([row], columns=feature_cols)

    # Model probability for LUNG_CANCER = 1 (YES)
    prob_yes = model.predict_proba(input_df)[0][1]
    prob_percent = prob_yes * 100

    risk_level, message = classify_risk(prob_yes)

    st.header("3️⃣ Risk Assessment Result")
    st.subheader("Model Estimate")

    st.metric("Estimated Probability of Lung Cancer", f"{prob_percent:.2f}%")

    if risk_level == "Low Risk":
        st.success(f"Risk Category: {risk_level}")
    elif risk_level == "Moderate Risk":
        st.warning(f"Risk Category: {risk_level}")
    else:
        st.error(f"Risk Category: {risk_level}")

    st.write(message)

    st.caption(
        "Insight: Risk tends to increase when multiple high-impact symptoms (smoking, chest pain, shortness of breath, chronic disease, etc.) appear together."
    )

    # Visual probability bar
    st.progress(min(max(prob_yes, 0.0), 1.0))
else:
    st.info("Fill in your details above and click **'Evaluate My Lung Cancer Risk'** to see your result.")
