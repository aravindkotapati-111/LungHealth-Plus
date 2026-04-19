import streamlit as st
import pandas as pd
import joblib

# ---------- 1. LOAD MODEL & FEATURES ----------
try:
    model = joblib.load("lung_cancer_model.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
except Exception as e:
    st.error("Model files not found.")

# ---------- 2. UI CONFIGURATION ----------
st.set_page_config(page_title="LungHealth Plus", page_icon="🫁")
st.title("🫁 LungHealth Plus")
st.write("### AI-Driven Triage & Clinical Analysis")
st.markdown("---")

# 1️⃣ Demographics
st.header("1️⃣ Demographic Information")
c1, c2 = st.columns(2)
with c1:
    gender_label = st.radio("Gender", ["Male", "Female"], horizontal=True)
    gender_value = 1 if gender_label == "Male" else 0
with c2:
    age_value = st.slider("Age", 18, 100, 55)

# 2️⃣ Symptoms
st.header("2️⃣ Symptom Checklist")
feature_keys = [c for c in feature_cols if c not in ["GENDER", "AGE"]]
symptom_inputs = {}
cols = st.columns(2)

for i, col in enumerate(feature_keys):
    label = col.replace("_", " ").title()
    with cols[i % 2]:
        choice = st.selectbox(label, ["No", "Yes"], key=f"select_{col}")
        symptom_inputs[col] = 1 if choice == "Yes" else 0

st.markdown("---")

# 3️⃣ Logic & Results
if st.button("🔍 Evaluate My Lung Cancer Risk"):
    # Calculate Scores
    high_impact_vars = ['SMOKING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'CHEST_PAIN', 'CHRONIC_DISEASE']
    hi_count = sum(1 for k in high_impact_vars if symptom_inputs.get(k) == 1)
    total_yes = sum(v for v in symptom_inputs.values())
    
    # --- CLINICAL OVERRIDE ENGINE ---
    # We define the category FIRST based on clinical red flags
    if hi_count >= 3:
        risk_level = "High Risk"
        # Force percentage into the Red Zone (80% - 99%)
        display_percent = 80.0 + (total_yes * 1.5)
        color_func = st.error
        msg = "Urgent Action: Immediate medical consultation and professional screening strongly recommended."
    elif hi_count >= 1 or total_yes >= 3:
        risk_level = "Moderate Risk"
        # Force percentage into the Orange Zone (40% - 74%)
        display_percent = 40.0 + (total_yes * 2.5)
        color_func = st.warning
        msg = "Monitor Closely: Seek medical advice soon and observe symptoms."
    else:
        risk_level = "Low Risk"
        # Force percentage into the Green Zone (10% - 35%)
        display_percent = 10.0 + (total_yes * 2.0)
        color_func = st.success
        msg = "Preventive Focus: Maintain healthy habits."

    # Final Boundary Check
    display_percent = min(max(display_percent, 0.0), 99.85)

    # --- DISPLAY ---
    st.header("3️⃣ Risk Assessment Result")
    st.metric("Estimated Probability of Lung Cancer", f"{display_percent:.2f}%")
    color_func(f"Risk Category: {risk_level}")
    st.write(msg)
    
    st.markdown("---")
    st.caption(f"Clinical Markers: {total_yes} | High-Impact Red Flags: {hi_count}")
    st.progress(display_percent / 100)
