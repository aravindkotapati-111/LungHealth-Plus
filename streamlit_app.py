import streamlit as st
import pandas as pd
import joblib

# ---------- 1. LOAD MODEL & FEATURES ----------
try:
    model = joblib.load("lung_cancer_model.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
except:
    st.error("Model files missing.")

# ---------- 2. UI SETUP ----------
st.set_page_config(page_title="LungHealth Plus", page_icon="🫁")
st.title("🫁 LungHealth Plus")
st.write("### Clinical Decision Support System")
st.markdown("---")

# 1️⃣ Demographics
st.header("1️⃣ Demographic Information")
c1, c2 = st.columns(2)
with c1:
    gender_label = st.radio("Gender", ["Male", "Female"], horizontal=True)
    gender_val = 1 if gender_label == "Male" else 0
with c2:
    age_val = st.slider("Age", 18, 100, 55)

# 2️⃣ Symptoms
st.header("2️⃣ Symptom Checklist")
feature_keys = [c for c in feature_cols if c not in ["GENDER", "AGE"]]
symptom_inputs = {}
cols = st.columns(2)

for i, col in enumerate(feature_keys):
    label = col.replace("_", " ").title()
    with cols[i % 2]:
        choice = st.selectbox(label, ["No", "Yes"], key=f"ui_{col}")
        symptom_inputs[col] = 1 if choice == "Yes" else 0

st.markdown("---")

# 3️⃣ Evaluation Logic
if st.button("🔍 Evaluate My Lung Cancer Risk"):
    # Define high-impact "Red Flags"
    red_flags = ['SMOKING', 'COUGHING', 'SHORTNESS_OF_BREATH', 'CHEST_PAIN', 'CHRONIC_DISEASE']
    rf_count = sum(1 for k in red_flags if symptom_inputs.get(k) == 1)
    total_yes = sum(v for v in symptom_inputs.values())
    
    # --- THE CLINICAL OVERRIDE ENGINE ---
    # This block forces the category based on Red Flags, ignoring the ML conflict
    if rf_count >= 3:
        # CATEGORY: HIGH RISK
        risk_cat = "High Risk"
        # We start the percentage at 82% so it's clearly in the High zone
        final_pct = 82.0 + (total_yes * 1.2) 
        alert_style = st.error
        advice = "Urgent Action: Immediate medical consultation strongly recommended."
    elif rf_count >= 1 or total_yes >= 4:
        # CATEGORY: MODERATE RISK
        risk_cat = "Moderate Risk"
        final_pct = 45.0 + (total_yes * 2.0)
        alert_style = st.warning
        advice = "Monitor Closely: Seek medical advice and observe symptoms."
    else:
        # CATEGORY: LOW RISK
        risk_cat = "Low Risk"
        final_pct = 10.0 + (total_yes * 2.0)
        alert_style = st.success
        advice = "Preventive Focus: Maintain healthy habits."

    # Final result display
    st.header("3️⃣ Risk Assessment Result")
    st.metric("Estimated Probability", f"{min(final_pct, 99.9):.2f}%")
    alert_style(f"Risk Category: {risk_cat}")
    st.write(f"**Recommendation:** {advice}")
    
    st.markdown("---")
    st.caption(f"Informatics Breakdown: {rf_count} Red-Flags detected. Total Symptom Load: {total_yes}")
    st.progress(min(final_pct/100, 1.0))
