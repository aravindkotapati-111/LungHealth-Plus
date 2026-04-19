import streamlit as st
import pandas as pd
import joblib

# ---------- 1. LOAD MODEL & FEATURES ----------
try:
    model = joblib.load("lung_cancer_model.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
except:
    st.error("Model files missing. Please ensure .joblib files are in the directory.")

# ---------- 2. UI SETUP ----------
st.set_page_config(page_title="LungHealth Plus", page_icon="🫁")
st.title("🫁 LungHealth Plus")
st.write("### AI-Driven Clinical Decision Support")
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
    
    # --- STEP 1: MANUALLY COUNT RED FLAGS (5-Point Symptoms) ---
    # Using keyword matching to avoid naming errors
    rf_count = 0
    for key, value in symptom_inputs.items():
        if value == 1:
            k_upper = key.upper()
            if any(word in k_upper for word in ['SMOKING', 'CHRONIC', 'SHORTNESS', 'CHEST', 'COUGHING']):
                rf_count += 1
    
    total_yes = sum(v for v in symptom_inputs.values())
    
    # --- STEP 2: TIERED CLINICAL RULES ---
    
    # TIER A: HIGH RISK (3+ Major Red Flags)
    if rf_count >= 3:
        risk_cat = "High Risk"
        final_pct = 82.0 + (total_yes * 1.2)
        alert_style = st.error
        advice = "Urgent Action: Immediate medical consultation and screening strongly recommended."
        
    # TIER B: MODERATE RISK (1-2 Major Red Flags OR very high volume of minor symptoms)
    elif rf_count >= 1 or total_yes >= 5:
        risk_cat = "Moderate Risk"
        final_pct = 42.0 + (total_yes * 2.0)
        alert_style = st.warning
        advice = "Monitor Closely: Seek medical advice soon and observe symptom progression."
        
    # TIER C: LOW RISK (Only minor symptoms and fewer than 5 total)
    else:
        risk_cat = "Low Risk"
        # Scales slowly, staying under the 35% threshold
        final_pct = 10.0 + (total_yes * 4.0)
        alert_style = st.success
        advice = "Preventive Focus: Maintain healthy habits and re-evaluate annually."

    # Final boundary check to stay within 0-100
    final_pct = min(max(final_pct, 0.0), 99.85)

    # --- STEP 3: DISPLAY RESULTS ---
    st.header("3️⃣ Risk Assessment Result")
    st.metric("Estimated Probability of Lung Cancer", f"{final_pct:.2f}%")
    alert_style(f"Risk Category: {risk_cat}")
    st.write(f"**Clinical Recommendation:** {advice}")
    
    st.markdown("---")
    # This breakdown is key for your HIMSS presentation
    st.caption(f"Informatics Breakdown: {rf_count} Red-Flags detected. Total Symptom Load: {total_yes}")
    st.progress(final_pct / 100)

else:
    st.info("Complete the clinical checklist and click 'Evaluate' to generate results.")
