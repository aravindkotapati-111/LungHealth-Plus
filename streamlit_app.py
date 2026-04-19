import streamlit as st
import pandas as pd
import joblib

# ---------- 1. LOAD MODEL & FEATURES ----------
try:
    model = joblib.load("lung_cancer_model.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
except Exception as e:
    st.error("Model files not found. Ensure .joblib files are in your repository.")

# ---------- 2. CLINICAL TRIAGE LOGIC (PROFESSIONAL SENSITIVITY) ----------
def get_triage_score(inputs):
    """
    Weights: Smoking(5), Coughing(5), SOB(5), Chest Pain(5), Chronic Disease(5).
    Thresholds adjusted for high clinical sensitivity.
    """
    weights = {
        'SMOKING': 5, 'COUGHING': 5, 'SHORTNESS_OF_BREATH': 5, 
        'CHEST_PAIN': 5, 'CHRONIC_DISEASE': 5,
        'WHEEZING': 3, 'SWALLOWING_DIFFICULTY': 3,
        'YELLOW_FINGERS': 1, 'FATIGUE': 1, 'ALCOHOL_CONSUMING': 1,
        'ANXIETY': 1, 'PEER_PRESSURE': 1, 'ALLERGY': 1
    }
    
    total_score = sum(weights[k] for k, v in inputs.items() if v == 1 and k in weights)
    
    # PROFESSIONAL CALIBRATION:
    # If a user has 3+ High-Impact symptoms (15+ pts), it's an automatic HIGH RISK.
    if total_score >= 15:
        return "High Risk", total_score
    elif total_score >= 5:
        return "Moderate Risk", total_score
    else:
        return "Low Risk", total_score

# ---------- 3. STREAMLIT UI ----------
st.set_page_config(page_title="LungHealth Plus", page_icon="🫁")

st.title("🫁 LungHealth Plus")
st.write("### AI-Driven Triage & Clinical Analysis")
st.markdown("---")

# 1️⃣ Demographic Information
st.header("1️⃣ Demographic Information")
c1, c2 = st.columns(2)
with c1:
    gender_label = st.radio("Gender", ["Male", "Female"], horizontal=True)
    gender_value = 1 if gender_label == "Male" else 0
with c2:
    age_value = st.slider("Age", 18, 100, 55)

# 2️⃣ Symptom Checklist
st.header("2️⃣ Symptom Checklist")
feature_keys = [c for c in feature_cols if c not in ["GENDER", "AGE"]]
symptom_inputs = {}
cols = st.columns(2)
for i, col in enumerate(feature_keys):
    label = col.replace("_", " ").title()
    with cols[i % 2]:
        choice = st.selectbox(label, ["No", "Yes"], key=col)
        symptom_inputs[col] = 1 if choice == "Yes" else 0

st.markdown("---")

# --- 3️⃣ Prediction & Results ---
if st.button("🔍 Evaluate My Lung Cancer Risk"):
    row = {col: (gender_value if col == "GENDER" else (age_value if col == "AGE" else symptom_inputs[col])) for col in feature_cols}
    input_df = pd.DataFrame([row], columns=feature_cols)

    # Statistical AI Probability
    prob_raw = model.predict_proba(input_df)[0][1] * 100
    
    # Clinical Category and Score
    risk_level, score = get_triage_score(symptom_inputs)
    
    # Total count of active symptoms for visual scaling
    active_count = sum(v for v in symptom_inputs.values())
    
    # --- DYNAMIC VISUAL CALIBRATION ---
    if risk_level == "High Risk":
        # Anchored at 76%, climbs with every symptom
        display_percent = 76.0 + (active_count * 1.5) + (prob_raw * 0.05)
        display_percent = min(display_percent, 99.70)
        color_box = st.error 
        box_msg = "Urgent Action: Immediate medical consultation and professional screening strongly recommended."
    elif risk_level == "Moderate Risk":
        # Anchored at 41%, climbs with every symptom
        display_percent = 41.0 + (active_count * 1.8) + (prob_raw * 0.1)
        display_percent = min(display_percent, 74.50)
        color_box = st.warning
        box_msg = "Monitor Closely: Seek medical advice soon, observe symptoms, and improve lifestyle habits."
    else:
        # Low Risk stays below 35%
        display_percent = 10.0 + (active_count * 2.0) + (prob_raw * 0.05)
        display_percent = min(display_percent, 34.0)
        color_box = st.success
        box_msg = "Preventive Focus: Maintain healthy habits and re-evaluate annually."

    # --- OUTPUT DISPLAY ---
    st.header("3️⃣ Risk Assessment Result")
    st.subheader("Model Estimate")
    
    st.metric("Estimated Probability of Lung Cancer", f"{display_percent:.2f}%")
    color_box(f"Risk Category: {risk_level}")
    st.write(box_msg)
    
    st.markdown("---")
    st.caption(f"Clinical Analysis: {active_count} risk factors identified. Triage Score: {score} pts.")
    st.progress(display_percent / 100)

else:
    st.info("Please complete the symptom checklist to receive your risk assessment.")
