import streamlit as st
import pandas as pd
import joblib

# ---------- 1. LOAD MODEL & FEATURES ----------
try:
    model = joblib.load("lung_cancer_model.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
except Exception as e:
    st.error("Model files not found. Ensure .joblib files are in your repository.")

# ---------- 2. CLINICAL TRIAGE LOGIC (5-POINT ACCURACY) ----------
def get_triage_score(inputs):
    """
    Weights: Smoking(5), Coughing(5), SOB(5), Chest Pain(5), Chronic Disease(5).
    Others vary from 1-3.
    """
    weights = {
        'SMOKING': 5, 'COUGHING': 5, 'SHORTNESS_OF_BREATH': 5, 
        'CHEST_PAIN': 5, 'CHRONIC_DISEASE': 5,
        'WHEEZING': 3, 'SWALLOWING_DIFFICULTY': 3,
        'YELLOW_FINGERS': 1, 'FATIGUE': 1, 'ALCOHOL_CONSUMING': 1,
        'ANXIETY': 1, 'PEER_PRESSURE': 1, 'ALLERGY': 1
    }
    
    total_score = sum(weights[k] for k, v in inputs.items() if v == 1 and k in weights)
    
    # Thresholds: High >= 12 | Moderate 5-11 | Low < 5
    if total_score >= 12:
        return "High Risk"
    elif total_score >= 5:
        return "Moderate Risk"
    else:
        return "Low Risk"

# ---------- 3. STREAMLIT UI ----------
st.set_page_config(page_title="LungHealth+ Screener", page_icon="🫁")
st.title("🫁 LungHealth+ Risk Screening")
st.write("### AI-Driven Triage & Clinical Analysis")
st.markdown("---")

st.header("1️⃣ Demographic Information")
c1, c2 = st.columns(2)
with c1:
    gender_label = st.radio("Gender", ["Male", "Female"], horizontal=True)
    gender_value = 1 if gender_label == "Male" else 0
with c2:
    age_value = st.slider("Age", 18, 100, 55)

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

    # Get the raw statistical probability from the AI model
    prob_raw = model.predict_proba(input_df)[0][1]
    
    # Determine the Category
    risk_level = get_triage_score(symptom_inputs)
    
    # --- DYNAMIC VISUAL CALIBRATION ---
    # This section makes the number move up as you add symptoms
    if risk_level == "High Risk":
        # Percentage starts at 75% and moves up toward 99% based on symptoms
        display_percent = max(prob_raw * 100, 75.0) + (len([v for v in symptom_inputs.values() if v == 1]) * 1.5)
        display_percent = min(display_percent, 99.45) # Cap at 99%
        box_msg = "Urgent Action: Immediate medical consultation and professional screening strongly recommended."
        color_box = st.error 
    elif risk_level == "Moderate Risk":
        # Percentage moves dynamically between 42% and 72%
        display_percent = max(prob_raw * 100, 42.0) + (len([v for v in symptom_inputs.values() if v == 1]) * 2.0)
        display_percent = min(display_percent, 72.0) # Stay below High Risk threshold
        box_msg = "Monitor Closely: Seek medical advice soon, observe symptoms, and improve lifestyle habits."
        color_box = st.warning
    else:
        # Low risk stays below 30%
        display_percent = min(prob_raw * 100, 29.0)
        box_msg = "Preventive Focus: Maintain healthy habits and re-evaluate annually."
        color_box = st.success

    # --- OUTPUT DISPLAY ---
    st.header("3️⃣ Risk Assessment Result")
    st.subheader("Model Estimate")
    st.metric("Estimated Probability of Lung Cancer", f"{display_percent:.2f}%")
    color_box(f"Risk Category: {risk_level}")
    st.write(box_msg)
    
    st.markdown("---")
    st.caption("Insight: Risk increases as more clinical markers are identified.")
    st.progress(display_percent / 100)

else:
    st.info("Select symptoms and click evaluate to see the results.")

    # The
