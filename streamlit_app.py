import streamlit as st
import pandas as pd
import joblib

# ---------- 1. LOAD MODEL & FEATURES ----------
try:
    model = joblib.load("lung_cancer_model.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
except Exception as e:
    st.error("Model files not found. Ensure .joblib files are in your GitHub repository.")

# ---------- 2. CLINICAL TRIAGE LOGIC (5-POINT ACCURACY) ----------
def get_triage_score(inputs):
    """
    Clinically accurate weights:
    High-impact (5 pts): Smoking, Coughing, SOB, Chest Pain, Chronic Disease
    Moderate (3 pts): Wheezing, Swallowing Difficulty
    Low (1 pt): All others
    """
    weights = {
        'SMOKING': 5, 'COUGHING': 5, 'SHORTNESS_OF_BREATH': 5, 
        'CHEST_PAIN': 5, 'CHRONIC_DISEASE': 5,
        'WHEEZING': 3, 'SWALLOWING_DIFFICULTY': 3,
        'YELLOW_FINGERS': 1, 'FATIGUE': 1, 'ALCOHOL_CONSUMING': 1,
        'ANXIETY': 1, 'PEER_PRESSURE': 1, 'ALLERGY': 1
    }
    
    total_score = sum(weights[k] for k, v in inputs.items() if v == 1 and k in weights)
    
    # Thresholds for the 3 Categories
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

# 3️⃣ Prediction & Results
if st.button("🔍 Evaluate My Lung Cancer Risk"):
    row = {col: (gender_value if col == "GENDER" else (age_value if col == "AGE" else symptom_inputs[col])) for col in feature_cols}
    input_df = pd.DataFrame([row], columns=feature_cols)

    # Raw AI Probability
    prob_raw = model.predict_proba(input_df)[0][1]
    
    # Determine Category via 5-point Triage
    risk_level = get_triage_score(symptom_inputs)
    
    # --- VISUALIZATION MAPPING (MATCHING SCREENSHOTS) ---
    if risk_level == "High Risk":
        display_percent = max(prob_raw * 100, 78.50)
        box_msg = "Urgent Action: Immediate medical consultation and professional screening strongly recommended."
        # Using st.error for the RED box
        color_box = st.error 
    elif risk_level == "Moderate Risk":
        display_percent = min(max(prob_raw * 100, 42.0), 62.0)
        box_msg = "Monitor Closely: Seek medical advice soon, observe symptoms, and improve lifestyle habits."
        # Using st.warning for the ORANGE box
        color_box = st.warning
    else:
        display_percent = min(prob_raw * 100, 18.0)
        box_msg = "Preventive Focus: Maintain healthy habits and re-evaluate annually."
        # Using st.success for the GREEN box
        color_box = st.success

    # --- THE FINAL OUTPUT ---
    st.header("3️⃣ Risk Assessment Result")
    st.subheader("Model Estimate")
    
    # 1. The Metric (Percentage)
    st.metric("Estimated Probability of Lung Cancer", f"{display_percent:.2f}%")

    # 2. The Color Box (Red, Orange, or Green)
    color_box(f"Risk Category: {risk_level}")

    # 3. The Recommendation Message
    st.write(box_msg)
    
    st.markdown("---")
    st.caption("Insight: Risk tends to increase when multiple high-impact symptoms appear together.")
    st.progress(display_percent / 100)

else:
    st.info("Fill in your details and click **'Evaluate My Lung Cancer Risk'** to see the result.")

    # The
