import streamlit as st
import pandas as pd
import joblib

# ---------- 1. LOAD MODEL & FEATURES ----------
model = joblib.load("lung_cancer_model.joblib")
feature_cols = joblib.load("feature_columns.joblib")

# ---------- 2. DEFINE CLINICAL LOGIC FUNCTIONS ----------

def classify_risk(probability: float):
    """
    ML Logic: Maps model probability to risk categories.
    """
    if probability < 0.33:
        return "Low Risk", "Preventive Focus: Maintain healthy habits."
    elif probability < 0.66:
        return "Moderate Risk", "Monitor Closely: Seek medical advice soon."
    else:
        return "High Risk", "Urgent Action: Immediate medical consultation recommended."

def get_triage_score(inputs):
    """
    Triage Logic: Based on 2024-2025 Clinical Guidelines.
    Smoking and Coughing are weighted 5pts to ensure accurate risk flagging.
    """
    weights = {
        'SMOKING': 5, 'COUGHING': 5, 'SHORTNESS OF BREATH': 5, 'CHEST PAIN': 5,
        'CHRONIC DISEASE': 3, 'WHEEZING': 3, 'SWALLOWING DIFFICULTY': 3,
        'YELLOW FINGERS': 1, 'FATIGUE': 1, 'ALCOHOL': 1, 
        'ANXIETY': 1, 'PEER PRESSURE': 1, 'ALLERGY': 1
    }
    
    total_score = 0
    # Calculate score based on user selecting 'Yes' (value 1)
    for col, value in inputs.items():
        if value == 1:
            # Match the column name to the weights dictionary
            if col in weights:
                total_score += weights[col]
                
    # Clinical Thresholds: 10 pts (Smoking+Cough) now triggers Moderate Risk
    if total_score >= 12:
        return "High Risk", total_score, "red"
    elif total_score >= 5:
        return "Medium Risk", total_score, "orange"
    else:
        return "Low Risk", total_score, "green"

# ---------- 3. STREAMLIT UI ----------
st.set_page_config(page_title="Lung Cancer Risk Screener", page_icon="🫁")

st.title("🫁 LungHealth+ Risk Screening")
st.write("""
This application uses a **dual-assessment framework**:
1. **Clinical Triage:** Rule-based logic prioritizing high-impact symptoms.
2. **Machine Learning:** Probabilistic prediction based on historical data.
""")

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

# Display symptoms in a clean grid
cols = st.columns(2)
for i, col in enumerate(binary_cols):
    label = col.replace("_", " ").title()
    with cols[i % 2]:
        choice = st.selectbox(label, options=["No", "Yes"], key=col)
        symptom_inputs[col] = 1 if choice == "Yes" else 0

st.markdown("---")

# --- Section 3: Prediction & Results ---
if st.button("🔍 Evaluate My Lung Cancer Risk"):
    # Prepare data for ML Model
    row = {col: (gender_value if col == "GENDER" else (age_value if col == "AGE" else symptom_inputs[col])) for col in feature_cols}
    input_df = pd.DataFrame([row], columns=feature_cols)

    # Calculate Results
    prob_yes = model.predict_proba(input_df)[0][1]
    ml_risk, ml_message = classify_risk(prob_yes)
    triage_risk, triage_score, triage_color = get_triage_score(symptom_inputs)

    st.header("3️⃣ Risk Assessment Result")
    
    # DISPLAY CLINICAL TRIAGE (Priority for accuracy)
    st.subheader("Clinical Triage Assessment")
    if triage_color == "red":
        st.error(f"**Category: {triage_risk}** (Score: {triage_score}/39)")
    elif triage_color == "orange":
        st.warning(f"**Category: {triage_risk}** (Score: {triage_score}/39)")
    else:
        st.success(f"**Category: {triage_risk}** (Score: {triage_score}/39)")
    st.info("The Triage score prioritizes alarm symptoms like Smoking and Chronic Cough.")

    st.markdown("---")

    # DISPLAY ML ESTIMATE
    st.subheader("Machine Learning Prediction")
    st.metric("Model Estimated Probability", f"{prob_yes*100:.2f}%")
    st.progress(min(max(prob_yes, 0.0), 1.0))
    st.write(f"**AI Assessment:** {ml_message}")

else:
    st.info("Fill in your details and click the button to see your clinical and AI risk profiles.")
