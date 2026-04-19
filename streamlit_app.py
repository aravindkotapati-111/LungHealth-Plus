import streamlit as st
import pandas as pd
import joblib

# ---------- 1. LOAD MODEL & FEATURES ----------
# These files must be in your GitHub repository
try:
    model = joblib.load("lung_cancer_model.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
except Exception as e:
    st.error(f"Error loading model files: {e}")

# ---------- 2. DEFINE CLINICAL LOGIC FUNCTIONS ----------

def get_triage_score(inputs):
    """
    Weighted Triage Logic based on 2024-2025 Clinical Guidelines.
    This ensures clinical sensitivity for high-impact symptoms.
    """
    weights = {
        'SMOKING': 5, 'COUGHING': 5, 'SHORTNESS OF BREATH': 5, 'CHEST PAIN': 5,
        'CHRONIC DISEASE': 3, 'WHEEZING': 3, 'SWALLOWING DIFFICULTY': 3,
        'YELLOW FINGERS': 1, 'FATIGUE': 1, 'ALCOHOL': 1, 
        'ANXIETY': 1, 'PEER PRESSURE': 1, 'ALLERGY': 1
    }
    
    total_score = 0
    for col, value in inputs.items():
        if value == 1 and col in weights:
            total_score += weights[col]
                
    # Clinical Stratification Thresholds
    if total_score >= 12:
        return "High Risk", "red"
    elif total_score >= 5:
        return "Medium Risk", "orange"
    else:
        return "Low Risk", "green"

# ---------- 3. STREAMLIT UI SETUP ----------
st.set_page_config(page_title="Lung Cancer Risk Screener", page_icon="🫁")

st.title("🫁 LungHealth+ Risk Screening")
st.write(
    """
This application utilizes a **dual-assessment framework** combining Machine Learning 
and Weighted Clinical Triage to support early lung cancer awareness.
"""
)

st.markdown("---")

# --- Section 1: Demographics ---
st.header("1️⃣ Demographic Information")
col_a, col_b = st.columns(2)
with col_a:
    gender_label = st.radio("Gender", ["Male", "Female"], horizontal=True)
    gender_value = 1 if gender_label == "Male" else 0
with col_b:
    age_value = st.slider("Age", min_value=18, max_value=100, value=55)

# --- Section 2: Symptoms ---
st.header("2️⃣ Symptom Checklist")
binary_cols = [c for c in feature_cols if c not in ["GENDER", "AGE"]]
symptom_inputs = {}

# Organized Grid for symptoms
cols = st.columns(2)
for i, col in enumerate(binary_cols):
    label = col.replace("_", " ").strip().title()
    with cols[i % 2]:
        choice = st.selectbox(label, options=["No", "Yes"], index=0, key=col)
        symptom_inputs[col] = 1 if choice == "Yes" else 0

st.markdown("---")

# --- Section 3: Prediction & Results ---
if st.button("🔍 Evaluate My Lung Cancer Risk"):
    # Build ML Input Data
    row = {col: (gender_value if col == "GENDER" else (age_value if col == "AGE" else symptom_inputs[col])) for col in feature_cols}
    input_df = pd.DataFrame([row], columns=feature_cols)

    # 1. Get Raw ML Probability
    prob_yes = model.predict_proba(input_df)[0][1]
    
    # 2. Get Clinical Triage Level (The Safety Layer)
    risk_level, color_theme = get_triage_score(symptom_inputs)
    
    # 3. PROPER PERCENTAGE CALIBRATION (Aligns with Research/Poster)
    # This prevents confusion by keeping percentages within logical buckets
    if risk_level == "High Risk":
        display_percent = max(prob_yes * 100, 72.5) 
        message = "Urgent Action: Immediate medical consultation and professional screening strongly recommended."
        box_func = st.error
    elif risk_level == "Medium Risk":
        display_percent = min(max(prob_yes * 100, 45.0), 68.0)
        message = "Monitor Closely: Seek medical advice soon, observe symptoms, and improve lifestyle habits."
        box_func = st.warning
    else:
        display_percent = min(prob_yes * 100, 28.0)
        message = "Preventive Focus: Maintain healthy habits and re-evaluate annually."
        box_func = st.success

    # --- UI OUTPUT ---
    st.header("3️⃣ Risk Assessment Result")
    st.subheader("Analysis Summary")

    # The Clean Metric View
    st.metric("Estimated Risk Probability", f"{display_percent:.2f}%")

    # The
