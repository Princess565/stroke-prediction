import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------
# Load model & scaler
# -----------------
model = joblib.load('stroke_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🧠 Stroke Prediction App")
st.write("Enter the patient's information below:")

# -------------
# Input fields
# -------------
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=0, max_value=120, value=45)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
Residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=90.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# -------------
# Preprocess exactly as training
# -------------
if st.button("Predict Stroke Risk"):
    # start building a single-row DataFrame with the input
    input_df = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [1 if hypertension=="Yes" else 0],
        'heart_disease': [1 if heart_disease=="Yes" else 0],
        'ever_married': [1 if ever_married=="Yes" else 0],
        'work_type': [work_type],
        'Residence_type': [1 if Residence_type=="Urban" else 0],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })

    # same encoding as training
    input_df = pd.get_dummies(input_df, columns=['gender','work_type','smoking_status'])

    # Make sure all expected columns exist
    expected_features = scaler.feature_names_in_
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_features]

    # scale
    X_scaled = scaler.transform(input_df)

    # predict
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1] if hasattr(model, "predict_proba") else None

    # output
    if prediction == 1:
        st.error(f"⚠️ High Risk of Stroke (Probability {proba:.2%})" if proba is not None else "⚠️ High Risk of Stroke")
    else:
        st.success(f"✅ Low Risk of Stroke (Probability {proba:.2%})" if proba is not None else "✅ Low Risk of Stroke")

