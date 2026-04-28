import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

model = joblib.load("heart_knn_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.title("❤️ AI Heart Disease Predictor")
st.markdown("Enter patient medical details below to estimate heart disease risk.")

st.sidebar.title("About")

st.sidebar.info(
"""
This AI model predicts the risk of heart disease using a KNN model trained on medical data.

Model: K-Nearest Neighbors  
Accuracy: ~88%
"""
)

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 100)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_amgina = st.selectbox("Exercise-Induced Angina", ["Y","N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        "Age" : age,
        "RestingBP" : resting_bp,
        "Cholesterol" : cholesterol,
        "FastingBS" : fasting_bs,
        "Max_HR" : max_hr,
        "Oldpeak" : oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain : 1,
        "RestingECG_" + resting_ecg : 1,
        "ExerciseAngina_" + exercise_amgina : 1,
        "ST_Slope_" + st_slope : 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    with st.spinner("Analyzing patient data..."):
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("Prediction Confidence")
    st.progress(float(probability))
    st.write(f"Risk Probability: {probability*100:.2f}%")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")