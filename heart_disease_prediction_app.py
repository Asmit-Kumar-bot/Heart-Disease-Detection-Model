import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('logisticregression.pkl')

# App title
st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter the patient details to check the presence of heart disease.")

# Input fields
age = st.number_input('Age', min_value=1, max_value=120, value=30)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=50, max_value=250, value=120)
chol = st.number_input('Serum Cholestoral in mg/dl (chol)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', [0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=250, value=150)
exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox('Slope of Peak Exercise ST Segment', [0, 1, 2])
ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (ca)', [0, 1, 2, 3, 4])
thal = st.selectbox('Thalassemia (thal)', [0, 1, 2, 3])

# Convert categorical to numeric
sex = 1 if sex == 'Male' else 0

# Feature names (must match model training)
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Check button
if st.button('Check'):
    input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal]], columns=columns)

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ The person **might have** heart disease.")
    else:
        st.success("✅ The person **does not** have heart disease.")
