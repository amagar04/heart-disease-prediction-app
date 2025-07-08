import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS for aesthetic background and styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.pexels.com/photos/7130480/pexels-photo-7130480.jpeg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .css-1d391kg {
        color: #004d66;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
    }
    .stButton>button {
        background-color: #004d66;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 10em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("Heart Disease Prediction App")

# Load your trained model
model = joblib.load('heart_disease_model.pkl')

# Two-column input layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ['Male', 'Female'])
    chest_pain = st.selectbox("Chest Pain Type", ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
    resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])

with col2:
    resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
    max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=202, value=150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ['Yes', 'No'])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)
    st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

# Encode inputs to match training data
sex = 1 if sex == 'Male' else 0
chest_pain_mapping = {'Typical Angina':0, 'Atypical Angina':1, 'Non-Anginal Pain':2, 'Asymptomatic':3}
chest_pain = chest_pain_mapping[chest_pain]
resting_ecg_mapping = {'Normal':1, 'ST':2, 'LVH':0}
resting_ecg = resting_ecg_mapping[resting_ecg]
exercise_angina = 1 if exercise_angina == 'Yes' else 0
st_slope_mapping = {'Up':2, 'Flat':1, 'Down':0}
st_slope = st_slope_mapping[st_slope]

# Create input DataFrame with correct column names
input_df = pd.DataFrame([[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg,
                          max_hr, exercise_angina, oldpeak, st_slope]],
                        columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                                 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.error("High risk of Heart Disease.")
    else:
        st.success("No Heart Disease detected.")

# Footer branding
st.markdown(
    """
    <hr>
    <center>
    <small>Developed by Akanksha Laxman Magargit init
 | Heart Disease Prediction App</small>
    </center>
    """,
    unsafe_allow_html=True
)
