import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)

@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

st.title("AI-Based Healthcare Analysis")
st.header("Diabetes Prediction Using Machine Learning")

st.write(
    "Enter the details below to get a machine-learning prediction."
)

if not os.path.exists("diabetes_model.pkl"):
    st.error("Model file not found!")
    st.stop()

model = load_model()

with st.form("prediction_form"):

    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Other"]
    )

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=25
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    smoking_history = st.selectbox(
        "Smoking History",
        [
            "never",
            "No Info",
            "current",
            "former",
            "ever",
            "not current"
        ]
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=100.0,
        value=22.0
    )

    hba1c_level = st.number_input(
        "HbA1c Level",
        min_value=0.0,
        max_value=20.0,
        value=5.5
    )

    blood_glucose_level = st.number_input(
        "Blood Glucose Level",
        min_value=0,
        max_value=500,
        value=100
    )

    submitted = st.form_submit_button("Predict")

if submitted:

    input_data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c_level,
        "blood_glucose_level": blood_glucose_level
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.warning(
                "The model predicts a higher likelihood of diabetes. "
                "Please consult a healthcare professional."
            )
        else:
            st.success(
                "The model predicts a lower likelihood of diabetes. "
                "This does not rule out diabetes."
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.caption(
    "Educational project only. This is not a medical diagnosis."
)
