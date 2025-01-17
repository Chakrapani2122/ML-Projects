import streamlit as st
import joblib

# Load models
model1 = joblib.load('Advertising Sales Prediction/best_random_forest_model.pkl')
model2 = joblib.load('E-mail Spam detection/email_spam_detection_logistic_regression.pkl')
model3 = joblib.load('Iris Flower Species recognition/best_model.pkl')

st.title("Machine Learning Models")

# Project 1
st.header("Project 1")
input1 = st.text_input("Enter input for Project 1")
if st.button("Predict Project 1"):
    result1 = model1.predict([input1])
    st.write(f"Output: {result1}")

# Project 2
st.header("Project 2")
input2 = st.text_input("Enter input for Project 2")
if st.button("Predict Project 2"):
    result2 = model2.predict([input2])
    st.write(f"Output: {result2}")

# Project 3
st.header("Project 3")
input3 = st.text_input("Enter input for Project 3")
if st.button("Predict Project 3"):
    result3 = model3.predict([input3])
    st.write(f"Output: {result3}")
