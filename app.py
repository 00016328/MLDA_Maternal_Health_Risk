import streamlit as st
import pickle
import pandas as pd

@st.cache_data()
def load_pickle(path):
    return pickle.load(open(path, 'rb')) 

model = load_pickle('final_model.pkl')
scaler = load_pickle('standard_scaler.pkl')

if model and scaler:
    st.toast('Model and scaler loaded successfully')
else:
    st.toast('Model and/or scaler could not be loaded')

st.title("Maternal Health Risk Predictor")
st.write("Enter physiological measurements to predict maternal health risk level.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=70, value=30)
    systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=70, max_value=160, value=114)

with col2:
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=7, max_value=90, value=74)
    diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=49, max_value=100, value=76)

bs = st.slider("Blood Sugar", min_value=6.0, max_value=19.0, value=8.7, format="%.1f")
body_temp = st.slider("Body Temperature (°F)", min_value=98.0, max_value=103.0, value=98.6,format="%.1f")

# to show inputed values in the form of table to the user
input_df = pd.DataFrame([[
    age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate
]], columns=["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"])


if st.button("Predict"):
    st.write("Input features:")
    st.table(input_df.T.rename(columns={0: "value"}))
    X_scaled = scaler.transform(input_df)

    pred = model.predict(X_scaled)
    pred_label = int(pred[0])
    RISK_MAP = {0: "LOW", 1: "MID", 2: "HIGH"}
    st.subheader(f"Predicted risk level is {RISK_MAP.get(pred_label, pred_label)}")
