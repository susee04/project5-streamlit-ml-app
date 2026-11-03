import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict the estimated survival chance.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

try:
    model = load_model()
except FileNotFoundError:
    st.error("model.joblib is missing. Please place it in the same folder as app.py.")
    st.stop()

# Inputs
pclass = st.selectbox("Ticket Class", [1, 2, 3])
sex = st.radio("Sex", ["Female", "Male"])
age = st.slider("Age", 1, 80, 30)
fare = st.slider("Fare Paid", 0, 500, 30)

# Prediction
if st.button("Predict Survival"):
    sex_value = 1 if sex == "Female" else 0

    input_data = pd.DataFrame(
        [[pclass, sex_value, age, fare]],
        columns=["pclass", "sex", "age", "fare"]
    )

    probability = model.predict_proba(input_data)[0][1]

    st.metric("Estimated Survival Chance", f"{probability:.0%}")

    st.caption(
        "Educational project using the Titanic dataset. "
        "This prediction is for learning purposes only."
    )