import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Enter car details to predict the price")

# Basic inputs (example subset)
enginesize = st.number_input("Engine Size", min_value=50, max_value=500)
horsepower = st.number_input("Horsepower", min_value=40, max_value=500)
curbweight = st.number_input("Curb Weight", min_value=500, max_value=4000)
citympg = st.number_input("City MPG", min_value=5, max_value=60)
highwaympg = st.number_input("Highway MPG", min_value=5, max_value=60)

if st.button("Predict Price"):
    input_data = np.array([[enginesize, horsepower, curbweight, citympg, highwaympg]])

    # Pad remaining features with zeros (simple baseline)
    padded = np.zeros((1, 24))
    padded[:, :5] = input_data

    scaled_data = scaler.transform(padded)
    prediction = model.predict(scaled_data)

    st.success(f"💰 Estimated Car Price: ₹ {prediction[0]:,.2f}")
