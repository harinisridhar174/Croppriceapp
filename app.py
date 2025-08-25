import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained models
with open('lstm_models.pkl', 'rb') as f:   # make sure the file is in same folder as app.py
    models = pickle.load(f)

# Load CSV for price reference
df = pd.read_csv('multi_crop_prices_extended_cleaned.csv')
df.columns = df.columns.str.strip().str.capitalize()
df['Date'] = pd.to_datetime(df['Date'])

st.title("🌾 Crop Price Prediction & Recommendation")
st.write("Enter the crop and state to get a recommendation.")

crop_input = st.text_input("Crop")
state_input = st.text_input("State")

if st.button("Predict"):
    key = (crop_input.lower(), state_input.lower())
    if key not in models:
        st.warning("Not enough historical data for this crop-state combination.")
    else:
        model, scaler = models[key]
        prices = df[(df['Crop'].str.lower()==crop_input.lower()) & 
                    (df['State'].str.lower()==state_input.lower())]['Price'].values
        seq_length = 30
        last_seq = scaler.transform(prices[-seq_length:].reshape(-1,1)).reshape(1, seq_length, 1)
        pred_scaled = model.predict(last_seq)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        current_price = prices[-1]
        recommendation = "SELL NOW" if current_price >= pred_price else "WAIT"
        
        st.success(f"Current Price: ₹{current_price:.2f}")
        st.success(f"Predicted Price: ₹{pred_price:.2f}")
        st.info(f"Recommendation: **{recommendation}**")
