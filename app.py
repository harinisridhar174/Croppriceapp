import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Title
st.title("🌾 Crop Price Prediction & Suggestion")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('multi_crop_prices_extended_cleaned.csv')

data = load_data()

# Load trained LSTM model
with open('lstm_models.pkl', 'rb') as f:
    model = pickle.load(f)

# Dropdowns for crop and state
crops = data['Crop'].unique()
states = data['State'].unique()

crop = st.selectbox("Select Crop", crops)
state = st.selectbox("Select State", states)

# Function to prepare input for LSTM
def prepare_input(crop, state):
    df = data[(data['Crop'] == crop) & (data['State'] == state)]
    df = df.sort_values('Date')  # Ensure chronological order
    recent_prices = df['Price'].values[-30:]  # Last 30 days
    if len(recent_prices) < 30:
        recent_prices = np.pad(recent_prices, (30 - len(recent_prices), 0), 'constant', constant_values=df['Price'].mean())
    scaler = MinMaxScaler()
    recent_prices_scaled = scaler.fit_transform(recent_prices.reshape(-1, 1))
    return recent_prices_scaled.reshape(1, 30, 1), scaler

# Prediction
if st.button("Get Suggestion"):
    try:
        X_input, scaler = prepare_input(crop, state)
        predicted_scaled = model.predict(X_input)
        predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

        avg_price = data[data['Crop'] == crop]['Price'].mean()
        suggestion = "Sell ✅" if predicted_price >= avg_price else "Wait ⏳"

        st.success(f"Predicted Price: ₹{predicted_price:.2f}")
        st.info(f"Suggestion: {suggestion}")
    except Exception as e:
        st.error(f"Error: {e}")














