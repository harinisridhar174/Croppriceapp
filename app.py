import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

st.title("🌾 Crop Price Prediction & Suggestion")

# ----------------- Embedded Data -----------------
data = pd.DataFrame({
    'Crop': ['Wheat', 'Rice', 'Maize', 'Sugarcane'],
    'State': ['State1', 'State1', 'State2', 'State2'],
    'Price': [2000, 1500, 1800, 2200],
    'Date': pd.date_range(start='2025-01-01', periods=4)
})

# ----------------- Load LSTM model -----------------
with open('lstm_models.pkl', 'rb') as f:
    model = pickle.load(f)

# Dropdowns
crop = st.selectbox("Select Crop", data['Crop'].unique())
state = st.selectbox("Select State", data['State'].unique())

# Prepare input for LSTM (using embedded data)
def prepare_input(crop, state):
    df = data[(data['Crop']==crop) & (data['State']==state)]
    df = df.sort_values('Date')
    recent_prices = df['Price'].values
    if len(recent_prices) < 30:
        recent_prices = np.pad(recent_prices, (30-len(recent_prices), 0),
                               'constant', constant_values=df['Price'].mean())
    scaler = MinMaxScaler()
    recent_prices_scaled = scaler.fit_transform(recent_prices.reshape(-1,1))
    return recent_prices_scaled.reshape(1,30,1), scaler

# Prediction button
if st.button("Get Suggestion"):
    X_input, scaler = prepare_input(crop, state)
    predicted_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

    avg_price = data[data['Crop']==crop]['Price'].mean()
    suggestion = "Sell ✅" if predicted_price >= avg_price else "Wait ⏳"

    st.success(f"Predicted Price: ₹{predicted_price:.2f}")
    st.info(f"Suggestion: {suggestion}")
















