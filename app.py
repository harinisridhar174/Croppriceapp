# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ------------------- Page Config -------------------
st.set_page_config(page_title="🌾 Crop Price Predictor", layout="centered")
st.title("🌾 Crop Price Prediction with LSTM")

# ------------------- Load Data -------------------
@st.cache_data
def load_data():
    return pd.read_csv("multi_crop_prices_reduced_2000.csv")

data = load_data()

# ------------------- Load Model -------------------
@st.cache_resource
def load_model():
    with open("lstm_models.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ------------------- Farmer Input -------------------
crop = st.selectbox("👉 Select Crop", data["Crop"].unique())
state = st.selectbox("👉 Select State", data["State"].unique())

# ------------------- Predict -------------------
if st.button("🔮 Predict Future Price"):
    subset = data[(data["Crop"] == crop) & (data["State"] == state)]

    if subset.empty:
        st.error("⚠️ No data available for this crop/state.")
    else:
        # Use last 30 prices for prediction (adjust window if trained differently)
        prices = subset["Price"].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)

        last_seq = scaled_prices[-30:].reshape(1, 30, 1)  # shape for LSTM
        predicted_scaled = model.predict(last_seq)
        predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

        # Suggestion
        avg_recent = subset["Price"].tail(5).mean()
        suggestion = "✅ Sell now!" if predicted_price >= avg_recent else "⏳ Better to Wait."

        # Show results
        st.success(f"🌱 Crop: {crop} | 📍 State: {state}")
        st.write(f"💰 Predicted Future Price: **₹{predicted_price:.2f}**")
        st.write(f"📉 Recent Average Price: **₹{avg_recent:.2f}**")
        st.subheader(suggestion)


