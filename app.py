import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ------------------- Page Config -------------------
st.set_page_config(page_title="🌾 Crop Price Predictor", layout="centered")
st.title("🌾 Crop Price Prediction with LSTM")

# ------------------- Load Model -------------------
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_models.h5")

try:
    model = load_lstm_model()
    st.success("✅ LSTM model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ------------------- File Upload -------------------
uploaded_file = st.file_uploader("📂 Upload crop price CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data Preview:", df.head())

    # Ask user to pick column if multiple
    price_column = st.selectbox("👉 Select the Price Column", df.columns)

    # Extract prices
    prices = df[price_column].dropna().values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    # Ensure enough data
    if len(scaled_prices) < 60:
        st.error("⚠️ Not enough data! Please upload at least 60 rows of prices.")
    else:
        # Prepare last 60 steps
        X_test = []
        X_test.append(scaled_prices[-60:])
        X_test = np.array(X_test)

        # Predict
        prediction = model.predict(X_test)
        predicted_price = scaler.inverse_transform(prediction)

        # Show result
        st.subheader("📈 Predicted Next Price:")
        st.success(f"💰 {predicted_price[0][0]:.2f}")



