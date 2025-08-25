import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Crop Price Prediction", page_icon="🌾", layout="centered")

# Custom CSS for better UI
st.markdown("""
    <style>
    body {
        background-color: #F0F8FF;
    }
    .title {
        text-align: center;
        color: #2E8B57;
        font-size: 40px;
        font-weight: bold;
    }
    .prediction {
        color: #1E90FF;
        font-size: 24px;
        text-align: center;
    }
    .recommend {
        color: #FF4500;
        font-size: 20px;
        text-align: center;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">🌾 Crop Price Prediction App</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔍 About")
st.sidebar.info("""
This app predicts future crop prices using a **trained LSTM model**.
Enter your crop name and details to get a recommendation whether to SELL or WAIT.
""")

# Load LSTM Model
try:
    model_data = joblib.load("lstm_models.joblib")
    st.sidebar.success("✅ Model Loaded Successfully!")
except:
    st.sidebar.error("❌ Model file not found. Please upload lstm_models.joblib to the repo.")

# Crop selection
crops = list(model_data.keys()) if isinstance(model_data, dict) else ["Wheat", "Rice", "Maize"]
selected_crop = st.selectbox("Select Crop", crops)

# Farmer input section
st.subheader("📌 Enter Details")
recent_price = st.number_input("Enter Recent Price (₹ per quintal):", min_value=100.0, max_value=10000.0, step=10.0)

if st.button("Predict Price"):
    if isinstance(model_data, dict) and selected_crop in model_data:
        model = model_data[selected_crop]

        # Create dummy sequence for LSTM input (example: last 30 prices, here just repeated value)
        input_data = np.array([recent_price] * 30).reshape(1, 30, 1)

        predicted_price = model.predict(input_data)[0][0]

        st.markdown(f'<p class="prediction">💰 Predicted Price: ₹{predicted_price:.2f}</p>', unsafe_allow_html=True)

        if predicted_price > recent_price:
            st.markdown('<p class="recommend">📈 Recommendation: WAIT (Price may increase)</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="recommend">📉 Recommendation: SELL NOW</p>', unsafe_allow_html=True)

        # Optional price trend chart
        fig, ax = plt.subplots()
        prices = [recent_price, predicted_price]
        ax.plot(['Current', 'Predicted'], prices, marker='o')
        ax.set_ylabel('Price (₹)')
        ax.set_title(f'{selected_crop} Price Trend')
        st.pyplot(fig)

    else:
        st.error("Model for selected crop not found!")
