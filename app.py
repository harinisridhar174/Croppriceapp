import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timedelta
import base64

# ----------------- Page Config -----------------
st.set_page_config(page_title="Agri Crop Price Predictor", layout="wide")

# ----------------- Background Image -----------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #1B4332;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Use Wikipedia plough image (download it as plough_tool.jpg and keep in project folder)
try:
    add_bg_from_local("plough_tool.jpg")
except FileNotFoundError:
    st.warning("Background image not found. Proceeding without background.")

# ----------------- Load Model -----------------
with open('crop_price_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

model = model_data['model']
scaler = model_data['scaler']
crop_state_data = model_data['crop_state_data']

# ----------------- Title -----------------
st.markdown("<h1 style='color:#2E8B57;'>🌾 Agri Crop Price Predictor</h1>", unsafe_allow_html=True)
st.write("Enter the details below to get the predicted price and sell recommendation:")

# ----------------- Farmer Inputs -----------------
crop_name = st.selectbox("Select Crop", crop_state_data['Crop'].unique())
state = st.selectbox("Select State", crop_state_data['State'].unique())
current_price = st.number_input("Enter Current Market Price", min_value=0.0, value=0.0)

# ----------------- Prediction -----------------
if st.button("Predict Price and Recommendation"):
    input_df = crop_state_data[(crop_state_data['Crop']==crop_name) & 
                               (crop_state_data['State']==state)].copy()

    if input_df.empty:
        st.warning("No data available for this crop & state combination.")
    else:
        # Take last available features
        last_features = input_df.iloc[-1:].drop(['Price'], axis=1).values
        last_scaled = scaler.transform(last_features)

        # Predict price
        predicted_price_scaled = model.predict(last_scaled)
        predicted_price = scaler.inverse_transform(
            np.hstack([last_features[:, :-1], predicted_price_scaled.reshape(-1,1)])
        )[:, -1][0]

        # Recommendation logic (enhanced)
        if predicted_price > current_price * 1.10:
            recommendation = "🚜 Strongly wait – price likely to increase further"
            best_time = datetime.now() + timedelta(days=10)
        elif predicted_price > current_price * 1.05:
            recommendation = "🌱 Wait to sell for higher profit"
            best_time = datetime.now() + timedelta(days=7)
        else:
            recommendation = "💰 Sell now"
            best_time = datetime.now()

        # 🔹 Trend Indicator
        if predicted_price > current_price:
            trend = "📈 Rising"
        elif predicted_price < current_price:
            trend = "📉 Falling"
        else:
            trend = "➖ Stable"

        # ----------------- Display Results -----------------
        st.success(f"Predicted Price: ₹{predicted_price:.2f}")
        st.info(f"Recommendation: {recommendation}")
        st.info(f"Suggested Best Time to Sell: {best_time.strftime('%Y-%m-%d')}")
        st.warning(f"Trend: {trend}")
