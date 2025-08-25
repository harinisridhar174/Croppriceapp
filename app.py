import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from gtts import gTTS
import os

# -------------------------------
# ✅ Load LSTM model
# -------------------------------
MODEL_PATH = "lstm_models_h5"  # Make sure this is uploaded in your repo
model = load_model(MODEL_PATH)

# -------------------------------
# ✅ UI Setup
# -------------------------------
st.set_page_config(page_title="🌾 Crop Price Prediction", layout="wide")
st.title("🌾 Crop Price Prediction App")
st.markdown("Helping Farmers Decide When to Sell Crops")

# Language Switcher
language = st.radio("Choose Language / மொழி தேர்ந்தெடுக்கவும்", ["English", "தமிழ்"])

# Input section
crop_name = st.text_input("Enter Crop Name" if language == "English" else "பயிரின் பெயரை உள்ளிடவும்")
state_name = st.text_input("Enter State Name" if language == "English" else "மாநிலத்தின் பெயரை உள்ளிடவும்")
days = st.number_input("Enter number of days for prediction" if language == "English" else "எத்தனை நாட்களுக்கு முன்னறிவிப்பு வேண்டும்", min_value=1, max_value=30)

# Crop images from Wikipedia (dynamic URLs)
crop_images = {
    "rice": "https://upload.wikimedia.org/wikipedia/commons/6/6f/Oryza_sativa_plant.jpg",
    "wheat": "https://upload.wikimedia.org/wikipedia/commons/3/3d/Wheat_close-up.JPG",
    "maize": "https://upload.wikimedia.org/wikipedia/commons/5/57/Maize_plants.jpg",
    "sugarcane": "https://upload.wikimedia.org/wikipedia/commons/b/b1/Sugarcane_field_in_Haryana_India.jpg"
}

if crop_name.lower() in crop_images:
    st.image(crop_images[crop_name.lower()], caption=crop_name.capitalize(), use_column_width=True)

# -------------------------------
# ✅ Prediction Function
# -------------------------------
def predict_price(model, recent_prices, future_days=7):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(recent_prices).reshape(-1, 1))

    X_input = scaled_data[-30:].reshape(1, 30, 1)
    predictions = []
    for _ in range(future_days):
        pred_price = model.predict(X_input, verbose=0)
        predictions.append(scaler.inverse_transform(pred_price)[0][0])
        X_input = np.append(X_input[:, 1:, :], [[pred_price]], axis=1)

    return predictions

# -------------------------------
# ✅ Predict button
# -------------------------------
if st.button("Predict" if language == "English" else "முன்னறிவிப்பு செய்யவும்"):
    # Dummy recent prices (replace with real DB or API)
    recent_prices = np.random.randint(1000, 3000, size=60)  # last 60 days prices

    predictions = predict_price(model, recent_prices, future_days=days)
    final_price = round(predictions[-1], 2)

    if language == "English":
        st.success(f"Predicted price after {days} days: ₹{final_price}")
        advice = "Sell your crop now" if final_price > np.mean(recent_prices) else "Wait for a better price"
    else:
        st.success(f"{days} நாட்களுக்குப் பிறகு கணிக்கப்பட்ட விலை: ₹{final_price}")
        advice = "உங்கள் பயிரை இப்போது விற்கவும்" if final_price > np.mean(recent_prices) else "சிறந்த விலைக்காக காத்திருக்கவும்"

    st.subheader(advice)

    # ✅ Audio Response
    tts = gTTS(text=advice, lang='ta' if language == "தமிழ்" else 'en')
    audio_file = "advice.mp3"
    tts.save(audio_file)
    st.audio(audio_file, format="audio/mp3")

# Footer
st.markdown("---")
st.caption("Developed for Farmers | Powered by LSTM AI")


