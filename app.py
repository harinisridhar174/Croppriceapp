import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
from gtts import gTTS
from io import BytesIO
from PIL import Image
import requests

# Page Config
st.set_page_config(page_title="Crop Price Predictor", page_icon="🌾", layout="wide")
st.title("🌾 Crop Price Prediction for Farmers")

# Language switcher
language = st.radio("Select Language", ["English", "தமிழ்"])

# Crop images dictionary (Wikipedia links)
crop_images = {
    "rice": "https://upload.wikimedia.org/wikipedia/commons/6/6f/Rice_Plant.jpg",
    "wheat": "https://upload.wikimedia.org/wikipedia/commons/1/15/Wheat_close-up.JPG",
    "sugarcane": "https://upload.wikimedia.org/wikipedia/commons/3/34/Sugarcane_in_field.jpg",
}

# Input fields
crop = st.selectbox("Select Crop", ["rice", "wheat", "sugarcane"])
state = st.text_input("Enter State (e.g., Tamil Nadu)")
recent_price = st.number_input("Enter Recent Price (₹ per quintal)", min_value=0.0, step=0.1)

# Display crop image
if crop in crop_images:
    img_url = crop_images[crop]
    img = Image.open(requests.get(img_url, stream=True).raw)
    st.image(img, caption=crop.capitalize(), use_column_width=True)

# Prediction button
if st.button("Predict Price"):
    model_path = f"models/{crop}_{state.lower()}.h5"
    
    if not os.path.exists(model_path):
        st.error("Model not found for this crop and state.")
    else:
        # Load model
        model = load_model(model_path)
        
        # Prepare input
        scaler = MinMaxScaler(feature_range=(0, 1))
        price_scaled = scaler.fit_transform(np.array([[recent_price]]))
        price_scaled = price_scaled.reshape((1, 1, 1))
        
        # Predict
        prediction_scaled = model.predict(price_scaled)
        predicted_price = scaler.inverse_transform(prediction_scaled)[0][0]
        
        # Price Message
        message_eng = f"Predicted Price: ₹{predicted_price:.2f}"
        message_tam = f"மதிப்பிடப்பட்ட விலை: ₹{predicted_price:.2f}"
        
        # SELL / WAIT Logic
        suggestion_eng = ""
        suggestion_tam = ""
        if predicted_price > recent_price * 1.05:
            suggestion_eng = "Suggestion: WAIT! Price might increase."
            suggestion_tam = "பரிந்துரை: காத்திருக்கவும்! விலை உயரும்."
        elif predicted_price < recent_price * 0.95:
            suggestion_eng = "Suggestion: SELL NOW! Price might decrease."
            suggestion_tam = "பரிந்துரை: இப்போது விற்கவும்! விலை குறையலாம்."
        else:
            suggestion_eng = "Suggestion: No major change. You can decide."
            suggestion_tam = "பரிந்துரை: பெரிய மாற்றம் இல்லை. நீங்கள் முடிவு செய்யலாம்."
        
        # Display messages
        if language == "English":
            st.success(message_eng)
            st.info(suggestion_eng)
        else:
            st.success(message_tam)
            st.info(suggestion_tam)
        
        # Audio output for both
        final_text = (message_eng + " " + suggestion_eng) if language == "English" else (message_tam + " " + suggestion_tam)
        tts = gTTS(text=final_text, lang="en" if language == "English" else "ta")
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        st.audio(audio_bytes.getvalue(), format="audio/mp3")




