import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

st.title("🔍 LSTM Model Test")

try:
    model = load_model("lstm_models.h5")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")




