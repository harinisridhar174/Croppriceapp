import streamlit as st
from tensorflow.keras.models import load_model

st.title("🔍 LSTM Model Test")

try:
    model = load_model("lstm_models.h5")
    st.success("✅ Model loaded successfully!")
    st.write("Model Summary:")
    model.summary(print_fn=lambda x: st.text(x))
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")



