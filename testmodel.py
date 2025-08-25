import streamlit as st
import pickle
import numpy as np

st.title("🔍 LSTM Model Test")

try:
    # Load model
    with open("lstm_models.pkl", "rb") as f:
        model = pickle.load(f)

    st.success("✅ Model loaded successfully!")

    # Test dummy prediction
    dummy = np.random.rand(1, 30, 1)  # (batch=1, timesteps=30, features=1)
    try:
        pred = model.predict(dummy)
        st.write("🔮 Dummy prediction:", pred)
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")

