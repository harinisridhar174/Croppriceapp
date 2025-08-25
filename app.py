import streamlit as st
import pickle
import numpy as np

# Title and Description
st.set_page_config(page_title="Crop Price Predictor", page_icon="🌾", layout="centered")
st.title("🌾 Crop Price Prediction App")
st.markdown("### Helping Farmers Decide When to Sell Crops")

# Load the trained LSTM model
try:
    with open("lstm_models.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file not found! Please check if `lstm_models.pkl` is uploaded.")
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("Use this app to predict crop prices and decide whether to **sell now or wait**.")

# Inputs for the farmer
crop_name = st.selectbox("Select Crop", ["Wheat", "Rice", "Maize", "Sugarcane", "Cotton", "Soybean"])
state = st.text_input("Enter State")
price_today = st.number_input("Enter Today's Price (₹ per quintal)", min_value=0.0, step=0.01)

# Predict Button
if st.button("Predict Price"):
    if price_today <= 0:
        st.warning("Please enter a valid price.")
    else:
        # Prepare input for the model
        input_data = np.array([[price_today]])  # Modify based on your model input shape
        try:
            predicted_price = model.predict(input_data)[0]
            st.success(f"📈 Predicted Price: ₹ {predicted_price:.2f}")

            # Advice for the farmer
            if predicted_price > price_today:
                st.markdown("✅ **Advice:** Wait! Price is expected to go up. 🌟")
            else:
                st.markdown("❌ **Advice:** Sell now! Price might drop. ⚠")
        except Exception as e:
            st.error(f"Prediction Error: {e}")



