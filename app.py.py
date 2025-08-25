import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Page config ---
st.set_page_config(page_title="🌾 Farmer Crop Price Predictor", layout="wide")

# --- Custom CSS for background & styling ---
st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
    }
    .stButton>button {
        height: 3em;
        width: 100%;
        font-size: 18px;
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    .stMetric>div {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Load pretrained models ---
with open('lstm_models.joblib', 'rb') as f:
    models = joblib.load(f)

# --- Load dataset (optional, for historical chart) ---
df = pd.read_csv('multi_crop_prices_extended_cleaned.csv')
df.columns = df.columns.str.strip().str.capitalize()
df['Date'] = pd.to_datetime(df['Date'])

# --- App header ---
st.markdown("<h1 style='text-align:center; color:#2E8B57;'>🌾 Farmer Crop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#556B2F;'>Get instant SELL / WAIT recommendation for your crop</h4>", unsafe_allow_html=True)
st.write("---")

# --- Crop icons mapping ---
crop_icons = {
    "Wheat": "🌾",
    "Rice": "🌱",
    "Maize": "🌽",
    "Sugarcane": "🍬",
    "Cotton": "🧵",
    "Barley": "🌾",
    "Soybean": "🌱",
    "Potato": "🥔",
    "Tomato": "🍅",
    "Chili": "🌶️"
}

# --- Sidebar with crop icons ---
st.sidebar.header("Select your crop and state")

# Radio buttons with icons for crops
crop_input = st.sidebar.radio(
    "Crop",
    options=list(crop_icons.keys()),
    format_func=lambda x: f"{crop_icons.get(x, '')}  {x}"
)

# Dropdown for states
states = sorted(df['State'].unique())
state_input = st.sidebar.selectbox("State", states)

# --- Prediction button ---
if st.sidebar.button("Predict"):
    key = (crop_input.lower(), state_input.lower())
    if key not in models:
        st.warning("⚠️ Not enough historical data for this crop-state combination.")
    else:
        model, scaler = models[key]
        prices = df[(df['Crop'].str.lower()==crop_input.lower()) &
                    (df['State'].str.lower()==state_input.lower())]['Price'].values
        seq_length = 30
        last_seq = scaler.transform(prices[-seq_length:].reshape(-1,1)).reshape(1, seq_length, 1)
        pred_scaled = model.predict(last_seq)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        current_price = prices[-1]
        recommendation = "SELL NOW" if current_price >= pred_price else "WAIT"

        # --- Color-coded recommendation ---
        st.markdown("### Recommendation")
        if recommendation == "SELL NOW":
            st.markdown(f"<h2 style='color:red; text-align:center;'>{recommendation} 🔴</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:green; text-align:center;'>{recommendation} 🟢</h2>", unsafe_allow_html=True)

        # --- Display prices ---
        col1, col2 = st.columns(2)
        col1.metric("Current Price (₹)", f"{current_price:.2f}")
        col2.metric("Predicted Price (₹)", f"{pred_price:.2f}")

        # --- Historical price chart ---
        st.markdown("### 📈 Historical Prices")
        plt.figure(figsize=(10,4))
        plt.plot(df[(df['Crop']==crop_input) & (df['State']==state_input)]['Date'],
                 prices, marker='o', color='#FFA500')
        plt.title(f"{crop_input} Price Trend in {state_input}", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Price (₹)")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(plt)
