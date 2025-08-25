# app_basic_crop.py
import streamlit as st
import pandas as pd

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="Crop Price Predictor",
    layout="centered",
    page_icon="🌾"
)

# ----------------- Custom CSS -----------------
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
        }
        .title {
            text-align: center;
            color: #2E8B57;
            font-size: 36px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: #444;
            font-size: 18px;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #2E8B57;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #256d47;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- Header -----------------
st.markdown('<p class="title">🌾 Crop Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Get smart insights on when to sell your crops</p>', unsafe_allow_html=True)

# ----------------- Sample Data -----------------
data = pd.DataFrame({
    'Crop': ['Wheat', 'Rice', 'Maize', 'Sugarcane'],
    'State': ['State1', 'State2', 'State1', 'State2'],
    'Price': [2000, 1500, 1800, 2200]
})

# ----------------- Farmer Input -----------------
st.subheader("📌 Enter Your Details")
col1, col2 = st.columns(2)
with col1:
    crop = st.selectbox("🌱 Select Crop", data['Crop'].unique())
with col2:
    state = st.selectbox("📍 Select State", data['State'].unique())

# ----------------- Prediction -----------------
st.markdown("---")
if st.button("🔍 Get Suggestion"):
    df = data[(data['Crop']==crop) & (data['State']==state)]
    
    if not df.empty:
        predicted_price = df['Price'].values[0]
        avg_price = data[data['Crop']==crop]['Price'].mean()
        suggestion = "✅ Sell Now" if predicted_price >= avg_price else "⏳ Wait for Better Price"
        
        st.success(f"💰 **Predicted Price:** ₹{predicted_price}")
        if "Sell" in suggestion:
            st.success(f"📢 Suggestion: {suggestion}")
        else:
            st.warning(f"📢 Suggestion: {suggestion}")
    else:
        st.error("⚠️ No data available for this crop/state combination.")


















