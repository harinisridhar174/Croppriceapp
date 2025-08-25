# app_basic_crop.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crop Price Predictor", layout="centered")
st.title("🌾 Crop Price Predictor (Basic Version)")

# ----------------- Load Data -----------------
data = pd.read_csv("multi_crop_prices_reduced_2000.csv")

# ----------------- Farmer Input -----------------
crop = st.selectbox("Select Crop", data['Crop'].unique())
state = st.selectbox("Select State", data['State'].unique())

# ----------------- Predict Price & Suggestion -----------------
if st.button("Get Suggestion"):
    # Filter data for selected crop and state
    df = data[(data['Crop'] == crop) & (data['State'] == state)]
    
    if not df.empty:
        # Use last available price as "predicted"
        predicted_price = df['Price'].iloc[-1]
        # Recent average for that crop
        avg_price = data[data['Crop'] == crop]['Price'].tail(5).mean()
        suggestion = "✅ Sell" if predicted_price >= avg_price else "⏳ Wait"
        
        st.success(f"Predicted Price: ₹{predicted_price:.2f}")
        st.info(f"Suggestion: {suggestion}")
        
        # ----------------- Chart -----------------
        st.subheader("📊 Price Trend")
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(df.index, df['Price'], marker='o', linestyle='-', color="green")
        ax.set_title(f"Price Trend for {crop} in {state}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (₹)")
        st.pyplot(fig)
        
    else:
        st.warning("⚠️ No data available for this crop/state combination.")







