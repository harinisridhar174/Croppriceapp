# app.py
import streamlit as st
import pandas as pd

# ----------------- Page Config -----------------
st.set_page_config(page_title="🌾 Crop Price Predictor", layout="wide")

# ----------------- Background Styling -----------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://upload.wikimedia.org/wikipedia/commons/6/6d/Indian_farmer_in_his_field.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.8);
}
.css-1d391kg, .css-1v0mbdj, .css-1cpxqw2 {
    background: rgba(255, 255, 255, 0.8);
    border-radius: 12px;
    padding: 15px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ----------------- Title -----------------
st.title("🌾 Crop Price Predictor")
st.markdown("### Helping Farmers Decide: **Sell Now or Wait?**")

# ----------------- Load Real CSV -----------------
data = pd.read_csv("multi_crop_prices_reduced_2000.csv")

# ----------------- Farmer Input -----------------
col1, col2 = st.columns(2)

with col1:
    crop = st.selectbox("Select Crop", sorted(data['crop'].unique()))

with col2:
    state = st.selectbox("Select State", sorted(data['state'].unique()))

# ----------------- Predict Price & Suggestion -----------------
if st.button("Get Suggestion 🚜"):
    df = data[(data['crop'] == crop) & (data['state'] == state)]
    
    if not df.empty:
        predicted_price = df['price'].values[-1]   # latest available price
        avg_price = data[data['crop'] == crop]['price'].mean()
        suggestion = "✅ Sell" if predicted_price >= avg_price else "⏳ Wait"
        
        st.success(f"📌 Predicted Price: **₹{predicted_price:.2f}**")
        st.info(f"📊 Average Price for {crop}: ₹{avg_price:.2f}")
        st.warning(f"💡 Suggestion: {suggestion}")
    else:
        st.error("⚠️ No data available for this crop/state combination.")












