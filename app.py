# app.py
import streamlit as st
import pandas as pd

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="🌾 Crop Price Predictor",
    layout="centered"
)

# ----------------- Background Image -----------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://upload.wikimedia.org/wikipedia/commons/0/0a/Farmer_in_India.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ----------------- Title -----------------
st.title("🌱 Crop Price Predictor")
st.subheader("Empowering Farmers with Data-Driven Price Insights 💰")

# ----------------- Load CSV -----------------
try:
    data = pd.read_csv("multi_crop_prices_reduced_2000.csv")

    # Normalize column names
    data.columns = data.columns.str.strip().str.lower()

    required_cols = {"crop", "state", "price"}
    if not required_cols.issubset(set(data.columns)):
        st.error(f"❌ CSV file must have columns: {required_cols}")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading CSV: {e}")
    st.stop()

# ----------------- Farmer Input -----------------
crop = st.selectbox("🌾 Select Crop", sorted(data['crop'].unique()))
state = st.selectbox("📍 Select State", sorted(data['state'].unique()))

# ----------------- Predict Price & Suggestion -----------------
if st.button("🔮 Get Suggestion"):
    df = data[(data['crop'] == crop) & (data['state'] == state)]
    
    if not df.empty:
        predicted_price = df['price'].values[-1]  # last available price
        avg_price = data[data['crop'] == crop]['price'].mean()
        suggestion = "✅ Sell now!" if predicted_price >= avg_price else "⏳ Better to wait."
        
        st.success(f"💵 Predicted Price: ₹{predicted_price}")
        st.info(f"📊 Recent Average Price: ₹{avg_price:.2f}")
        st.warning(f"📌 Suggestion: {suggestion}")
    else:
        st.warning("⚠️ No data available for this crop/state combination.")














