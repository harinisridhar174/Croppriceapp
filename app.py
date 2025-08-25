# app.py
import streamlit as st
import pandas as pd

# ----------------- Page Config -----------------
st.set_page_config(page_title="🌾 Crop Price Predictor", layout="wide")

# Background image (farmer field)
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://upload.wikimedia.org/wikipedia/commons/6/65/Rice_fields_in_Tamil_Nadu_01.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
.block-container {
    background-color: rgba(255,255,255,0.85);
    padding: 2rem;
    border-radius: 15px;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("🌾 Crop Price Predictor")
st.markdown("### Get price insights and suggestions for your crops")

# ----------------- Load CSV -----------------
data = pd.read_csv("multi_crop_prices_reduced_2000.csv")

# Normalize column names
data.columns = [c.strip().lower() for c in data.columns]

# Make sure required columns exist
required_cols = {"crop", "state", "price"}
if not required_cols.issubset(set(data.columns)):
    st.error(f"CSV file must contain these columns: {required_cols}")
    st.stop()

# ----------------- Farmer Input -----------------
crop = st.selectbox("Select Crop", sorted(data['crop'].unique()))
state = st.selectbox("Select State", sorted(data['state'].unique()))

# ----------------- Predict Price & Suggestion -----------------
if st.button("Get Suggestion"):
    df = data[(data['crop'] == crop) & (data['state'] == state)]

    if not df.empty:
        latest_price = df['price'].iloc[-1]  # most recent price
        avg_price = data[data['crop'] == crop]['price'].mean()

        suggestion = "✅ Sell" if latest_price >= avg_price else "⏳ Wait"

        st.success(f"**Predicted Price (Latest): ₹{latest_price:.2f}**")
        st.info(f"**Average Price (All States): ₹{avg_price:.2f}**")
        st.warning(f"**Suggestion: {suggestion}**")
    else:
        st.error("⚠️ No data available for this crop/state combination.")










