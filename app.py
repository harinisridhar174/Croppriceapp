import streamlit as st
import pandas as pd

# ----------------- Page Config -----------------
st.set_page_config(page_title="🌾 Crop Price Predictor", layout="wide")

# ----------------- Custom CSS with Farmer Background -----------------
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("https://upload.wikimedia.org/wikipedia/commons/1/15/Indian_farmer_in_field.jpg");  
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}
.css-1d391kg, .css-hxt7ib {{
    background-color: rgba(255, 255, 255, 0.85) !important;
    border-radius: 15px;
    padding: 15px;
}}
h1, h2, h3, h4, h5, h6, p, span, label {{
    color: #1a1a1a !important;
    font-weight: 600;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ----------------- Title -----------------
st.title("🚜🌾 Crop Price Predictor")
st.markdown("### Empowering Farmers with Data-Driven Price Insights 💰")

# ----------------- Load Data -----------------
@st.cache_data
def load_data():
    return pd.read_csv("multi_crop_prices_reduced_2000.csv")

data = load_data()

# Ensure required columns exist
if not {"Crop", "State", "Price"}.issubset(data.columns):
    st.error("❌ CSV file must have columns: Crop, State, Price")
    st.stop()

# ----------------- Farmer Input -----------------
col1, col2 = st.columns(2)

with col1:
    crop = st.selectbox("🌱 Select Crop", sorted(data["Crop"].unique()))

with col2:
    state = st.selectbox("📍 Select State", sorted(data["State"].unique()))

# ----------------- Predict Price & Suggestion -----------------
if st.button("🔍 Get Suggestion"):
    df = data[(data["Crop"] == crop) & (data["State"] == state)]

    if not df.empty:
        predicted_price = df["Price"].iloc[-1]  # last available price
        avg_price = data[data["Crop"] == crop]["Price"].tail(5).mean()  # last 5 entries avg
        suggestion = "✅ Sell Now!" if predicted_price >= avg_price else "⏳ Better to Wait."

        st.success(f"💰 **Predicted Price:** ₹{predicted_price}")
        st.info(f"📊 **Recent Avg Price:** ₹{avg_price:.2f}")
        st.warning(f"👉 Suggestion: {suggestion}")
    else:
        st.error("⚠️ No data available for this crop/state combination.")






