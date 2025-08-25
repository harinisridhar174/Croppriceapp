# app.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="🌾 Crop Price Predictor", layout="wide")

# ----------------- Load CSV -----------------
try:
    data = pd.read_csv("multi_crop_prices_reduced_2000.csv")

    # 🔑 Try renaming common column names
    rename_map = {
        "crop_name": "Crop",
        "Crop_Name": "Crop",
        "state": "State",
        "State_Name": "State",
        "modal_price": "Price",
        "Price_Rs": "Price",
        "price": "Price"
    }
    data = data.rename(columns=rename_map)

    required_cols = {"Crop", "State", "Price"}
    if not required_cols.issubset(set(data.columns)):
        st.error(f"❌ CSV file must have columns: {required_cols}")
        st.write("✅ Found columns in your CSV:", list(data.columns))
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading CSV: {e}")
    st.stop()

# ----------------- Title -----------------
st.markdown(
    """
    <h1 style="text-align:center; color:#2E7D32;">
    🌾 Empowering Farmers with Data-Driven Price Insights 💰
    </h1>
    """,
    unsafe_allow_html=True
)

# ----------------- User Input -----------------
col1, col2 = st.columns(2)

with col1:
    crop = st.selectbox("Select Crop", sorted(data["Crop"].unique()))

with col2:
    state = st.selectbox("Select State", sorted(data["State"].unique()))

# ----------------- Predict Price & Suggestion -----------------
if st.button("📈 Get Price Suggestion"):
    df = data[(data["Crop"] == crop) & (data["State"] == state)]

    if not df.empty:
        predicted_price = df["Price"].iloc[-1]  # latest available price
        avg_price = data[data["Crop"] == crop]["Price"].mean()

        suggestion = "✅ Sell Now!" if predicted_price >= avg_price else "⏳ Wait for Better Price"

        st.success(f"🔮 Predicted Price: ₹{predicted_price:.2f}")
        st.info(f"📊 Recent Average Price: ₹{avg_price:.2f}")
        st.warning(f"💡 Suggestion: {suggestion}")
    else:
        st.error("⚠️ No data available for this crop/state combination.")
















