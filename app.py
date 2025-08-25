import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 🎨 Background image
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.ibb.co/8YdBBF0/farmer-bg.jpg");
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

# Load dataset
data = pd.read_csv("multi_crop_reduced_2000.csv")

st.title("🌾 Crop Price Forecast (விவசாயி விலை முன்னறிவு)")

# User input
crop = st.selectbox("👉 Select Crop (பயிர்)", data["Crop"].unique())
state = st.selectbox("👉 Select State (மாநிலம்)", data["State"].unique())

if st.button("🔮 Predict Next 7 Days"):
    subset = data[(data["Crop"] == crop) & (data["State"] == state)]

    if subset.empty:
        st.error("⚠️ No data available for this crop and state.")
    else:
        # Historical data
        latest_price = subset["Price"].iloc[-1]
        avg_price = subset["Price"].tail(5).mean()

        # Train Linear Regression model
        X = np.arange(len(subset)).reshape(-1, 1)
        y = subset["Price"].values
        model = LinearRegression().fit(X, y)

        # Predict next 7 days
        future_days = np.arange(len(subset)+1, len(subset)+8).reshape(-1, 1)
        predicted_prices = model.predict(future_days)

        # Suggestion logic
        if predicted_prices.mean() >= avg_price:
            suggestion = "✅ SELL soon! (விற்கவும்!)"
        else:
            suggestion = "⏳ WAIT for better price. (மேலும் நல்ல விலை காத்திருக்கவும்.)"

        # Show results
        st.success(f"🌱 Crop: {crop} | 📍 State: {state}")
        st.write(f"💰 Latest Price: **{latest_price:.2f}**")
        st.write(f"📉 Recent Avg Price: **{avg_price:.2f}**")
        st.write("🔮 **Predicted Prices for Next 7 Days:**")
        forecast_df = pd.DataFrame({
            "Day": [f"Day {i+1}" for i in range(7)],
            "Predicted Price": predicted_prices.round(2)
        })
        st.table(forecast_df)
        st.subheader(suggestion)

        # 📈 Chart with colors
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(len(y)), y, label="Historical", color="green", marker="o")
        ax.plot(range(len(y), len(y)+7), predicted_prices, label="Predicted", color="orange", marker="o")
        ax.set_title(f"Price Trend for {crop} in {state}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)


