import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset
data = pd.read_csv("multi_crop_reduced_2000.csv")

st.title("🌾 Simple Crop Price Predictor")

# User input
crop = st.selectbox("Select Crop", data["Crop"].unique())
state = st.selectbox("Select State", data["State"].unique())

if st.button("Predict Price"):
    subset = data[(data["Crop"] == crop) & (data["State"] == state)]

    if subset.empty:
        st.error("No data available for this crop and state.")
    else:
        # Historical prices
        y = subset["Price"].values
        X = np.arange(len(y)).reshape(-1, 1)

        # Train a simple regression model
        model = LinearRegression().fit(X, y)

        # Predict next day price
        predicted_price = model.predict([[len(y)+1]])[0]

        # Recent average
        avg_price = subset["Price"].tail(5).mean()

        # Suggestion
        suggestion = "SELL now ✅" if predicted_price > avg_price else "WAIT ⏳"

        # Show results
        st.write(f"Predicted Price: {predicted_price:.2f}")
        st.write(f"Recent Avg Price: {avg_price:.2f}")
        st.write(f"Suggestion: {suggestion}")




