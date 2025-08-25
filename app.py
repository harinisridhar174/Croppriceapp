import streamlit as st
import pandas as pd
import pickle

# Load trained LSTM model
with open("lstm_models.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset
data = pd.read_csv("multi_crop_reduced_2000.csv")  # updated CSV filename

st.title("🌾 Crop Price Prediction & Selling Suggestion")

# User input
crop = st.selectbox("Select Crop", data["Crop"].unique())
state = st.selectbox("Select State", data["State"].unique())

if st.button("Predict Price"):
    # Filter dataset
    subset = data[(data["Crop"] == crop) & (data["State"] == state)]

    if subset.empty:
        st.error("❌ No data available for this crop and state.")
    else:
        # Prepare features (adjust if your model training used different features)
        X = subset.drop(columns=["Price"])  # assuming "Price" is target
        y = subset["Price"]

        # Predict using last available row
        predicted_price = model.predict([X.values[-1]])[0]

        # Recent average price
        recent_avg = y.tail(5).mean()

        # Suggestion
        suggestion = "✅ SELL now" if predicted_price > recent_avg else "⏳ WAIT for better price"

        # Show results
        st.subheader("📊 Prediction Result")
        st.write(f"**Crop:** {crop}")
        st.write(f"**State:** {state}")
        st.write(f"**Predicted Price:** {predicted_price:.2f}")
        st.write(f"**Recent Avg Price:** {recent_avg:.2f}")
        st.success(f"💡 Suggestion: {suggestion}")

        # Add visualization
        st.subheader("📈 Price Trend")
        chart_data = subset[["Price"]].reset_index(drop=True)
        chart_data["Type"] = "Historical"

        # Append predicted price as "future" point
        future_point = pd.DataFrame({
            "Price": [predicted_price],
            "Type": ["Predicted"]
        })

        chart_data = pd.concat([chart_data, future_point], ignore_index=True)

        st.line_chart(chart_data, y="Price", x=None, color="Type")




