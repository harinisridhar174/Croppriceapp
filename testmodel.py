import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ------------------- CONFIG -------------------
CSV_FILE = "multi_crop_prices_reduced_2000.csv"
MODELS_DIR = "."   # folder where your .h5 models are stored

# ------------------- Load Data -------------------
print("🔄 Loading dataset...")
data = pd.read_csv(CSV_FILE)

# ------------------- Helper Function -------------------
def test_model(model_file, crop, state):
    subset = data[(data["Crop"].str.lower() == crop.lower()) &
                  (data["State"].str.lower() == state.lower())]

    if subset.empty:
        print(f"⚠️ No data for {crop}-{state}, skipping...")
        return

    prices = subset["Price"].values.reshape(-1, 1)

    if len(prices) < 30:
        print(f"⚠️ Not enough data for {crop}-{state} (need at least 30).")
        return

    try:
        model = tf.keras.models.load_model(model_file)
    except Exception as e:
        print(f"❌ Failed to load {model_file}: {e}")
        return

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    last_seq = scaled_prices[-30:].reshape(1, 30, 1)

    predicted_scaled = model.predict(last_seq)
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

    avg_recent = subset["Price"].tail(5).mean()
    suggestion = "✅ Sell now!" if predicted_price >= avg_recent else "⏳ Better to Wait."

    print("\n===== Prediction Result =====")
    print(f"📁 Model File: {model_file}")
    print(f"🌱 Crop: {crop} | 📍 State: {state}")
    print(f"💰 Predicted Future Price: ₹{predicted_price:.2f}")
    print(f"📉 Recent Average Price: ₹{avg_recent:.2f}")
    print(f"📢 Suggestion: {suggestion}")


# ------------------- Loop Through Models -------------------
for file in os.listdir(MODELS_DIR):
    if file.endswith(".h5"):
        # Example filename: "wheat_uttar_pradesh_model.h5"
        parts = file.replace("_model.h5", "").split("_")
        if len(parts) < 2:
            print(f"⚠️ Skipping unrecognized filename: {file}")
            continue

        crop = parts[0]
        state = "_".join(parts[1:])   # handles multi-word states
        test_model(os.path.join(MODELS_DIR, file), crop, state)






