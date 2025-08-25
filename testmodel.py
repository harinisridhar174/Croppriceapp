import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ------------------- CONFIG -------------------
MODEL_FILE = "wheat_uttar_pradesh_model.h5"   # 👈 change this to test other models
CROP = "wheat"
STATE = "uttar_pradesh"

CSV_FILE = "multi_crop_prices_reduced_2000.csv"

# ------------------- Load Data -------------------
print("🔄 Loading dataset...")
data = pd.read_csv(CSV_FILE)

subset = data[(data["Crop"].str.lower() == CROP.lower()) &
              (data["State"].str.lower() == STATE.lower())]

if subset.empty:
    print(f"⚠️ No data found for crop={CROP}, state={STATE}")
    exit()

prices = subset["Price"].values.reshape(-1, 1)

if len(prices) < 30:
    print("⚠️ Not enough data (need at least 30 entries).")
    exit()

# ------------------- Load Model -------------------
print(f"🔄 Loading model from {MODEL_FILE}...")
model = tf.keras.models.load_model(MODEL_FILE)

# ------------------- Prepare Data -------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

last_seq = scaled_prices[-30:].reshape(1, 30, 1)

# ------------------- Predict -------------------
predicted_scaled = model.predict(last_seq)
predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

# ------------------- Suggestion -------------------
avg_recent = subset["Price"].tail(5).mean()
suggestion = "✅ Sell now!" if predicted_price >= avg_recent else "⏳ Better to Wait."

# ------------------- Results -------------------
print("\n===== Prediction Result =====")
print(f"🌱 Crop: {CROP} | 📍 State: {STATE}")
print(f"💰 Predicted Future Price: ₹{predicted_price:.2f}")
print(f"📉 Recent Average Price: ₹{avg_recent:.2f}")
print(f"📢 Suggestion: {suggestion}")





