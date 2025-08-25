import keras
import numpy as np

# Load your model (example: barley_bihar_model.h5)
model = keras.models.load_model("barley_bihar_model.h5")

# Print model summary
print("✅ Model loaded successfully!")
model.summary()

# Create dummy input (adjust timesteps/features if your model is different)
# Most LSTM models expect shape: (batch_size, timesteps, features)
dummy_input = np.random.rand(1, 30, 1)  # 1 sample, 30 time steps, 1 feature

# Predict
prediction = model.predict(dummy_input)
print("🔮 Prediction from model:", prediction)







