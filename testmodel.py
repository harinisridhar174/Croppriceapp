import streamlit as st
import sys, types, pickle, importlib

st.title("🔍 LSTM Model Test")

# --- Show environment versions ---
st.write("**Python**:", sys.version)
try:
    import tensorflow as tf
    st.write("**TensorFlow**:", tf.__version__)
except Exception as e:
    st.error(f"TensorFlow not importable: {e}")

# Try keras import
keras = None
keras_import_err = None
try:
    import keras
    keras = keras
    st.write("**Keras**:", keras.__version__)
except Exception as e:
    keras_import_err = e
    st.warning(f"Keras import failed: {e}")

# --- Compatibility shim (only if keras import failed) ---
if keras is None:
    try:
        import tensorflow as tf
        # map the module name "keras" to tf.keras so pickle can find it
        sys.modules["keras"] = tf.keras
        sys.modules["keras.activations"] = tf.keras.activations
        sys.modules["keras.layers"] = tf.keras.layers
        sys.modules["keras.models"] = tf.keras.models
        sys.modules["keras.optimizers"] = tf.keras.optimizers
        sys.modules["keras.losses"] = tf.keras.losses
        sys.modules["keras.utils"] = tf.keras.utils
        keras = tf.keras
        st.info("✅ Applied compatibility shim: using tf.keras as 'keras'")
    except Exception as e:
        st.error(f"Failed to apply keras shim: {e}")

# --- Try loading the pickle model ---
try:
    with open("lstm_models.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded from pickle!")
    # Try a tiny dummy prediction with a common window size
    import numpy as np
    dummy = np.random.rand(1, 30, 1)
    try:
        pred = model.predict(dummy)
        st.write("🔮 Dummy prediction shape:", getattr(pred, "shape", type(pred)))
    except Exception as e:
        st.warning(f"Model loaded, but prediction failed: {e}")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    if keras_import_err:
        st.caption("Hint: Above you can see the keras import error; ensure requirements & runtime are correct.")


