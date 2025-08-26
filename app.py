import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# =======================================================
# Load data
# =======================================================
df = pd.read_csv("multi_crop_prices_reduced_2000.csv")
df.columns = [c.strip().lower() for c in df.columns]  # expects: state,crop,price,date

states_ui = sorted(df["state"].str.title().unique())
crops_ui  = sorted(df["crop"].str.title().unique())

# =======================================================
# Load models (safe)
# =======================================================
available_models = []
models = {}
try:
    with open("lstm_models.pkl", "rb") as f:
        models = pickle.load(f)
    try:
        available_models = sorted([str(k) for k in models.keys()])
    except Exception:
        available_models = []
except Exception:
    models = {}
    st.sidebar.warning("⚠️ Could not load Keras models. Using fallback estimator.")

# =======================================================
# Tamil Translations
# =======================================================
state_translation = {
    "Andhra Pradesh": "ஆந்திரப் பிரதேசம்",
    "Karnataka": "கர்நாடகா",
    "Kerala": "கேரளா",
    "Tamil Nadu": "தமிழ்நாடு",
    "Telangana": "தெலங்கானா",
    "Maharashtra": "மகாராஷ்டிரா",
    "Madhya Pradesh": "மத்திய பிரதேசம்",
    "Uttar Pradesh": "உத்தர பிரதேசம்",
    "Rajasthan": "ராஜஸ்தான்",
    "Punjab": "பஞ்சாப்",
    "Haryana": "ஹரியானா",
    "Bihar": "பீகார்",
    "Gujarat": "குஜராத்",
    "West Bengal": "மேற்கு வங்காளம்",
    "Odisha": "ஒடிசா",
}

crop_translation = {
    "Rice": "அரிசி",
    "Wheat": "கோதுமை",
    "Maize": "சோளம்",
    "Sugarcane": "கரும்பு",
    "Cotton": "பருத்தி",
    "Groundnut": "நிலக்கடலை",
    "Turmeric": "மஞ்சள்",
    "Onion": "வெங்காயம்",
    "Tomato": "தக்காளி",
    "Potato": "உருளைக்கிழங்கு",
    "Banana": "வாழை",
    "Mango": "மாம்பழம்",
    "Chilli": "மிளகாய்",
    "Pulses": "பருப்பு வகைகள்",
    "Barley": "வரகு",
    "Mustard": "கடுகு",
    "Soybean": "சோயாபீன்"
}

# =======================================================
# Language Packs
# =======================================================
LANG_PACK = {
    "en": {
        "title": "🌾 Uzhavan | Smart Crop Price Predictor",
        "subtitle": "Helping farmers decide when to SELL or HOLD crops",
        "state": "🌍 Select State",
        "crop": "🌱 Select Crop",
        "predict": "🚀 Predict Price",
        "result": "📊 Prediction Result",
        "suggestion": "✅ Suggestion",
        "sell": "SELL now — price may drop 📉",
        "hold": "HOLD your crop — price may rise 📈",
        "trained": "✅ Trained crops available",
        "no_data": "No historical data for this State + Crop.",
        "error": "Error"
    },
    "ta": {
        "title": "🌾 உழவன் | புத்திசாலி பயிர் விலை கணிப்பான்",
        "subtitle": "விவசாயிகளுக்கு விற்க / வைத்திருக்க முடிவு செய்ய உதவும் கருவி",
        "state": "🌍 மாநிலத்தைத் தேர்ந்தெடுக்கவும்",
        "crop": "🌱 பயிரைத் தேர்ந்தெடுக்கவும்",
        "predict": "🚀 விலையை கணிக்கவும்",
        "result": "📊 கணிப்பு முடிவு",
        "suggestion": "✅ பரிந்துரை",
        "sell": "இப்போது விற்கவும் — விலை குறையலாம் 📉",
        "hold": "உங்கள் பயிரை வைத்திருங்கள் — விலை உயரும் 📈",
        "trained": "✅ பயிற்சி செய்யப்பட்ட பயிர்கள்",
        "no_data": "இந்த மாநிலம் + பயிருக்கு தரவு இல்லை.",
        "error": "பிழை"
    }
}

# =======================================================
# Page Config
# =======================================================
st.set_page_config(page_title="Uzhavan", layout="centered")

# ---------------- Use Local Background ----------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .main-card {{
        background: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 12px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }}
    .header-card {{
        background:#228B22cc;
        padding: 15px;
        border-radius: 10px;
        text-align:center;
        margin-bottom: 20px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg("ui.jpg")

# ---------------- Language Switch Inline ----------------
col1, col2, col3 = st.columns([8,1,1])
with col2:
    if st.button("🌐 EN"):
        st.session_state["lang"] = "en"
with col3:
    if st.button("🇮🇳 தமிழ்"):
        st.session_state["lang"] = "ta"

if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

lang_code = st.session_state["lang"]
T = LANG_PACK[lang_code]

# ---------------- Header ----------------
st.markdown(
    f"""
    <div class="header-card">
      <h1 style="color:white;margin:0">{T['title']}</h1>
      <p style="color:white;margin:0">{T['subtitle']}</p>
    </div>
    """, unsafe_allow_html=True
)

# =======================================================
# State + Crop Selection
# =======================================================
if lang_code == "ta":
    states_display = [state_translation.get(s, s) for s in states_ui]
    crops_display = [crop_translation.get(c, c) for c in crops_ui]
else:
    states_display = states_ui
    crops_display = crops_ui

with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        state_display = st.selectbox(T["state"], states_display)
        if lang_code == "ta":
            state_ui = [eng for eng, ta in state_translation.items() if ta == state_display]
            state_ui = state_ui[0] if state_ui else state_display
        else:
            state_ui = state_display
    with c2:
        crop_display = st.selectbox(T["crop"], crops_display)
        if lang_code == "ta":
            crop_ui = [eng for eng, ta in crop_translation.items() if ta == crop_display]
            crop_ui = crop_ui[0] if crop_ui else crop_display
        else:
            crop_ui = crop_display
    st.markdown('</div>', unsafe_allow_html=True)

# =======================================================
# Sidebar Info
# =======================================================
st.sidebar.markdown("### " + T["trained"])
if available_models:
    st.sidebar.write(", ".join(sorted(set([k.title() for k in available_models]))))
else:
    st.sidebar.write("Unknown (fallback mode will still work)")

# =======================================================
# Predict
# =======================================================
if st.button(T["predict"], use_container_width=True):
    try:
        state = state_ui.strip().lower()
        crop  = crop_ui.strip().lower()

        sub = df[(df["state"].str.lower()==state) & (df["crop"].str.lower()==crop)]
        if sub.empty:
            st.error(T["no_data"])
        else:
            last_price = float(sub["price"].iloc[-1])

            model = None
            for key in [crop, crop.title(), crop.upper()]:
                if key in models:
                    model = models[key]
                    break

            source = "fallback"
            pred = last_price * 1.05
            if model is not None:
                try:
                    pred = float(np.array(model.predict(np.array([[last_price]]))).ravel()[0])
                    source = "model"
                except Exception:
                    source = "fallback"

            # Result Card
            st.markdown(
                f"""
                <div class="main-card">
                  <h3 style="color:#228B22;margin-top:0">{T['result']}</h3>
                  <p style="font-size:18px;margin:0 0 8px 0">
                    {crop_ui} ({state_ui}) :
                  </p>
                  <h2 style="color:#FF8C00;margin:0">₹ {pred:.2f} / kg</h2>
                  <p style="margin:6px 0 0 0;font-size:12px"><i>Source: {source}</i></p>
                </div>
                """, unsafe_allow_html=True
            )

            # Suggestion Card
            if pred > last_price:
                st.markdown(
                    f"""
                    <div class="main-card" style="background:#dff0d8cc">
                      <h3 style="color:green;margin:0 0 6px 0">{T['suggestion']}</h3>
                      <p style="font-size:18px;margin:0">{T['hold']}</p>
                    </div>
                    """, unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="main-card" style="background:#f2dedecc">
                      <h3 style="color:red;margin:0 0 6px 0">{T['suggestion']}</h3>
                      <p style="font-size:18px;margin:0">{T['sell']}</p>
                    </div>
                    """, unsafe_allow_html=True
                )
    except Exception as e:
        st.error(f"{T['error']}: {e}")




















