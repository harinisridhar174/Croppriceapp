# app_farmers_pro_api.py
import streamlit as st
import pandas as pd
import pickle
import requests
from PIL import Image
from io import BytesIO
from gtts import gTTS
import wikipediaapi

# ----------------- Load Model & Data -----------------
with open('lstm_models.pkl', 'rb') as f:
    model = pickle.load(f)

data = pd.read_csv('multi_crop_prices_extended_cleaned.csv')

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="🌾 Crop Price Predictor", layout="wide", page_icon="🌱")

# Add background image via HTML/CSS
page_bg_img = '''
<style>
.stApp {
background-image: url("https://images.unsplash.com/photo-1506806732259-39c2d0268443?auto=format&fit=crop&w=1350&q=80");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("🌾 Crop Price Predictor")
st.markdown("Helping farmers decide the best time to sell crops!")

# Language selection
language = st.radio("Select Language / மொழியை தேர்ந்தெடுக்கவும்", ["English", "தமிழ்"])

def t(text):
    translations = {
        "Enter Crop": "பண்ணை வகையை உள்ளிடவும்",
        "Select State": "மாநிலத்தை தேர்ந்தெடுக்கவும்",
        "Predict Price": "முன்னறிவிப்பு விலை",
        "Predicted Price": "முன்னறிவிக்கப்பட்ட விலை",
        "Suggestion": "சூழல் ஆலோசனை",
        "Sell": "விற்கவும்",
        "Wait": "காத்திருங்கள்",
        "Crop Info": "பண்ணை தகவல்",
        "Wikipedia Summary": "விக்கிப்பீடியா சுருக்கம்",
        "Play Audio": "ஒலி கேளுங்கள்"
    }
    if language=="தமிழ்":
        return translations.get(text, text)
    return text

# ----------------- Crop Selection with Icons -----------------
st.markdown("#### " + t("Enter Crop"))
cols = st.columns(4)
crop_list = data['Crop'].unique().tolist()
crop_selection = None

# Optional small icons for each crop
crop_icons = {crop: "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Wheat_close-up.JPG/50px-Wheat_close-up.JPG" for crop in crop_list}

for i, crop_name in enumerate(crop_list):
    col = cols[i % 4]
    if col.button(crop_name):
        crop_selection = crop_name

if crop_selection:
    crop = crop_selection
else:
    crop = st.selectbox(t("Enter Crop"), crop_list)

state = st.selectbox(t("Select State"), data['State'].unique())

# ----------------- Prediction -----------------
if st.button(t("Predict Price")):
    input_df = pd.DataFrame([[crop, state]], columns=['Crop', 'State'])
    predicted_price = model.predict(input_df)[0]
    
    recent_prices = data[(data['Crop']==crop) & (data['State']==state)]['Price'].tail(30)
    avg_recent = recent_prices.mean() if not recent_prices.empty else predicted_price
    suggestion = t("Sell") if predicted_price >= avg_recent else t("Wait")
    
    # ----------------- Display Prediction -----------------
    col1, col2 = st.columns([1,1])
    col1.markdown(f"<h2 style='background-color:white;padding:10px;border-radius:10px;text-align:center'>{t('Predicted Price')}: ₹{predicted_price:.2f}</h2>", unsafe_allow_html=True)
    
    color = "green" if suggestion==t("Sell") else "orange"
    col2.markdown(f"<h2 style='background-color:{color};padding:10px;border-radius:10px;text-align:center;color:white'>{t('Suggestion')}: {suggestion}</h2>", unsafe_allow_html=True)
    
    # ----------------- Wikipedia Info -----------------
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page = wiki_wiki.page(crop)
    if page.exists():
        st.markdown(f"**{t('Wikipedia Summary')}:** {page.summary[0:500]}")
    else:
        st.warning("Crop info not found on Wikipedia.")
    
    # ----------------- Crop Image -----------------
    try:
        # You can fetch first image from Wikipedia or use placeholder
        img_url = f"https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Wheat_close-up.JPG/400px-Wheat_close-up.JPG"
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        st.image(img, caption=crop, use_column_width=True)
    except:
        st.warning("Crop image not available.")
    
    # ----------------- Audio -----------------
    text_to_speak = f"{t('Predicted Price')}: {predicted_price:.2f}. {t('Suggestion')}: {suggestion}."
    tts = gTTS(text=text_to_speak, lang='ta' if language=="தமிழ்" else 'en')
    tts.save("prediction.mp3")
    
    audio_file = open("prediction.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3", start_time=0)








