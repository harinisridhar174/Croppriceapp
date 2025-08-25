import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import plotly.graph_objs as go

st.title("🌾 Crop Price Prediction & Suggestion")

# ----------------- Embedded Data -----------------
data = pd.DataFrame({
    'Crop': ['Wheat', 'Rice', 'Maize', 'Sugarcane'],
    'State': ['State1', 'State1', 'State2', 'State2'],
    'Price': [2000, 1500, 1800, 2200],
    'Date': pd.date_range(start='2025-01-01', periods=4)
})

# ----------------- Load LSTM model -----------------
with open('lstm_models.pkl', 'rb') as f:
    model = pickle.load(f)

# ----------------- User Inputs -----------------
crop = st.selectbox("Select Crop", data['Crop'].unique())
state = st.selectbox("Select State", data['State'].unique())

# ----------------- Prepare Input -----------------
def prepare_input(crop, state):
    df = data[(data['Crop']==crop) & (data['State']==state)]
    df = df.sort_values('Date')
    recent_prices = df['Price'].values
    if len(recent_prices) < 30:
        recent_prices = np.pad(recent_prices, (30-len(recent_prices), 0),
                               'constant', constant_values=df['Price'].mean())
    scaler = MinMaxScaler()
    recent_prices_scaled = scaler.fit_transform(recent_prices.reshape(-1,1))
    return recent_prices_scaled.reshape(1,30,1), scaler, df

# ----------------- Prediction -----------------
if st.button("Get Suggestion"):
    X_input, scaler, df = prepare_input(crop, state)
    predicted_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

    # Suggestion logic
    avg_price = data[data['Crop']==crop]['Price'].mean()
    suggestion = "Sell ✅" if predicted_price >= avg_price else "Wait ⏳"

    # Trend indicator
    last_price = df['Price'].iloc[-1]
    if predicted_price > last_price:
        trend = "📈 Rising"
    elif predicted_price < last_price:
        trend = "📉 Falling"
    else:
        trend = "➖ Stable"

    # ----------------- Display Results -----------------
    st.success(f"Predicted Price: ₹{predicted_price:.2f}")
    st.info(f"Suggestion: {suggestion}")
    st.warning(f"Trend: {trend}")

    # ----------------- Interactive Chart -----------------
    future_date = df['Date'].iloc[-1] + pd.Timedelta(days=1)
    df_future = pd.DataFrame({'Date': [future_date], 'Price': [predicted_price]})

    fig = go.Figure()

    # Past prices
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Price'],
        mode='lines+markers', name="Past Prices"
    ))

    # Predicted price
    fig.add_trace(go.Scatter(
        x=df_future['Date'], y=df_future['Price'],
        mode='markers+text', name="Predicted Price",
        text=["Predicted"], textposition="top center",
        marker=dict(color="red", size=10)
    ))

    fig.update_layout(
        title=f"Price Trend for {crop} in {state}",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
