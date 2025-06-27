import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ------------------- Load and Prepare Data -------------------
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("daily_minimum_temps.csv", parse_dates=["Date"], index_col="Date")
    df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")
    df = df.dropna(subset=['Temp'])

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df["Temp"].values.reshape(-1, 1))

    seq_length = 30

    def create_sequences(data_scaled, seq_length):
        x, y = [], []
        for i in range(len(data_scaled) - seq_length):
            x.append(data_scaled[i:i + seq_length])
            y.append(data_scaled[i + seq_length])
        return np.array(x), np.array(y)

    x, y = create_sequences(data_scaled, seq_length)
    x_train, x_test = x[:int(len(x)*0.8)], x[int(len(x)*0.8):]
    y_train, y_test = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

    model = Sequential([
        LSTM(64, activation="relu", input_shape=(seq_length, 1)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mae")
    model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=0)

    return df, scaler, data_scaled, model, x_test, y_test

df, scaler, data_scaled, model, x_test, y_test = load_model_and_data()

# ------------------- Streamlit UI -------------------
st.title("🌡️ Daily Minimum Temperature Prediction")
st.write("This app uses an LSTM model to predict daily minimum temperatures based on historical data.")

# Show dataset
if st.checkbox("Show Raw Data"):
    st.dataframe(df.head(20))

# Plot original temperature data
st.subheader("Temperature Over Time")
st.line_chart(df["Temp"])

# Show test predictions
st.subheader("Model Predictions on Test Data")
y_pred_scaled = model.predict(x_test)
y_pred_scaled = np.clip(y_pred_scaled, 0, 1)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_actual = scaler.inverse_transform(y_test)

fig, ax = plt.subplots()
ax.plot(y_actual, label="Actual Temperature")
ax.plot(y_pred, label="Predicted Temperature")
ax.legend()
st.pyplot(fig)

# Predict next day's temperature
st.subheader("Next Day Temperature Prediction")
last_sequence = data_scaled[-30:].reshape(1, 30, 1)
next_temp_scaled = model.predict(last_sequence)
next_temp_scaled = np.clip(next_temp_scaled, 0, 1)
next_day_temp = scaler.inverse_transform(next_temp_scaled)

st.success(f"Predicted next day's minimum temperature: **{next_day_temp[0][0]:.2f} °C**")
