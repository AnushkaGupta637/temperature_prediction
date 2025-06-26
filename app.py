import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("🌡️ Temperature Prediction using LSTM")

# Load the dataset
df = pd.read_csv("daily_minimum_temps.csv", parse_dates=["Date"], index_col="Date")
df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")
df.dropna(subset=["Temp"], inplace=True)

# Show the dataset
if st.checkbox("Show Raw Dataset"):
    st.write(df)

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df["Temp"].values.reshape(-1, 1))

# Create sequences
seq_length = 30
def create_sequences(data_scaled, seq_length):
    x, y = [], []
    for i in range(len(data_scaled) - seq_length):
        x.append(data_scaled[i:i + seq_length])
        y.append(data_scaled[i + seq_length])
    return np.array(x), np.array(y)

x, y = create_sequences(data_scaled, seq_length)

# Train-test split
split_idx = int(len(x) * 0.8)
x_train, x_test = x[:split_idx], x[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Load the trained model
model = tf.keras.models.load_model("model (1).h5", compile=False)

# Make predictions
y_pred_scaled = model.predict(x_test)
y_pred_scaled = np.clip(y_pred_scaled, 0, 1)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test)

# Plot actual vs predicted
st.subheader("📈 Actual vs Predicted Temperatures")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index[-len(y_test):], y_test_actual.flatten(), label="Actual Temp")
ax.plot(df.index[-len(y_test):], y_pred.flatten(), label="Predicted Temp")
ax.set_xlabel("Date")
ax.set_ylabel("Temperature")
ax.legend()
st.pyplot(fig)

# Predict next day's temperature
last_sequence = data_scaled[-seq_length:].reshape(1, seq_length, 1)
next_temp_scaled = model.predict(last_sequence)
next_temp_scaled = np.clip(next_temp_scaled, 0, 1)
next_day_temp = scaler.inverse_transform(next_temp_scaled)

st.subheader("🌤️ Next Day Temperature Prediction")
st.write(f"Predicted next day temperature: **{next_day_temp.flatten()[0]:.2f} °C**")
