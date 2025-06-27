#import the libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#load the dataset into the dataframe
#parse date is used to convert the date column into the date time object
df = pd.read_csv("daily_minimum_temps.csv",parse_dates=["Date"],index_col="Date")

#if in the temp there is string values then convert them into neumeric values by removing the double quotes
#coerce convert all string values into NAN (null values)
df["Temp"] = pd.to_numeric(df["Temp"],errors="coerce")

df = df.dropna(subset=['Temp'])

#normalize the features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df["Temp"].values.reshape(-1,1))

#sequence length for temperature
seq_length = 30

#function for creating sequences
def create_sequences(data_scaled,seq_length):
  x,y=[],[]
  for i in range(len(data_scaled)-seq_length):
    x.append(data_scaled[i:i+seq_length])
    y.append(data_scaled[i+seq_length])
  return np.array(x),np.array(y)

#calling the function and storing the values in x and y
x,y = create_sequences(data_scaled,seq_length)

#divide the dataset into train test and split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=False)

#building the rnn model
model = Sequential([
    LSTM(64,activation="relu",input_shape=(seq_length,1)),
    Dense(1)  #since the output will be single value
])

#compile the model
model.compile(optimizer="adam",loss="mae")

#train the model
model.fit(x_train,y_train,epochs=20,batch_size=32)

#make predictions
y_pred_scaled = model.predict(x_test)

#inverse transform the scaled data
y_pred_scaled = np.clip(y_pred_scaled,0,1)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_pred_actual = scaler.inverse_transform(y_test)

#predict the last day temperature
last_sequence = data_scaled[-seq_length:].reshape(1,seq_length,1)
next_temp_scaled = model.predict(last_sequence)
next_temp_scaled = np.clip(next_temp_scaled,0,1)
next_day_temp = scaler.inverse_transform(next_temp_scaled)

print("next day temperature is",next_day_temp)