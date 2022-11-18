#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      lenovo
#
# Created:     16-11-2022
# Copyright:   (c) lenovo 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2008-01-01'
end = '2021-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo' , start, end)

#DESCRIBING DATA

st.subheader('Data from 2008-2021')
st.write(df.describe())

#VISUALISATION

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader( 'Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r', label= '100MA')
plt.plot(ma200, 'g', label= '200MA')
plt.plot(df.Close, 'b', label= 'Closing price')
st.pyplot(fig)

# Splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])    #USING 70% DATA FOR TRAINING
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])    #USING 30% DATA FOR TESTING

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0,1))

data_training_array= scaler.fit_transform(data_training)

#Load my model

model = load_model('keras_model.h5')

#Testing part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True )
input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]   # BECOZ NOW OUR SCALAR IS PRESENT IN SCALAR ARRAY AT INDEX 0
y_predicted = y_predicted* scale_factor
y_test = y_test * scale_factor

#FINAL GRAPH

st.subheader('PREDICTIONS vs ORIGINAL')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'ORIGINAL PRICE')
plt.plot(y_predicted, 'r', label = 'PREDICTED PRICE')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
