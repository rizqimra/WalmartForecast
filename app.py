import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

with open ("XGB_Walmart.pkl", "rb") as file:
    model = pickle.load(file)

def predict_sales(store, date, holiday_flag, temperature, fuel_price, cpi, unemployment):
    # Membuat DataFrame
    data = pd.DataFrame({
        'Store': [store],
        'Date': [date],
        'Holiday_Flag': [holiday_flag],
        'Temperature': [temperature],
        'Fuel_Price': [fuel_price],
        'CPI': [cpi],
        'Unemployment': [unemployment],
    })

    # Formatting Tanggal
    data['Date'] = pd.to_datetime(data['Date'])
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data = data.drop(['Date'], axis=1)

    # MinMax Scaling pada input
    data = scaler.fit_transform(data)

    # Prediksi
    prediction = model.predict(data)

    return prediction[0]

st.title('Walmart Sales Prediction App')

scaler = MinMaxScaler()

# Input Data
store = st.number_input('Pilih Nomor Cabang (1-45)', min_value=1, max_value=45, value=1)
date = st.date_input("Pilih Tanggal", value=None)
holiday_flag = st.radio('Hari Libur Besar', [0, 1], index=0)
temperature = st.number_input('Suhu Daerah', value=0.0)
fuel_price = st.number_input('Harga BBM Daerah', value=0.0)
cpi = st.number_input('CPI', value=0.0)
unemployment = st.number_input('Tingkat Unemployment Daerah', value=0.0)

# Tombol Predict
if st.button('Predict'):
    # Prediksi
    prediction = predict_sales(store, date, holiday_flag, temperature, fuel_price, cpi, unemployment)

    # Menampilkan hasil prediksi
    st.subheader('Predicted Sales:')
    st.write(prediction)