#pip install streamlit
#pip install yfinance
#pip install plotly
#pip install pandas numpy json datetime

import streamlit as st
import pandas as pd 
import numpy as np 
import plotly.express as px
import datetime
import json
import yfinance as yf


with open('data.json', 'r') as _json:
    data_string = _json.read()

obj = json.loads(data_string)

crypto_names = obj["crypto_names"]

crypto_symbols = obj["crypto_symbols"]

crypto_dict = dict(zip(crypto_names, crypto_symbols))

crypto_selected = st.selectbox(label = 'Select your Crypto: ',
                               options = crypto_dict.keys())

today_date = datetime.datetime.now()
delta_date = datetime.timedelta(days=360)

col1, col2 = st.columns(2)

with col1:

    start_date = st.date_input(label = 'Select start date: ',
                               value= today_date - delta_date)
    
with col2:

    final_date = st.date_input(label = 'Select final date: ',
                               value= today_date)
    
st.title(final_date)

_symbol = crypto_dict[crypto_selected] + '-USD'

df = yf.Ticker(_symbol).history(interval='1d',
                                start=start_date,
                                end=final_date)

st.title (f'Valores de {crypto_selected}')
st.dataframe(df)

fig = px.line(df,
              x = df.index,
              y = 'Close')

st.plotly_chart(fig)