import streamlit as st
import csv
import json
import yfinance as yf
import plotly.express as px
import datetime
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def create_users_file():
    if not os.path.exists('users.csv'):
        with open('users.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['username', 'password', 'email'])

def add_user_csv(username, password, email):
    with open('users.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([username, password, email])

def validate_user_csv(username, password):
    with open('users.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == username and row[1] == password:
                return True
    return False

def is_logged_in():
    return st.session_state.get('logged_in', False)

def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = None

def login_success(username):
    st.session_state['logged_in'] = True
    st.session_state['username'] = username

create_users_file()

if not is_logged_in():
    menu = ['Login', 'Cadastro']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Cadastro':
        st.subheader('Criar Novo Usuário')
        new_user = st.text_input('Nome de Usuário')
        new_email = st.text_input('E-mail')
        new_password = st.text_input('Senha', type='password')
        if st.button('Cadastrar'):
            add_user_csv(new_user, new_password, new_email)
            st.success('Usuário cadastrado com sucesso!')
            st.info('Agora você pode fazer login.')

    elif choice == 'Login':
        st.subheader('Login')
        username = st.text_input('Nome de Usuário')
        password = st.text_input('Senha', type='password')
        if st.button('Login'):
            if validate_user_csv(username, password):
                st.success(f'Bem-vindo, {username}!')
                login_success(username)
            else:
                st.error('Nome de usuário ou senha incorretos')

if is_logged_in():

    st.sidebar.text(f'Logado como: {st.session_state["username"]}')
    st.sidebar.button("Logout", on_click=logout)

    with open('data.json', 'r') as _json:
        data_string = _json.read()

    obj = json.loads(data_string)
    crypto_names = obj["crypto_names"]
    crypto_symbols = obj["crypto_symbols"]

    crypto_dict = dict(zip(crypto_names, crypto_symbols))
    crypto_selected = st.selectbox(label='Select your Crypto:', options=crypto_dict.keys())

    today_date = datetime.datetime.now()
    delta_date = datetime.timedelta(days=360)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(label='Select start date:', value=today_date - delta_date)
    with col2:
        final_date = st.date_input(label='Select final date:', value=today_date)

    _symbol = crypto_dict[crypto_selected] + '-USD'
    df = yf.Ticker(_symbol).history(interval='1d', start=start_date, end=final_date)

    st.title(f'Valores de {crypto_selected}')
    st.dataframe(df)

    st.subheader('Previsão de Preço')

    df['Date'] = df.index
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days

    df['Volume'] = df['Volume']
    df['MA_10'] = df['Close'].rolling(window=10).mean()  # Média móvel de 10 dias
    df['MA_30'] = df['Close'].rolling(window=30).mean() 
    df = df.dropna()  

    # Normalizar as features para evitar discrepâncias nas escalas
    scaler = StandardScaler()

    X = df[['Days', 'Volume', 'MA_10', 'MA_30']]
    X_scaled = scaler.fit_transform(X)
    y = df['Close'].values

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model_choice = st.selectbox('Escolha o modelo de previsão', ['Regressão Linear', 'Random Forest'])

    if model_choice == 'Regressão Linear':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=20)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    st.write(f'MSE ({model_choice}): {mean_squared_error(y_test, y_pred)}')

    future_days = np.array([df['Days'].max() + i for i in range(1, 8)])

    # Em vez de usar valores constantes, projete variações com base nos últimos dias
    recent_volume = df['Volume'].values[-7:]  # Usar os últimos 7 dias de volume como base
    future_volumes = np.roll(recent_volume, shift=-1)  # Rotacionar para criar uma projeção simples

    recent_ma_10 = df['MA_10'].values[-7:]  # Últimos 7 dias da média móvel de 10 dias
    future_ma_10 = np.roll(recent_ma_10, shift=-1)  # Rotacionar para simular uma variação futura

    recent_ma_30 = df['MA_30'].values[-7:]  # Últimos 7 dias da média móvel de 30 dias
    future_ma_30 = np.roll(recent_ma_30, shift=-1)  # Rotacionar para simular uma variação futura

    future_X = pd.DataFrame({
        'Days': future_days,
        'Volume': future_volumes,
        'MA_10': future_ma_10,
        'MA_30': future_ma_30
    })

    # Normalizar as variáveis futuras com o mesmo scaler
    future_X_scaled = scaler.transform(future_X)
    future_prices = model.predict(future_X_scaled)

    future_dates = [df['Date'].max() + datetime.timedelta(days=i) for i in range(1, 8)]

    # Combinar dados históricos e previsões futuras para plotar no gráfico
    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Close': future_prices
    })

    st.subheader(f'Histórico e Previsão ({model_choice})')
    
    combined_df = pd.concat([df[['Date', 'Close']], prediction_df])

    fig = px.line(combined_df, x='Date', y='Close', title=f'{crypto_selected} - Histórico e Previsão ({model_choice})')

    fig.add_scatter(x=prediction_df['Date'], y=prediction_df['Close'], mode='lines', 
                    name='Previsão', line=dict(color='red', dash='dash'))

    st.plotly_chart(fig)

    st.subheader(f'Previsão para os próximos 7 dias ({model_choice})')
    st.dataframe(prediction_df)