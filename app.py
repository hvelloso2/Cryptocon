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
import logging
import itertools

# Função para criar o arquivo de usuários
def create_users_file():
    if not os.path.exists('users.csv'):
        with open('users.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['username', 'password', 'email'])

# Função para adicionar usuário
def add_user_csv(username, password, email):
    with open('users.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([username, password, email])

# Função para validar o login
def validate_user_csv(username, password):
    with open('users.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == username and row[1] == password:
                return True
    return False

# Função para verificar se o usuário está logado
def is_logged_in():
    return st.session_state.get('logged_in', False)

# Função de logout
def logout():
    username = st.session_state['username']
    log_user_action(username, "Usuario fez logout.")
    # Resetar o logger ao fazer logout para evitar que o próximo usuário use o mesmo logger
    logging.getLogger(username).handlers.clear()
    st.session_state['logged_in'] = False
    st.session_state['username'] = None

# Função ao fazer login com sucesso
def login_success(username):
    st.session_state['logged_in'] = True
    st.session_state['username'] = username
    log_user_action(username, "Usuario fez login.")

# Função para criar arquivo de log do usuário
def create_log_file(username):
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f'{username}_log.txt')

    # Criar um logger específico para o usuário
    logger = logging.getLogger(username)
    logger.setLevel(logging.INFO)
    
    # Verificar se já existe um handler, se existir, remover para evitar duplicação
    if logger.hasHandlers():
        logger.handlers.clear()

    # Criar um handler para o arquivo específico do usuário
    file_handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

# Função para registrar ações no log
def log_user_action(username, action):
    logger = logging.getLogger(username)
    if not logger.handlers:
        # Se não houver handlers, criar um novo arquivo de log
        logger = create_log_file(username)
    logger.info(action)

# Função para buscar as principais criptomoedas por preço
def get_top_cryptos(limit=5):
    top_cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD']
    data = []
    for crypto in top_cryptos:
        ticker = yf.Ticker(crypto)
        hist = ticker.history(period='1d')
        if not hist.empty:
            price = hist['Close'][0]
            name = crypto.split('-')[0]
            data.append((name, price))

    # Ordenar pelas criptomoedas com maior preço
    data.sort(key=lambda x: x[1], reverse=True)
    return data[:limit]

# Função para mostrar carrossel das criptos
def show_crypto_carousel(cryptos):
    st.markdown("## Principais Criptomoedas")
    cols = itertools.cycle(st.columns(3))  # Criar um carrossel com 3 colunas
    for crypto, price in cryptos:
        col = next(cols)
        col.metric(label=crypto, value=f"${price:,.2f}")

# Função principal da aplicação
create_users_file()

if not is_logged_in():
    # Carregar as criptomoedas principais e exibir no carrossel
    top_cryptos = get_top_cryptos()
    show_crypto_carousel(top_cryptos)

    # Menu de login e cadastro
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

# Após o login, mostra o dashboard
if is_logged_in():
    st.sidebar.text(f'Logado como: {st.session_state["username"]}')
    st.sidebar.button("Logout", on_click=logout)

    # Carregar informações de criptomoedas
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
    log_user_action(st.session_state['username'], f"Visualizou valores de {crypto_selected}.")

    # Previsão de Preços
    st.subheader('Previsão de Preço')
    df['Date'] = df.index
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days

    # Adicionando média móvel
    df['MA_10'] = df['Close'].rolling(window=10).mean()  
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df = df.dropna()

    # Normalizar as features para evitar discrepâncias nas escalas
    scaler = StandardScaler()
    X = df[['Days', 'Volume', 'MA_10', 'MA_30']]
    X_scaled = scaler.fit_transform(X)
    y = df['Close'].values

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Escolha do modelo
    model_choice = st.selectbox('Escolha o modelo de previsão', ['Regressão Linear', 'Random Forest'])

    if model_choice == 'Regressão Linear':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=20)

    # Treinar o modelo
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f'MSE ({model_choice}): {mean_squared_error(y_test, y_pred)}')

    # Prever preços para os próximos dias
    forecast_days = st.slider('Quantos dias no futuro você deseja prever?', min_value=1, max_value=30, value=7)
    future_days = np.array([df['Days'].max() + i for i in range(1, forecast_days + 1)])

    # Gerar os dados futuros baseados nos últimos valores conhecidos
    recent_volume = df['Volume'].values[-forecast_days:]
    recent_ma_10 = df['MA_10'].values[-forecast_days:]
    recent_ma_30 = df['MA_30'].values[-forecast_days:]

    # Se os valores forem menores que forecast_days, complete com o último valor conhecido
    if len(recent_volume) < forecast_days:
        recent_volume = np.pad(recent_volume, (0, forecast_days - len(recent_volume)), mode='edge')
    if len(recent_ma_10) < forecast_days:
        recent_ma_10 = np.pad(recent_ma_10, (0, forecast_days - len(recent_ma_10)), mode='edge')
    if len(recent_ma_30) < forecast_days:
        recent_ma_30 = np.pad(recent_ma_30, (0, forecast_days - len(recent_ma_30)), mode='edge')

    # Criar DataFrame para os dados futuros
    future_X = pd.DataFrame({
        'Days': future_days,
        'Volume': recent_volume,
        'MA_10': recent_ma_10,
        'MA_30': recent_ma_30
    })

    # Escalar os dados futuros com o mesmo scaler do treinamento
    future_X_scaled = scaler.transform(future_X)
    future_prices = model.predict(future_X_scaled)

    # Gerar datas futuras
    future_dates = [df['Date'].max() + datetime.timedelta(days=i) for i in range(1, forecast_days + 1)]

    # Criar DataFrame para exibir a previsão
    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Close': future_prices
    })

    # Combinar dados históricos e previstos para o gráfico
    combined_df = pd.concat([df[['Date', 'Close']], prediction_df])

    # Exibir o gráfico
    fig = px.line(combined_df, x='Date', y='Close', title=f'{crypto_selected} - Histórico e Previsão ({model_choice})')

    # Adicionar a linha de previsão
    fig.add_scatter(x=prediction_df['Date'], y=prediction_df['Close'], mode='lines', 
                    name='Previsão', line=dict(color='red', dash='dash'))

    st.plotly_chart(fig)

    st.subheader(f'Previsão para os próximos {forecast_days} dias ({model_choice})')
    st.dataframe(prediction_df)