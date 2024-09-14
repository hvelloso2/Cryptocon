import streamlit as st
import csv
import json
import yfinance as yf
import plotly.express as px
import datetime
import os

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

    st.title(final_date)

    _symbol = crypto_dict[crypto_selected] + '-USD'
    df = yf.Ticker(_symbol).history(interval='1d', start=start_date, end=final_date)

    st.title(f'Valores de {crypto_selected}')
    st.dataframe(df)
    fig = px.line(df, x=df.index, y='Close')

    st.plotly_chart(fig)
