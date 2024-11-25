# Cryptocon

## Introdução

Este projeto tem como objetivo fornecer uma plataforma interativa e fácil de usar para acompanhar o preço das principais criptomoedas e prever suas flutuações futuras. Utilizando a biblioteca **Streamlit** para a interface de usuário, o aplicativo oferece funcionalidades como cadastro de usuários, login, exibição de preços históricos de criptomoedas e previsão de preços futuros, com base em modelos de aprendizado de máquina.

## Grupo
Andrey Ranieri, Cássio Cavalcante, Henrique Velloso

## Objetivos

- **Visualização de Preços de Criptomoedas**: O aplicativo permite que os usuários acompanhem os preços atuais de algumas das principais criptomoedas, como Bitcoin, Ethereum e Solana.
  
- **Previsão de Preços**: Utilizando modelos de aprendizado de máquina, o sistema é capaz de prever os preços das criptomoedas para o futuro com base em dados históricos.
  
- **Funcionalidade de Login e Cadastro**: O projeto inclui um sistema de cadastro e login de usuários, permitindo que os usuários personalizem sua experiência e tenham um registro de suas interações com o aplicativo.

- **Facilidade de Uso**: A plataforma foi construída para ser intuitiva e fácil de navegar, permitindo que até usuários sem experiência em programação possam fazer uso completo da ferramenta.

## Como Funciona

### 1. Cadastro e Login

Antes de começar a utilizar o aplicativo, o usuário precisa se cadastrar com um nome de usuário, email e senha. Após o cadastro, o usuário pode fazer login para acessar as funcionalidades do aplicativo.

### 2. Exibição de Preços de Criptomoedas

Após o login, o aplicativo mostra os preços atuais das principais criptomoedas. O usuário pode escolher entre as criptos mais populares (como **Bitcoin**, **Ethereum**, **Solana**, etc.) e acompanhar as flutuações de preços ao longo do tempo.

### 3. Previsão de Preço de Criptomoedas

Com base nos dados históricos coletados pela API **yFinance**, o aplicativo utiliza modelos de aprendizado de máquina para prever o preço das criptomoedas para os próximos dias. O usuário pode selecionar o período desejado (1 ano, 6 meses, 3 meses) e o modelo de previsão que deseja utilizar (Regressão Linear ou Random Forest).

### 4. Visualização de Resultados

Após a previsão, o usuário poderá visualizar os preços históricos e as previsões futuras em gráficos interativos criados com a biblioteca **Plotly**. Os gráficos ajudam a entender melhor as tendências do mercado e as possíveis flutuações futuras.

## Tecnologias Utilizadas

- **Streamlit**: Para a criação da interface interativa do aplicativo.
- **yFinance**: Para coletar dados históricos de preços das criptomoedas.
- **Plotly**: Para a criação de gráficos interativos e visualização de dados.
- **Scikit-learn**: Para a criação e treino dos modelos de aprendizado de máquina (Regressão Linear e Random Forest).
- **Pandas e Numpy**: Para manipulação de dados e cálculos estatísticos.
- **CSV e JSON**: Para armazenar dados dos usuários e das criptomoedas.

## Como Este Projeto Pode Ajudar

O mercado de criptomoedas é altamente volátil e imprevisível, tornando difícil para investidores preverem movimentos futuros com precisão. Este aplicativo fornece uma ferramenta útil para os usuários visualizarem as tendências históricas dos preços e obterem previsões baseadas em modelos estatísticos, ajudando-os a tomar decisões mais informadas sobre seus investimentos.

Além disso, o sistema de login e cadastro permite que cada usuário tenha uma experiência personalizada, com registros de suas ações dentro do aplicativo, como login, logout e interações com as criptomoedas.


