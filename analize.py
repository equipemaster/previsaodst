import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Definir tipos de dados para economizar memória
dtype_labels = {'period': 'category', 'dst': 'float32'}
dtype_satellite = {
    'period': 'category',
    'timedelta': 'object',  # ler timedelta como string
    'gse_x_ace': 'float32',
    'gse_y_ace': 'float32',
    'gse_z_ace': 'float32',
    'gse_x_dscovr': 'float32',
    'gse_y_dscovr': 'float32',
    'gse_z_dscovr': 'float32'
}
dtype_solar_wind = {
    'period': 'category',
    'timedelta': 'object',  # Será convertido posteriormente
    'temperature': 'float32',
    'speed': 'float32',  
    'bt': 'float32',
    'bx_gsm': 'float32',
    'by_gsm': 'float32',
    'bz_gsm': 'float32'
}
dtype_sunspots = {'period': 'category', 'timedelta': 'object', 'smoothed_ssn': 'float32'}

# Carregar os arquivos com tipos de dados especificados e colunas necessárias
labels_df = pd.read_csv('labels.csv', dtype=dtype_labels, usecols=['period', 'timedelta', 'dst'])
satellite_pos_df = pd.read_csv('satellite_pos.csv', dtype=dtype_satellite, usecols=list(dtype_satellite.keys()))
sunspots_df = pd.read_csv('sunspots.csv', dtype=dtype_sunspots, usecols=['period', 'timedelta', 'smoothed_ssn'])
solar_wind_df = pd.read_csv('solar_wind.csv', dtype=dtype_solar_wind, usecols=list(dtype_solar_wind.keys()))

# Converter 'timedelta' após leitura como strings
labels_df['timedelta'] = pd.to_timedelta(labels_df['timedelta'])
satellite_pos_df['timedelta'] = pd.to_timedelta(satellite_pos_df['timedelta'])
sunspots_df['timedelta'] = pd.to_timedelta(sunspots_df['timedelta'])
solar_wind_df['timedelta'] = pd.to_timedelta(solar_wind_df['timedelta'])

# Mesclar os dados com base em 'period' e 'timedelta'
merged_df = labels_df.merge(satellite_pos_df, on=['period', 'timedelta'], how='left')
merged_df = merged_df.merge(sunspots_df, on=['period', 'timedelta'], how='left')
merged_df = merged_df.merge(solar_wind_df, on=['period', 'timedelta'], how='left')

# Preencher os valores ausentes com 'ffill' e 'bfill' para maior robustez
merged_df.fillna(method='ffill', inplace=True)
merged_df.fillna(method='bfill', inplace=True)

# Definir as features e a variável alvo
X = merged_df[['gse_x_ace', 'gse_y_ace', 'gse_z_ace',
               'gse_x_dscovr', 'gse_y_dscovr', 'gse_z_dscovr',
               'smoothed_ssn',

               'temperature', 'speed','bt', 'bx_gsm', 'by_gsm', 'bz_gsm']].astype('float32')
y = merged_df['dst'].astype('float32')

# Liberar memória excluindo o DataFrame mesclado
del merged_df

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar as features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Liberar memória excluindo os DataFrames originais
del X, y, X_train, X_test

# Construindo o modelo
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
     Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  # Saída linear para regressão
])

# Compilando o modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Treinando o modelo
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Avaliando o modelo no conjunto de teste
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'MAE no conjunto de teste: {mae}')

# Gerar predições no conjunto de teste
y_pred = model.predict(X_test_scaled).flatten()

# Criar DataFrame de resultados
resultado_df = pd.DataFrame({
    'Valor Real': y_test.reset_index(drop=True),
    'Valor Previsto': y_pred
})

# Classificar as tempestades
def classificar_tempestade(dst):
    if dst <= -200:
        return 'Tempestade Severa'
    elif dst <= -100:
        return 'Tempestade Intensa'
    elif dst <= -50:
        return 'Tempestade Moderada'
    else:
        return 'Sem Tempestade'

resultado_df['Classificação'] = resultado_df['Valor Previsto'].apply(classificar_tempestade)

# Identificar períodos de tempestade
resultado_df['Tempestade'] = resultado_df['Valor Previsto'] <= -50

# Plotar o índice Dst previsto com áreas de tempestade destacadas
plt.figure(figsize=(15, 7))
plt.plot(resultado_df.index, resultado_df['Valor Previsto'], label='Dst Previsto')

# Destacar as áreas de tempestade
for i in range(len(resultado_df)):
    if resultado_df['Tempestade'].iloc[i]:
        plt.axvspan(i-0.5, i+0.5, color='lightblue', alpha=0.5)

# Adicionar linhas de referência
plt.axhline(-50, color='yellow', linestyle='--', label='Moderada (-50 nT)')
plt.axhline(-100, color='orange', linestyle='--', label='Intensa (-100 nT)')
plt.axhline(-200, color='red', linestyle='--', label='Severa (-200 nT)')

plt.xlabel('Amostra')
plt.ylabel('Dst Previsto (nT)')
plt.title('Índice Dst Previsto com Períodos de Tempestade Destacados')
plt.legend()
plt.show()

# Contar as classificações
print(resultado_df['Classificação'].value_counts())
