import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados (substitua o caminho pelo seu arquivo .csv do Kaggle)
train_data = pd.read_csv('Data/train.csv')

# Mostrar as primeiras 5 linhas do dataset
print(train_data.head())

# Verificar a estrutura dos dados (colunas, tipos de dados)
print(train_data.info())

# Verificar a quantidade de valores ausentes por coluna
print(train_data.isnull().sum())

# Visualizar a distribuição do preço das casas (target)
sns.histplot(train_data['SalePrice'], kde=True)
plt.title('Distribuição de Preços das Casas')
plt.show()

# Selecionar apenas as colunas numéricas
numeric_columns = train_data.select_dtypes(include=[np.number])

# Calcular a matriz de correlação
correlation_matrix = numeric_columns.corr()

# Plotar a matriz de correlação
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlação')
plt.show()

