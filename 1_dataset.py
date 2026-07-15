import pandas as pd

df = pd.read_csv("Titanic_dataset.csv")

print("Visualizando o topo:")
print(df.head())
print("\nInformações do dataset:")
df.info()
print("\nEstatísticas descritivas:")
print(df.describe())
print("\nValores nulos por coluna:")
print(df.isnull().sum())
