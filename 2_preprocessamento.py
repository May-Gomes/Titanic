import pandas as pd

# 1. Carregar os dados originais
df = pd.read_csv("Titanic_dataset.csv")

# 2. Remoção de colunas desnecessárias
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# 3. Tratamento de valores ausentes
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 4. Converter categóricas com OneHotEncoding (via pandas get_dummies)
# Diferente do LabelEncoder (que criaria hierarquia falsa ex: C=0, S=1, Q=2),
# o get_dummies cria colunas binárias independentes para cada categoria.
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# 5. Salvar o dataset tratado
df.to_csv("Titanic_preprocessado.csv", index=False)
print("Pré-processamento finalizado. Arquivo 'Titanic_preprocessado.csv' salvo com sucesso.")