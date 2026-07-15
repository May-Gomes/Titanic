import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# 1. Carregar dataset preprocessado da etapa anterior
df = pd.read_csv("Titanic_preprocessado.csv")

# 2. Separar features (X) e classe alvo (y)
X = df.drop(columns=['Survived'])
y = df['Survived']

# 3. Dividir treino e teste com estratificação 
# (stratify=y garante que a proporção de sobreviventes seja idêntica no treino e teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 4. Padronizar os dados (StandardScaler)
# Essencial pois equaliza as escalas matemáticas (ex: tarifa não será mais "pesada" que a idade).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Treinar Modelos

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)


svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)

joblib.dump(knn, "modelo_knn.pkl")
joblib.dump(svm, "modelo_svm.pkl")
joblib.dump(X_test_scaled, "X_test_scaled.pkl")
joblib.dump(y_test, "y_test.pkl")
print("Treinamento concluído! Modelos e variáveis salvos para avaliação.")
