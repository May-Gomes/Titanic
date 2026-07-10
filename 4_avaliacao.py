import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

# 1. Carregar modelos e dados de teste gerados no script 3
knn = joblib.load("modelo_knn.pkl")
svm = joblib.load("modelo_svm.pkl")
X_test_scaled = joblib.load("X_test_scaled.pkl")
y_test = joblib.load("y_test.pkl")

def avaliar_e_plotar(nome, modelo, X_test, y_true):
    print(f"\n=============================")
    print(f"--- Resultados: {nome} ---")
    
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    
    # Imprimir as métricas no terminal (Copie estes valores para o Google Doc)
    print(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precisão: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:   {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y_true, y_prob):.4f}")

    # Plotar Matriz de Confusão e Curva ROC (Salve os gráficos para seus slides)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Avaliação Gráfica - {nome}")

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax[0], cmap="Blues")
    ax[0].set_title("Matriz de Confusão")

    RocCurveDisplay.from_estimator(modelo, X_test, y_true, ax=ax[1])
    ax[1].set_title("Curva ROC")
    
    plt.tight_layout()
    plt.show()

# 2. Executar avaliação (abrirá os gráficos que você colocará nos slides)
avaliar_e_plotar("KNN (Baseline)", knn, X_test_scaled, y_test)
avaliar_e_plotar("SVM (Sorteado)", svm, X_test_scaled, y_test)