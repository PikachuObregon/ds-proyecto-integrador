"""
tfidf_classifier.py
--------------------
Checkpoint: Clasificador Supervisado con TF-IDF (Módulo 3).

Pipeline: texto crudo -> preprocesamiento (reutilizado del Módulo 2) ->
TfidfVectorizer -> clasificador -> evaluación.

Evita Data Leakage: el TfidfVectorizer se ajusta (fit_transform) SOLO con
el set de train, y se aplica (transform) sobre el set de test, que nunca
participa del fit.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)

sys.path.append(os.path.dirname(__file__))
from preprocess import preprocess_corpus

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ag_news")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    # -------------------------------------------------------------
    # 1. Carga de datos: mismo corpus AG News del Módulo 2
    # -------------------------------------------------------------
    print("Cargando AG News (train/test)...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "ag_news_train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "ag_news_test.csv"))
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")

    # -------------------------------------------------------------
    # 2. Reutilización del preprocesamiento del Módulo 2
    #    (regex + normalizacion + lematizacion con spaCy)
    # -------------------------------------------------------------
    print("Preprocesando texto de train...")
    train_df["text_clean"] = preprocess_corpus(train_df["text"].tolist(), batch_size=128)
    print("Preprocesando texto de test...")
    test_df["text_clean"] = preprocess_corpus(test_df["text"].tolist(), batch_size=128)

    X_train_text = train_df["text_clean"]
    y_train = train_df["label"]
    X_test_text = test_df["text_clean"]
    y_test = test_df["label"]

    # -------------------------------------------------------------
    # 3. Vectorizacion TF-IDF
    #    - fit_transform SOLO en train (evita data leakage)
    #    - transform en test
    #    - max_features: limita el vocabulario a los N terminos con
    #      mayor tf-idf promedio, para evitar overfitting a terminos
    #      raros/ruidosos.
    #    - ngram_range=(1, 2): unigramas + bigramas, para capturar
    #      algo de contexto local (ej. "new york") ademas de palabras
    #      sueltas.
    #    - stop_words="english": las stop-words no aportan valor
    #      discriminativo entre clases de noticias y solo agrandan la
    #      matriz esparsa sin necesidad.
    # -------------------------------------------------------------
    MAX_FEATURES = 10_000
    NGRAM_RANGE = (1, 2)

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words="english",
        min_df=2,          # ignora terminos que aparecen en menos de 2 documentos (ruido/typos)
        sublinear_tf=True,  # usa 1+log(tf) en vez de tf crudo, atenua documentos muy repetitivos
    )

    print(f"\nAjustando TfidfVectorizer SOLO sobre train (max_features={MAX_FEATURES}, ngram_range={NGRAM_RANGE})...")
    X_train = vectorizer.fit_transform(X_train_text)   # fit_transform -> SOLO train
    X_test = vectorizer.transform(X_test_text)           # transform -> test (nunca fit)

    print(f"Matriz train: {X_train.shape} | Matriz test: {X_test.shape}")
    print(f"Densidad de la matriz train: {X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.4%}")

    # -------------------------------------------------------------
    # 4. Modelado: Regresion Logistica como baseline
    # -------------------------------------------------------------
    print("\nEntrenando Regresion Logistica...")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, y_train)

    # -------------------------------------------------------------
    # 5. Evaluacion
    # -------------------------------------------------------------
    y_pred = clf.predict(X_test)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)
    print("\n" + report_text)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    print(f"Accuracy: {acc:.4f} | F1-macro: {f1_macro:.4f} | F1-weighted: {f1_weighted:.4f}")

    # Guardar reporte en texto y JSON
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report_text)
    with open(os.path.join(RESULTS_DIR, "classification_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)

    # -------------------------------------------------------------
    # Matriz de confusion
    # -------------------------------------------------------------
    labels_sorted = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    plt.figure(figsize=(6.5, 5.5))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels_sorted)), labels_sorted, rotation=45, ha="right")
    plt.yticks(range(len(labels_sorted)), labels_sorted)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de confusión — TF-IDF + Regresión Logística\n(AG News, test)")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                      ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "matriz_confusion.png"), dpi=150)
    plt.close()

    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(
        os.path.join(RESULTS_DIR, "matriz_confusion.csv")
    )

    # -------------------------------------------------------------
    # Resumen para el README
    # -------------------------------------------------------------
    summary = {
        "max_features": MAX_FEATURES,
        "ngram_range": list(NGRAM_RANGE),
        "vocab_size_final": len(vectorizer.vocabulary_),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_f1": {k: v["f1-score"] for k, v in report_dict.items() if k in labels_sorted},
        "confusion_matrix_labels": labels_sorted,
    }
    with open(os.path.join(RESULTS_DIR, "tfidf_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nListo. Resultados guardados en:", RESULTS_DIR)
    return summary


if __name__ == "__main__":
    main()
