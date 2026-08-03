"""
eda.py
------
Aplica el pipeline de preprocesamiento sobre el corpus de entrenamiento
(AG News) y genera el EDA técnico completo pedido en la consigna:

  - Histograma de tokens por documento + percentil 95
  - Top 20 bi-gramas y top 20 tri-gramas más frecuentes
  - Top 50 palabras más frecuentes + cuántas son stop-words
  - Gráfico de distribución de clases

Guarda resultados intermedios (CSV/JSON/PNG) en results/ para armar el PDF.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

sys.path.append(os.path.dirname(__file__))
from preprocess import preprocess_corpus

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ag_news")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("Cargando corpus...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "ag_news_train.csv"))
    print(f"Documentos en train: {len(train_df)}")

    # -------------------------------------------------------------
    # 1. Aplicar el pipeline de preprocesamiento sobre el corpus
    # -------------------------------------------------------------
    print("Aplicando preprocess_text (regex + normalizacion + lematizacion con spaCy)...")
    train_df["text_clean"] = preprocess_corpus(train_df["text"].tolist(), batch_size=128)
    train_df["n_tokens"] = train_df["text_clean"].apply(lambda t: len(t.split()))

    train_df[["text", "text_clean", "n_tokens", "label"]].to_csv(
        os.path.join(RESULTS_DIR, "corpus_procesado.csv"), index=False
    )

    # -------------------------------------------------------------
    # 2. Histograma de longitud de tokens + percentil 95
    # -------------------------------------------------------------
    p95 = int(np.percentile(train_df["n_tokens"], 95))
    p50 = int(np.percentile(train_df["n_tokens"], 50))
    print(f"Percentil 50 (mediana) de tokens/doc: {p50}")
    print(f"Percentil 95 de tokens/doc (max_len sugerido): {p95}")

    plt.figure(figsize=(8, 5))
    plt.hist(train_df["n_tokens"], bins=40, color="#4C72B0", edgecolor="white")
    plt.axvline(p95, color="crimson", linestyle="--", label=f"Percentil 95 = {p95}")
    plt.xlabel("Cantidad de tokens por documento")
    plt.ylabel("Cantidad de documentos")
    plt.title("Distribución de longitud de tokens (AG News, train, post-preprocesamiento)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "hist_longitud_tokens.png"), dpi=150)
    plt.close()

    # -------------------------------------------------------------
    # 3. Top 20 bi-gramas y top 20 tri-gramas
    # -------------------------------------------------------------
    def top_ngrams(corpus, ngram_range, top_n=20):
        vec = CountVectorizer(ngram_range=ngram_range)
        X = vec.matrix if False else vec.fit_transform(corpus)
        freqs = np.asarray(X.sum(axis=0)).ravel()
        vocab = vec.get_feature_names_out()
        order = np.argsort(freqs)[::-1][:top_n]
        return pd.DataFrame({"ngram": vocab[order], "frecuencia": freqs[order]})

    bigrams_df = top_ngrams(train_df["text_clean"], (2, 2), 20)
    trigrams_df = top_ngrams(train_df["text_clean"], (3, 3), 20)
    bigrams_df.to_csv(os.path.join(RESULTS_DIR, "top_bigramas.csv"), index=False)
    trigrams_df.to_csv(os.path.join(RESULTS_DIR, "top_trigramas.csv"), index=False)

    print("\nTop 10 bigramas:")
    print(bigrams_df.head(10).to_string(index=False))
    print("\nTop 10 trigramas:")
    print(trigrams_df.head(10).to_string(index=False))

    # -------------------------------------------------------------
    # 4. Top 50 palabras mas frecuentes + cuantas son stop-words
    # -------------------------------------------------------------
    vec_words = CountVectorizer(ngram_range=(1, 1))
    Xw = vec_words.fit_transform(train_df["text_clean"])
    freqs_w = np.asarray(Xw.sum(axis=0)).ravel()
    vocab_w = vec_words.get_feature_names_out()
    order_w = np.argsort(freqs_w)[::-1][:50]
    top_words_df = pd.DataFrame({"palabra": vocab_w[order_w], "frecuencia": freqs_w[order_w]})
    top_words_df["es_stopword"] = top_words_df["palabra"].isin(ENGLISH_STOP_WORDS)
    n_stopwords = int(top_words_df["es_stopword"].sum())
    top_words_df.to_csv(os.path.join(RESULTS_DIR, "top_50_palabras.csv"), index=False)

    print(f"\nDe las top 50 palabras mas frecuentes, {n_stopwords} son stop-words estandar (sklearn).")

    # -------------------------------------------------------------
    # 5. Distribucion de clases
    # -------------------------------------------------------------
    class_counts = train_df["label"].value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    bars = plt.bar(class_counts.index, class_counts.values, color="#55A868")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad de documentos")
    plt.title("Distribución de clases - AG News (train)")
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 20, str(int(b.get_height())),
                  ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "distribucion_clases.png"), dpi=150)
    plt.close()

    # -------------------------------------------------------------
    # Guardar resumen para el PDF
    # -------------------------------------------------------------
    summary = {
        "n_documentos_train": int(len(train_df)),
        "n_clases": int(train_df["label"].nunique()),
        "clases": sorted(train_df["label"].unique().tolist()),
        "class_counts": class_counts.to_dict(),
        "percentil_50_tokens": p50,
        "percentil_95_tokens": p95,
        "min_tokens": int(train_df["n_tokens"].min()),
        "max_tokens": int(train_df["n_tokens"].max()),
        "n_stopwords_en_top50": n_stopwords,
    }
    with open(os.path.join(RESULTS_DIR, "eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nEDA completo. Resultados guardados en:", RESULTS_DIR)
    return summary


if __name__ == "__main__":
    main()
