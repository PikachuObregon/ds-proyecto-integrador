# Checkpoint: Clasificador Supervisado con TF-IDF (Módulo 3)

Pipeline de clasificación de texto clásico (no deep learning): texto crudo →
preprocesamiento (reutilizado del Módulo 2) → vectorización TF-IDF →
clasificador → evaluación. Corpus: **AG News** (mismo del Módulo 2).

## Cómo correrlo

```bash
pip install -r requirements.txt
cd src
python tfidf_classifier.py
```

Genera en `results/`: `classification_report.txt/.json`, `matriz_confusion.png/.csv`
y `tfidf_summary.json`.

## Prevención de Data Leakage

El `TfidfVectorizer` se ajusta (`fit_transform`) **únicamente** sobre el set
de **train**. El set de **test** solo pasa por `transform`, nunca por `fit` —
así el vocabulario y los pesos IDF se calculan sin ver ni una palabra del
conjunto de evaluación.

## Vectorización: parámetros de TfidfVectorizer

| Parámetro | Valor elegido | Motivo |
|---|---|---|
| `max_features` | `10000` | Limita el vocabulario a los 10k términos de mayor peso, evitando que términos rarísimos (typos, ruido residual) inflen el modelo y agranden la matriz esparsa sin aportar poder predictivo. |
| `ngram_range` | `(1, 2)` | Unigramas + bigramas. Los bigramas capturan algo de contexto local que se pierde con bag-of-words puro (ej. "new york" vs "new" + "york" sueltos). |
| `stop_words` | `"english"` | Las stop-words no discriminan entre categorías de noticias y solo agrandan la matriz sin aportar señal. |
| `min_df` | `2` | Ignora términos que aparecen en un solo documento (más ruido que señal, típicamente errores de tipeo o nombres propios únicos). |
| `sublinear_tf` | `True` | Usa `1 + log(tf)` en vez de la frecuencia cruda, atenuando el peso de documentos con palabras muy repetidas. |

Con esta configuración, el vocabulario final quedó en **10.000 términos**,
con una matriz de train de densidad ≈0.22% (muy esparsa, como se espera de
TF-IDF con n-gramas).

## Justificación del modelo: Regresión Logística

Se eligió **Regresión Logística** como baseline (frente a Naive Bayes o
SVM) por varias razones concretas para este problema:

- **Rendimiento esperado similar o mejor que Naive Bayes** sobre matrices
  TF-IDF: Naive Bayes multinomial asume independencia entre features, un
  supuesto que se ajusta mejor a conteos crudos que a pesos TF-IDF ya
  normalizados: Regresión Logística no depende de ese supuesto y suele
  sacarle ventaja en este escenario.
- **Más rápida de entrenar que SVM** sobre una matriz de 8.000×10.000: SVM
  con kernel no lineal escala mal en documentos × features; con kernel
  lineal el resultado es comparable a Regresión Logística pero con más
  costo de cómputo y ajuste de hiperparámetros.
- **Interpretabilidad**: los coeficientes por clase permiten inspeccionar
  qué términos empujan la predicción hacia cada categoría, algo valioso
  para explicar el modelo (relevante pensando en aplicaciones futuras
  sobre texto jurídico, donde la interpretabilidad importa).
- Es el baseline estándar de la industria para clasificación de texto con
  TF-IDF antes de pasar a modelos más pesados (que es justamente el rol de
  este checkpoint, como puente hacia el Módulo 4).

## Resultados

| Métrica | Valor |
|---|---|
| Accuracy | **0.8945** |
| F1-macro | **0.8946** |
| F1-weighted | **0.8946** |

| Clase | Precision | Recall | F1-score |
|---|---|---|---|
| Business | 0.85 | 0.86 | 0.85 |
| Sci_Tech | 0.87 | 0.87 | 0.87 |
| Sports | 0.95 | 0.96 | 0.96 |
| World | 0.92 | 0.88 | 0.90 |

## Análisis de la matriz de confusión

![Matriz de confusión](results/matriz_confusion.png)

**Sports es, por lejos, la categoría más fácil de predecir** (F1 = 0.96,
solo 19 errores sobre 500 documentos): el vocabulario deportivo (nombres de
equipos, "match", "score", "win") es muy distintivo y se solapa poco con
las otras clases.

**La confusión más marcada es entre Business y Sci_Tech**: 42 documentos de
Business se predijeron como Sci_Tech, y 45 de Sci_Tech se predijeron como
Business — es, con diferencia, el par de clases con más error cruzado en
toda la matriz. Tiene sentido de cara al dominio: buena parte de las
noticias de "Sci_Tech" en AG News son en realidad noticias de negocios de
empresas tecnológicas (resultados trimestrales, fusiones, cotizaciones de
Microsoft/Google/etc.), lo cual genera un vocabulario compartido (nombres
de empresas, términos financieros) entre ambas categorías.

**World** tiene el recall más bajo (0.88): se confunde un poco con las tres
categorías restantes (25 con Business, 21 con Sci_Tech, 12 con Sports),
consistente con ser la clase más "genérica" — noticias internacionales que
a veces tratan temas económicos, tecnológicos o incluso deportivos
(ej. una noticia sobre unos Juegos Olímpicos con componentes geopolíticos).

En síntesis: el modelo separa muy bien contenido con vocabulario técnico
distintivo (deportes), y tiene más dificultad cuando el tema de fondo se
solapa semánticamente entre categorías (negocios de tecnología, noticias
internacionales con componente económico).

## Requerimientos

Ver `requirements.txt`. Versión de scikit-learn utilizada: `1.8.0`.
