"""
preprocess.py
-------------
Pipeline de Preprocesamiento y Diagnóstico (PPD) - Módulo 2.

Integra:
  - Limpieza con Regex (URLs, HTML, caracteres no alfabéticos, espacios extra)
  - Normalización (lowercase, remoción de ruido)
  - Lematización con spaCy (en_core_web_sm)
"""

import html
import re
import spacy

# Cargamos el modelo de spaCy una sola vez, deshabilitando componentes que no
# necesitamos (parser, ner) para acelerar el procesamiento del corpus completo.
_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Limpieza con Regex + normalización (lowercase, remoción de ruido)."""
    # AG News trae ~4.5% de documentos con entidades HTML sin decodificar
    # (ej. "&lt;FONT ...&gt;"); hay que decodificarlas ANTES de remover tags,
    # o el regex de tags nunca las va a reconocer como tales.
    text = html.unescape(text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = text.lower()
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def lemmatize(text: str) -> str:
    """Lematiza el texto ya limpio usando spaCy."""
    doc = _nlp(text)
    tokens = [tok.lemma_ for tok in doc if tok.lemma_.strip()]
    return " ".join(tokens)


def preprocess_text(text: str) -> str:
    """Pipeline completo: limpieza -> normalización -> lematización."""
    return lemmatize(clean_text(text))


def preprocess_corpus(texts, batch_size: int = 64, n_process: int = 1):
    """
    Versión vectorizada (más rápida) para procesar un corpus completo,
    usando nlp.pipe en batch en lugar de llamar preprocess_text() texto a texto.
    """
    cleaned = [clean_text(t) for t in texts]
    results = []
    for doc in _nlp.pipe(cleaned, batch_size=batch_size, n_process=n_process):
        tokens = [tok.lemma_ for tok in doc if tok.lemma_.strip()]
        results.append(" ".join(tokens))
    return results


if __name__ == "__main__":
    ejemplo = "Red Hat Appoints New CFO! Check http://example.com for details. <b>Big news</b>."
    print("Original:  ", ejemplo)
    print("Procesado: ", preprocess_text(ejemplo))
