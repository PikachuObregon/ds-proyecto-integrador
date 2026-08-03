"""
build_report.py
----------------
Genera el PDF final del informe EDA técnico (Entrega 2 - Módulo 2)
a partir de los resultados generados por eda.py.
"""

import json
import os

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "EDA_NLP_Mariano.pdf")  # se renombra al final con el apellido

with open(os.path.join(RESULTS_DIR, "eda_summary.json")) as f:
    summary = json.load(f)

bigrams_df = pd.read_csv(os.path.join(RESULTS_DIR, "top_bigramas.csv"))
trigrams_df = pd.read_csv(os.path.join(RESULTS_DIR, "top_trigramas.csv"))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Justify", parent=styles["Normal"], alignment=4, spaceAfter=8, leading=14))
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], spaceBefore=14, spaceAfter=8))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8.5, textColor=colors.grey))

story = []

# ---------------------------------------------------------------------
# Portada / encabezado
# ---------------------------------------------------------------------
story.append(Paragraph("EDA Técnico — Pipeline de Preprocesamiento y Diagnóstico (PPD)", styles["Title"]))
story.append(Paragraph("Proyecto Integrador — Data Science III: NLP &amp; Deep Learning", styles["Normal"]))
story.append(Paragraph("Coderhouse — Entrega 2 (Módulo 2)", styles["Normal"]))
story.append(Spacer(1, 6))
story.append(Paragraph(f"Dataset: AG News &nbsp;|&nbsp; Documentos analizados (train): {summary['n_documentos_train']} "
                        f"&nbsp;|&nbsp; Clases: {summary['n_clases']}", styles["Small"]))
story.append(Spacer(1, 16))

# ---------------------------------------------------------------------
# 1. Resumen de pasos de preprocesamiento
# ---------------------------------------------------------------------
story.append(Paragraph("1. Resumen de pasos de preprocesamiento aplicados", styles["H1"]))
story.append(Paragraph(
    "Se implementó la función <b>preprocess_text(text)</b>, que integra las siguientes etapas, "
    "aplicadas en orden sobre cada documento del corpus de entrenamiento:", styles["Justify"]
))

pasos = [
    ("Decodificación de entidades HTML", "Se detectó que ~4.5% de los documentos de AG News contienen "
     "entidades HTML sin decodificar (ej. <font face='Courier'>&amp;lt;FONT&amp;gt;</font>, producto de tags de "
     "cotización bursátil incrustados en las noticias originales). Se decodifican con html.unescape() "
     "antes de cualquier otra limpieza."),
    ("Limpieza con Regex", "Remoción de URLs (http/https/www), tags HTML remanentes y todo carácter "
     "no alfabético (dígitos, signos de puntuación, símbolos)."),
    ("Normalización", "Conversión a minúsculas y colapso de espacios múltiples/saltos de línea a un "
     "único espacio."),
    ("Lematización con spaCy", "Se usa el modelo en_core_web_sm para reducir cada token a su lema "
     "(ej. 'appointed' → 'appoint', 'companies' → 'company'), reduciendo la dispersión del vocabulario."),
]
for titulo, desc in pasos:
    story.append(Paragraph(f"<b>• {titulo}:</b> {desc}", styles["Justify"]))

story.append(Paragraph(
    "<b>Nota sobre el corpus:</b> este es el primer checkpoint donde se trabaja sobre el corpus real que "
    "se usará en el resto del programa (Módulos 3, 4 y Proyecto Final): AG News, dataset provisto por la "
    "cátedra, con 8.000 documentos de entrenamiento distribuidos en 4 clases perfectamente balanceadas "
    "(2.000 documentos cada una: Business, Sci_Tech, Sports, World).", styles["Justify"]
))

# ---------------------------------------------------------------------
# 2. Distribución de longitud de tokens
# ---------------------------------------------------------------------
story.append(Paragraph("2. Distribución de longitud de tokens", styles["H1"]))
story.append(Paragraph(
    f"Tras el preprocesamiento, la longitud de los documentos (en tokens) tiene una mediana de "
    f"<b>{summary['percentil_50_tokens']} tokens</b>, con un mínimo de {summary['min_tokens']} y un "
    f"máximo de {summary['max_tokens']}. El <b>percentil 95 es de {summary['percentil_95_tokens']} tokens</b>, "
    f"valor que se toma como referencia de <i>max_len</i> para el padding/truncation en los modelos de "
    f"los próximos módulos: cubre al 95% del corpus sin generar secuencias innecesariamente largas.",
    styles["Justify"]
))
story.append(Image(os.path.join(RESULTS_DIR, "hist_longitud_tokens.png"), width=15 * cm, height=9.4 * cm))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "Al ser AG News un dataset de titulares/resúmenes cortos, la distribución está concentrada en un "
    "rango relativamente angosto (la mayoría de los documentos entre 20 y 55 tokens), sin una cola larga "
    "extrema, lo cual simplifica la elección de max_len para el módulo de fine-tuning.",
    styles["Justify"]
))

story.append(PageBreak())

# ---------------------------------------------------------------------
# 3. N-gramas más frecuentes
# ---------------------------------------------------------------------
story.append(Paragraph("3. Bi-gramas y tri-gramas más frecuentes", styles["H1"]))
story.append(Paragraph(
    "Se extrajeron los N-gramas más frecuentes del corpus lematizado con CountVectorizer "
    "(scikit-learn). A continuación, el top 20 de bi-gramas y tri-gramas:", styles["Justify"]
))

def df_to_table(df, col_names, col_widths):
    header = col_names
    data = [header] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
    ]))
    return t

story.append(Paragraph("Top 20 bi-gramas", styles["H2"]))
story.append(df_to_table(bigrams_df, ["Bi-grama", "Frecuencia"], [8 * cm, 4 * cm]))
story.append(Spacer(1, 10))

story.append(Paragraph("Top 20 tri-gramas", styles["H2"]))
story.append(df_to_table(trigrams_df, ["Tri-grama", "Frecuencia"], [8 * cm, 4 * cm]))
story.append(Spacer(1, 10))

story.append(Paragraph(
    "<b>Interpretación:</b> la mayoría de los bi-gramas top están dominados por combinaciones de "
    "preposiciones/artículos ('of the', 'in the', 'for the'), esperable en cualquier corpus en inglés y "
    "sin valor semántico distintivo por sí solas. Entre los patrones con más carga informativa aparecen "
    "<i>'new york'</i> (dateline geográfico típico de cables de noticias) y <i>'ap ap'</i> "
    "(duplicación de la marca de la agencia Associated Press en el texto fuente). En los tri-gramas se "
    "destacan <i>'new york reuters'</i> y <i>'the united states'</i>, ambos datelines/entidades geográficas "
    "recurrentes en noticias de agencia, junto con secuencias como <i>'quote profile research'</i>, "
    "residuo de bloques de cotización bursátil embebidos en el texto original de algunas noticias de "
    "Business/Sci_Tech. No se observan tags HTML ni entidades sin decodificar entre los N-gramas más "
    "frecuentes, lo cual confirma que la etapa de limpieza fue efectiva.",
    styles["Justify"]
))

story.append(PageBreak())

# ---------------------------------------------------------------------
# 4. Top 50 palabras y stop-words
# ---------------------------------------------------------------------
story.append(Paragraph("4. Palabras más frecuentes y stop-words", styles["H1"]))
story.append(Paragraph(
    f"De las 50 palabras más frecuentes del corpus lematizado, <b>{summary['n_stopwords_en_top50']} de 50 "
    f"({summary['n_stopwords_en_top50']*2}%) son stop-words</b> estándar del inglés (según el listado de "
    f"scikit-learn). Es un resultado esperado: en cualquier corpus en lenguaje natural, las palabras "
    f"funcionales (artículos, preposiciones, conjunciones) dominan el ranking de frecuencia absoluta. "
    f"Esto valida la decisión de <b>no</b> remover stop-words en esta etapa de EDA (para poder "
    f"diagnosticarlas), reservando su filtrado para la etapa de vectorización (Módulo 3), donde sí "
    f"afectan negativamente a métodos como TF-IDF si no se controlan.", styles["Justify"]
))

story.append(Paragraph("5. Distribución de clases", styles["H1"]))
story.append(Paragraph(
    "El corpus de entrenamiento de AG News está <b>perfectamente balanceado</b>: 2.000 documentos por "
    "cada una de las 4 clases (Business, Sci_Tech, Sports, World). Esto simplifica el entrenamiento de "
    "los módulos siguientes: no va a ser necesario aplicar estrategias de balanceo (undersampling, "
    "oversampling, class weights) y accuracy va a ser una métrica confiable, sin necesidad de recurrir "
    "obligatoriamente a F1 macro/weighted para compensar desbalanceo (aunque igualmente se recomienda "
    "reportarlas como buena práctica).", styles["Justify"]
))
story.append(Image(os.path.join(RESULTS_DIR, "distribucion_clases.png"), width=13 * cm, height=9.3 * cm))

story.append(PageBreak())

# ---------------------------------------------------------------------
# 6. Conclusión
# ---------------------------------------------------------------------
story.append(Paragraph("6. Conclusión sobre la preparación del dato para el modelo final", styles["H1"]))
conclusion = (
    "El corpus AG News, tras el pipeline de preprocesamiento aplicado, queda en condiciones adecuadas "
    "para el modelado: el principal desafío detectado fue la presencia de entidades HTML sin decodificar "
    "en aproximadamente el 4.5% de los documentos, originadas en bloques de cotización bursátil "
    "embebidos en las noticias de fuente. Sin una decodificación explícita previa a la remoción de tags, "
    "ese ruido se filtraba directamente al vocabulario final, contaminando los N-gramas con tokens sin "
    "valor semántico ('lt', 'gt', 'href'). Una vez corregido, los N-gramas más frecuentes reflejan "
    "patrones lingüísticos genuinos del corpus (preposiciones comunes, datelines geográficos y menciones "
    "a agencias de noticias), sin residuos de marcado. "
    "En cuanto a la distribución de clases, el corpus está perfectamente balanceado (2.000 documentos "
    "por clase), lo que evita la necesidad de técnicas de re-balanceo en los módulos de modelado "
    "posteriores y permite usar accuracy como métrica principal sin perder representatividad. "
    "Respecto a la longitud, los documentos son mayormente cortos (mediana de 37 tokens, percentil 95 en "
    "53), coherente con el formato de titular/resumen de AG News; este valor se toma como max_len de "
    "referencia para el truncation/padding en el fine-tuning del Módulo 4, minimizando el cómputo "
    "desperdiciado en padding sin descartar información relevante de la cola larga del corpus."
)
story.append(Paragraph(conclusion, styles["Justify"]))

n_palabras_conclusion = len(conclusion.split())
story.append(Spacer(1, 6))
story.append(Paragraph(f"({n_palabras_conclusion} palabras)", styles["Small"]))

doc = SimpleDocTemplate(
    OUTPUT_PATH, pagesize=letter,
    topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2.2 * cm, rightMargin=2.2 * cm
)
doc.build(story)
print(f"PDF generado en: {OUTPUT_PATH}")
print(f"Palabras en la conclusion: {n_palabras_conclusion} (limite: 300)")
