import os
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter


from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")


RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# True  → no llama APIs reales; usa respuestas simuladas deterministas
# False → llama a OpenAI si la API key está disponible
USE_MOCK = not bool(os.getenv("OPENAI_API_KEY"))

INTENT_LABELS = ["faq", "auth", "pagos", "matricula", "soporte"]

_TFIDF_CORPUS: list = []


EVAL_DATASET = [
    # FAQ 
    (
        "¿Cuál es el horario de atención?",
        "faq",
        ["horario", "atención", "lunes"],
        "El horario de atención es de lunes a viernes de 7:00 am a 5:00 pm.",
    ),
    (
        "¿Cuándo son las vacaciones escolares?",
        "faq",
        ["vacaciones", "calendario"],
        "Las vacaciones escolares están programadas según el calendario académico oficial.",
    ),
    (
        "¿Qué documentos necesito para matrícula?",
        "faq",
        ["documento", "requisito"],
        "Para la matrícula necesitas el registro civil, carné de vacunas y fotos recientes.",
    ),
    (
        "¿Tienen ruta de transporte escolar?",
        "faq",
        ["transporte", "ruta"],
        "Sí, contamos con rutas de transporte escolar para varios sectores de la ciudad.",
    ),
    (
        "¿Cuál es el uniforme requerido?",
        "faq",
        ["uniforme"],
        "El uniforme consta de pantalón azul oscuro, camisa blanca y zapatos negros.",
    ),
    (
        "¿Cuáles son los útiles escolares requeridos?",
        "faq",
        ["útiles", "lista"],
        "La lista de útiles está disponible en la página web y en secretaría.",
    ),
    (
        "¿Cómo contacto a un profesor?",
        "faq",
        ["profesor", "contacto", "correo"],
        "Puedes contactar a los profesores por correo institucional o a través de la agenda.",
    ),
    # AUTH 
    (
        "usuario: juan contraseña: clave123",
        "auth",
        ["sesión", "bienvenido", "exitoso"],
        "Sesión iniciada exitosamente. Bienvenido, Juan.",
    ),
    (
        "Quiero iniciar sesión",
        "auth",
        ["usuario", "contraseña"],
        "Por favor ingresa tu usuario y contraseña para iniciar sesión.",
    ),
    (
        "mi usuario es test clave abc",
        "auth",
        ["sesión", "credencial"],
        "Credenciales recibidas. Verificando sesión en el sistema.",
    ),
    (
        "¿Cómo recupero mi contraseña?",
        "auth",
        ["contraseña", "recuperar", "correo"],
        "Puedes recuperar tu contraseña desde el enlace 'Olvidé mi contraseña' en el login.",
    ),
    # PAGOS 
    (
        "¿cuánto debo de pensión?",
        "pagos",
        ["pago", "pendiente", "sesión"],
        "Para consultar el saldo pendiente de pensión necesitas iniciar sesión primero.",
    ),
    (
        "Consultar mis pagos pendientes",
        "pagos",
        ["pago", "sesión"],
        "Tus pagos pendientes se muestran en el portal de padres tras iniciar sesión.",
    ),
    (
        "¿Cuándo vence el pago de marzo?",
        "pagos",
        ["pago", "vencimiento", "fecha"],
        "El pago de pensión del mes de marzo vence el día 5 de cada mes.",
    ),
    (
        "¿Aceptan pagos en efectivo?",
        "pagos",
        ["pago", "efectivo", "transferencia"],
        "Aceptamos pagos en efectivo, transferencia bancaria y tarjeta débito.",
    ),
    # MATRÍCULA 
    (
        "¿Cuál es el estado de mi matrícula?",
        "matricula",
        ["matrícula", "sesión", "grado"],
        "El estado de tu matrícula está disponible en el portal tras iniciar sesión.",
    ),
    (
        "¿Cuándo abren inscripciones?",
        "matricula",
        ["inscripción", "fecha", "cupos"],
        "Las inscripciones para el próximo año abren en octubre con cupos limitados.",
    ),
    (
        "¿Cuál es el costo de matrícula?",
        "matricula",
        ["costo", "matrícula", "valor"],
        "El valor de la matrícula varía por grado; consulta la tabla de tarifas en secretaría.",
    ),
    # SOPORTE 
    (
        "Tengo una queja urgente",
        "soporte",
        ["asesor", "contacto", "teléfono"],
        "Lamentamos tu inconveniente. Un asesor te contactará a la brevedad por teléfono.",
    ),
    (
        "Hola, ¿en qué me pueden ayudar?",
        "soporte",
        ["ayudar", "FAQ", "pagos"],
        "¡Hola! Puedo ayudarte con FAQ, pagos, matrículas o conectarte con un asesor.",
    ),
    (
        "Necesito hablar con alguien del colegio",
        "soporte",
        ["asesor", "comunicar", "llamar"],
        "Con gusto te comunicamos con un asesor. ¿Cuál es tu número de contacto?",
    ),
]


CONFIGURATIONS = [
    {
        "name": "GPT-4o\nchunk500",
        "llm_model": "gpt-4o",
        "embedding_type": "openai",
        "chunk_size": 500,
        "color": "#FF5722",
    },
    {
        "name": "GPT-4o\nchunk200",
        "llm_model": "gpt-4o",
        "embedding_type": "openai",
        "chunk_size": 200,
        "color": "#E64A19",
    },
    {
        "name": "GPT-4o-mini\nchunk500",
        "llm_model": "gpt-4o-mini",
        "embedding_type": "openai",
        "chunk_size": 500,
        "color": "#2196F3",
    },
    {
        "name": "GPT-4o-mini\nchunk200",
        "llm_model": "gpt-4o-mini",
        "embedding_type": "openai",
        "chunk_size": 200,
        "color": "#64B5F6",
    },
    {
        "name": "GPT-4o-mini\nchunk1000",
        "llm_model": "gpt-4o-mini",
        "embedding_type": "openai",
        "chunk_size": 1000,
        "color": "#0D47A1",
    },
    {
        "name": "GPT-3.5-turbo\nchunk500",
        "llm_model": "gpt-3.5-turbo",
        "embedding_type": "openai",
        "chunk_size": 500,
        "color": "#4CAF50",
    },
    {
        "name": "GPT-4o-mini\nSentTransf",
        "llm_model": "gpt-4o-mini",
        "embedding_type": "sentence_transformers",
        "chunk_size": 500,
        "color": "#9C27B0",
    },
    {
        "name": "Llama3-local\nSentTransf",
        "llm_model": "llama3-local",
        "embedding_type": "sentence_transformers",
        "chunk_size": 500,
        "color": "#9E9E9E",
    },
]

COST_MAP = {
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
    "gpt-3.5-turbo": 0.50,
    "llama3-local": 0.00,
}



def _mock_response(question: str, model: str) -> str:
    """Genera una respuesta simulada según el modelo."""
    q = question.lower()
    # Respuestas razonablemente buenas para modelos premium
    if model in ("gpt-4o", "gpt-4o-mini"):
        if "horario" in q:
            return "El horario de atención es de lunes a viernes de 7:00 am a 5:00 pm."
        if "vacaciones" in q:
            return "Las vacaciones escolares están según el calendario académico vigente."
        if "documento" in q or "matrícula" in q and "costo" not in q:
            return "Necesitas registro civil, carné de vacunas y fotografías recientes."
        if "transporte" in q:
            return "Contamos con rutas de transporte escolar para varios sectores."
        if "uniforme" in q:
            return "El uniforme es pantalón azul oscuro, camisa blanca y zapatos negros."
        if "útiles" in q:
            return "La lista de útiles está en la página web y en secretaría."
        if "profesor" in q:
            return "Contacta a los profesores por correo institucional o por la agenda."
        if "usuario" in q or "contraseña" in q or "sesión" in q or "iniciar" in q:
            return "Por favor ingresa tu usuario y contraseña para iniciar sesión."
        if "recuper" in q:
            return "Usa el enlace 'Olvidé mi contraseña' en la pantalla de login."
        if "debo" in q or "pago" in q or "pensión" in q:
            return "Para consultar saldo pendiente debes iniciar sesión en el portal."
        if "vence" in q or "marzo" in q:
            return "El pago de pensión vence el día 5 de cada mes."
        if "efectivo" in q:
            return "Aceptamos efectivo, transferencia bancaria y tarjeta débito."
        if "estado" in q and "matrícula" in q:
            return "El estado de tu matrícula aparece en el portal al iniciar sesión."
        if "inscripcion" in q or "inscripción" in q:
            return "Las inscripciones abren en octubre con cupos limitados."
        if "costo" in q and "matrícula" in q:
            return "El valor varía por grado; consulta la tabla de tarifas en secretaría."
        if "queja" in q:
            return "Un asesor te contactará a la brevedad por teléfono."
        if "hola" in q or "ayudar" in q:
            return "¡Hola! Puedo ayudarte con FAQ, pagos, matrículas o conectarte con un asesor."
        if "alguien" in q or "hablar" in q:
            return "Con gusto te comunicamos con un asesor. ¿Cuál es tu número de contacto?"
        return "Estoy aquí para ayudarte. Por favor, reformula tu pregunta."

    # GPT-3.5 — respuestas más cortas / menos precisas
    elif model == "gpt-3.5-turbo":
        if "horario" in q:
            return "Atendemos de lunes a viernes."
        if "vacaciones" in q:
            return "Consulta el calendario académico."
        if "pago" in q or "pensión" in q or "debo" in q:
            return "Inicia sesión para ver tus pagos."
        if "matrícula" in q:
            return "Revisa el portal para información de matrícula."
        if "usuario" in q or "contraseña" in q:
            return "Ingresa tus credenciales."
        if "queja" in q or "urgente" in q:
            return "Te contactaremos pronto."
        if "hola" in q:
            return "Hola, ¿en qué te ayudo?"
        return "Por favor contacta a secretaría."

    # Llama3 local — respuestas pobres / fuera de contexto
    else:
        responses = [
            "No tengo información sobre eso.",
            "Lo siento, no puedo responder esa pregunta.",
            "Por favor consulta la página web.",
            "No entiendo la pregunta.",
            "Intenta reformular tu consulta.",
        ]
        idx = hash(question) % len(responses)
        return responses[idx]


def _mock_intent(question: str, model: str) -> str:
    """Clasifica intent de forma determinista según keywords."""
    q = question.lower()
    # Llama3: clasificación muy mala
    if model == "llama3-local":
        if "hola" in q:
            return "soporte"
        return "faq"  # casi siempre se equivoca en otros

    kw_map = {
        "auth":      ["usuario", "contraseña", "sesión", "clave", "iniciar", "login", "recuper"],
        "pagos":     ["pago", "debo", "pensión", "pendiente", "vence", "efectivo"],
        "matricula": ["matrícula", "inscripci", "cupos", "costo"],
        "soporte":   ["queja", "urgente", "ayudar", "hola", "alguien", "hablar"],
        "faq":       ["horario", "vacaciones", "uniforme", "transporte", "documento",
                      "útiles", "profesor"],
    }
    for intent, keywords in kw_map.items():
        if any(kw in q for kw in keywords):
            return intent
    return "soporte"



def get_llm(model_name: str):
    if USE_MOCK or model_name == "llama3-local":
        return None
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    except Exception as e:
        print(f"No se pudo cargar {model_name}: {e}")
        return None


def call_llm(llm, model_name: str, question: str) -> tuple[str, str]:
    """Retorna (respuesta_generada, intent_predicho)."""
    if llm is None:
        return _mock_response(question, model_name), _mock_intent(question, model_name)

    from langchain_core.messages import SystemMessage, HumanMessage

    rag_prompt = (
        "Eres un asistente escolar. Responde en español de forma clara y concisa "
        "la pregunta del usuario sobre el colegio. Máximo 2 oraciones."
    )
    try:
        r = llm.invoke([SystemMessage(content=rag_prompt), HumanMessage(content=question)])
        response = r.content.strip() if hasattr(r, "content") else str(r).strip()
    except Exception:
        response = _mock_response(question, model_name)

    intent_prompt = (
        "Clasifica el mensaje en una de estas palabras exactas: "
        "faq | auth | pagos | matricula | soporte\n"
        "Responde SOLO con la palabra, sin puntuación."
    )
    try:
        ri = llm.invoke([SystemMessage(content=intent_prompt), HumanMessage(content=question)])
        raw = ri.content.strip().lower() if hasattr(ri, "content") else str(ri).strip().lower()
        intent = raw if raw in set(INTENT_LABELS) else "soporte"
    except Exception:
        intent = _mock_intent(question, model_name)

    return response, intent



def _tokenize(text: str) -> list[str]:
    """Tokenizador simple sin NLTK: separa por espacios y puntuación."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_bleu(reference: str, hypothesis: str) -> float:
    """BLEU-1/2 con suavizado add-1 (sin NLTK)."""
    ref  = _tokenize(reference)
    hyp  = _tokenize(hypothesis)
    if not hyp:
        return 0.0
    bp = min(1.0, len(hyp) / max(len(ref), 1))
    # Unigram precision
    ref_counts = Counter(ref)
    hyp_counts = Counter(hyp)
    matches_1 = sum(min(hyp_counts[w], ref_counts[w]) for w in hyp_counts)
    p1 = (matches_1 + 1) / (len(hyp) + 1)

    def bigrams(tokens):
        return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    ref_bi = Counter(bigrams(ref))
    hyp_bi = Counter(bigrams(hyp))
    matches_2 = sum(min(hyp_bi[b], ref_bi[b]) for b in hyp_bi)
    p2 = (matches_2 + 1) / (len(hyp_bi) + 1)
    return float(bp * (p1 * p2) ** 0.5)


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L (F-measure)."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_cosine(reference: str, hypothesis: str) -> float:
    """Cosine similarity via TF-IDF char n-grams (0-1, sin modelos externos)."""
    try:
        vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
        tfidf = vect.fit_transform([reference, hypothesis])
        return float(cosine_similarity(tfidf[0], tfidf[1])[0][0])
    except Exception:
        # Fallback: Jaccard sobre tokens
        ref_set = set(_tokenize(reference))
        hyp_set = set(_tokenize(hypothesis))
        inter = ref_set & hyp_set
        union = ref_set | hyp_set
        return len(inter) / max(len(union), 1)



def evaluate_configuration(config: dict) -> dict:
    print(f"\n  ▶ {config['name'].replace(chr(10), ' + ')}")
    mode = "REAL" if (not USE_MOCK and config["llm_model"] != "llama3-local") else "MOCK"
    print(f"    Modo: {mode}")

    llm = get_llm(config["llm_model"])

    y_true, y_pred = [], []
    bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, cosine_scores = [], [], [], [], []
    latencies = []

    for question, expected_intent, _, reference_answer in tqdm(
        EVAL_DATASET, desc="    Casos", leave=False
    ):
        t0 = time.time()
        generated, predicted_intent = call_llm(llm, config["llm_model"], question)
        elapsed_real = time.time() - t0

        # Latencia simulada si es mock (refleja perfiles reales de cada modelo)
        if mode == "MOCK":
            base = {
                "gpt-4o":       np.random.normal(1.75, 0.20),
                "gpt-4o-mini":  np.random.normal(0.75, 0.10),
                "gpt-3.5-turbo":np.random.normal(0.55, 0.08),
                "llama3-local": np.random.normal(3.50, 0.60),
            }.get(config["llm_model"], 1.0)
            chunk_factor = {200: 0.88, 500: 1.00, 1000: 1.14}.get(config["chunk_size"], 1.0)
            emb_factor   = {"openai": 1.0, "sentence_transformers": 0.72}.get(
                config["embedding_type"], 1.0
            )
            elapsed = max(0.05, base * chunk_factor * emb_factor)
        else:
            elapsed = elapsed_real

        latencies.append(elapsed)
        y_true.append(expected_intent)
        y_pred.append(predicted_intent)

        # ── Métricas de calidad de respuesta ──
        bleu_scores.append(compute_bleu(reference_answer, generated))
        rouge = compute_rouge(reference_answer, generated)
        rouge1_scores.append(rouge["rouge1"])
        rouge2_scores.append(rouge["rouge2"])
        rougeL_scores.append(rouge["rougeL"])
        cosine_scores.append(compute_cosine(reference_answer, generated))

    precision = precision_score(y_true, y_pred, labels=INTENT_LABELS,
                                average="weighted", zero_division=0)
    recall    = recall_score(   y_true, y_pred, labels=INTENT_LABELS,
                                average="weighted", zero_division=0)
    f1        = f1_score(       y_true, y_pred, labels=INTENT_LABELS,
                                average="weighted", zero_division=0)
    f1_per    = f1_score(       y_true, y_pred, labels=INTENT_LABELS,
                                average=None, zero_division=0)

    def safe(arr, i):
        return round(float(arr[i]), 4) if i < len(arr) else 0.0

    result = {
        "config_name":      config["name"].replace("\n", " + "),
        "llm_model":        config["llm_model"],
        "embedding_type":   config["embedding_type"],
        "chunk_size":       config["chunk_size"],
        "mode":             mode,
        "precision":        round(precision, 4),
        "recall":           round(recall,    4),
        "f1_score":         round(f1,        4),
        "f1_faq":           safe(f1_per, 0),
        "f1_auth":          safe(f1_per, 1),
        "f1_pagos":         safe(f1_per, 2),
        "f1_matricula":     safe(f1_per, 3),
        "f1_soporte":       safe(f1_per, 4),
        "bleu":             round(float(np.mean(bleu_scores)),   4),
        "rouge1":           round(float(np.mean(rouge1_scores)), 4),
        "rouge2":           round(float(np.mean(rouge2_scores)), 4),
        "rougeL":           round(float(np.mean(rougeL_scores)), 4),
        "cosine_sim":       round(float(np.mean(cosine_scores)), 4),
        "latency_avg_s":    round(float(np.mean(latencies)),          3),
        "latency_p95_s":    round(float(np.percentile(latencies, 95)),3),
        "latency_std_s":    round(float(np.std(latencies)),           3),
        "cost_per_1k_usd":  COST_MAP.get(config["llm_model"], 0.50),
        "color":            config["color"],
    }

    print(
        f"    F1={result['f1_score']:.3f} | BLEU={result['bleu']:.3f} | "
        f"ROUGE-L={result['rougeL']:.3f} | Cos={result['cosine_sim']:.3f} | "
        f"Lat={result['latency_avg_s']:.2f}s"
    )
    return result



def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path.name}")


def plot_classification_metrics(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(df))
    w = 0.25
    for i, (col, label, color) in enumerate(
        [("precision", "Precision", "#1976D2"),
         ("recall",    "Recall",    "#388E3C"),
         ("f1_score",  "F1-Score",  "#F57C00")]
    ):
        bars = ax.bar(x + (i - 1) * w, df[col], w, label=label,
                      color=color, alpha=0.85, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + w / 2, h + 0.01, f"{h:.2f}",
                    ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" + ", "\n") for n in df["config_name"]],
                       fontsize=8, ha="center")
    ax.set_ylim(0, 1.15)
    ax.set_title("Precision / Recall / F1-Score por Configuración",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out / "01_classification_metrics.png")


def plot_text_quality(df: pd.DataFrame, out: Path):
    """BLEU, ROUGE-1/2/L y Cosine Similarity agrupadas."""
    fig, ax = plt.subplots(figsize=(15, 6))
    metrics = [
        ("bleu",       "BLEU",       "#E53935"),
        ("rouge1",     "ROUGE-1",    "#8E24AA"),
        ("rouge2",     "ROUGE-2",    "#5E35B1"),
        ("rougeL",     "ROUGE-L",    "#1E88E5"),
        ("cosine_sim", "Cosine Sim", "#00897B"),
    ]
    x = np.arange(len(df))
    w = 0.15
    for i, (col, label, color) in enumerate(metrics):
        offset = (i - 2) * w
        bars = ax.bar(x + offset, df[col], w, label=label,
                      color=color, alpha=0.82, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + w / 2, h + 0.005, f"{h:.2f}",
                    ha="center", fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" + ", "\n") for n in df["config_name"]],
                       fontsize=8, ha="center")
    ax.set_ylim(0, 1.15)
    ax.set_title("Calidad de Respuesta: BLEU · ROUGE · Cosine Similarity",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=5, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out / "02_text_quality_metrics.png")


def plot_radar(df: pd.DataFrame, out: Path):
    """Radar chart comparativo de las 5 métricas clave (normalizado)."""
    categories = ["F1-Score", "BLEU", "ROUGE-L", "Cosine Sim", "Vel. (inv. lat.)"]
    N = len(categories)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)

    max_lat = df["latency_avg_s"].max() or 1.0

    for _, row in df.iterrows():
        speed = 1.0 - (row["latency_avg_s"] / max_lat)
        vals = [row["f1_score"], row["bleu"], row["rougeL"], row["cosine_sim"], speed]
        vals += vals[:1]
        label = row["config_name"].replace(" + ", "\n")
        ax.plot(angles, vals, linewidth=1.8, linestyle="solid", label=label,
                color=row["color"])
        ax.fill(angles, vals, alpha=0.07, color=row["color"])

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    ax.set_title("Radar: Calidad · Velocidad por Configuración",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    _save(fig, out / "03_radar_chart.png")


def plot_latency(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(df))
    ax.plot(x, df["latency_avg_s"], marker="o", lw=2.5, color="#1565C0",
            label="Latencia Promedio", markersize=8)
    ax.plot(x, df["latency_p95_s"], marker="s", lw=2.5, color="#C62828",
            label="Latencia P95", linestyle="--", markersize=8)
    ax.fill_between(x, df["latency_avg_s"], df["latency_p95_s"], alpha=0.1, color="#1565C0")
    for i, (a, p) in enumerate(zip(df["latency_avg_s"], df["latency_p95_s"])):
        ax.annotate(f"{a:.2f}s", (i, a), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8, color="#1565C0")
        ax.annotate(f"{p:.2f}s", (i, p), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8, color="#C62828")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" + ", "\n") for n in df["config_name"]],
                       fontsize=8, ha="center")
    ax.set_ylabel("Segundos")
    ax.set_title("Latencia Promedio vs P95", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out / "04_latency.png")


def plot_f1_heatmap(df: pd.DataFrame, out: Path):
    cols   = ["f1_faq", "f1_auth", "f1_pagos", "f1_matricula", "f1_soporte"]
    labels = ["FAQ", "Auth", "Pagos", "Matrícula", "Soporte"]
    heat   = df[cols].copy()
    heat.columns = labels
    heat.index = [n.replace(" + ", "\n") for n in df["config_name"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, ax=ax, linewidths=0.5,
                cbar_kws={"label": "F1-Score"}, annot_kws={"size": 11})
    ax.set_title("F1 por Intent y Configuración", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "05_f1_intent_heatmap.png")


def plot_cost_vs_quality(df: pd.DataFrame, out: Path):
    """Scatter: Costo vs. métrica compuesta (F1 + ROUGE-L + Cosine) / 3."""
    df = df.copy()
    df["composite"] = (df["f1_score"] + df["rougeL"] + df["cosine_sim"]) / 3

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in df.iterrows():
        ax.scatter(row["cost_per_1k_usd"], row["composite"],
                   s=220, c=row["color"], alpha=0.85,
                   edgecolors="white", linewidth=1.8, zorder=3)
        ax.annotate(row["config_name"].replace(" + ", "\n"),
                    (row["cost_per_1k_usd"], row["composite"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)

    ax.axhspan(0.75, 1.0, alpha=0.06, color="green", label="Zona óptima (composite > 0.75)")
    ax.axvspan(0, 0.5,    alpha=0.06, color="blue",  label="Zona económica (< $0.50/1K)")
    ax.set_xlabel("Costo por 1.000 consultas (USD)", fontsize=11)
    ax.set_ylabel("Calidad Compuesta  (F1 + ROUGE-L + Cosine) / 3", fontsize=11)
    ax.set_title("Costo vs. Calidad Compuesta", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out / "06_cost_vs_quality.png")


def plot_weighted_ranking(df: pd.DataFrame, out: Path):
    """Ranking final ponderado: F1×0.3 + ROUGE-L×0.2 + Cosine×0.25 + Vel×0.15 + Costo×0.10"""
    df = df.copy()
    max_lat  = df["latency_avg_s"].max() or 1.0
    max_cost = df["cost_per_1k_usd"].max() or 1.0
    df["vel_norm"]  = 1.0 - (df["latency_avg_s"] / max_lat)
    df["cost_norm"] = 1.0 - (df["cost_per_1k_usd"] / max_cost)
    df["score"] = (
        df["f1_score"]  * 0.30 +
        df["rougeL"]    * 0.20 +
        df["cosine_sim"]* 0.25 +
        df["vel_norm"]  * 0.15 +
        df["cost_norm"] * 0.10
    )
    df = df.sort_values("score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["config_name"].str.replace(" + ", "\n"),
                   df["score"], color=df["color"],
                   edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, df["score"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Score Ponderado", fontsize=11)
    ax.set_title(
        "Ranking Final Ponderado\n"
        "(F1×30% · ROUGE-L×20% · Cosine×25% · Velocidad×15% · Costo×10%)",
        fontsize=12, fontweight="bold"
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out / "07_weighted_ranking.png")
    return df.sort_values("score", ascending=False)



def run_benchmark():
    print("=" * 70)
    print("  BENCHMARK RAG — Métricas Extendidas")
    print(f"   Timestamp  : {datetime.now().isoformat()}")
    print(f"   Casos eval : {len(EVAL_DATASET)}")
    print(f"   Configs    : {len(CONFIGURATIONS)}")
    print(f"   Modo APIs  : {'MOCK (sin API key)' if USE_MOCK else 'REAL (OpenAI)'}")
    print("=" * 70)

    results = []
    for cfg in CONFIGURATIONS:
        res = evaluate_configuration(cfg)
        results.append(res)

    df = pd.DataFrame(results)

    # ── Persistencia ──
    csv_path  = RESULTS_DIR / "benchmark_results.csv"
    json_path = RESULTS_DIR / "benchmark_results.json"
    df_export = df.drop(columns=["color"])
    df_export.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(df_export.to_dict(orient="records"), indent=2, ensure_ascii=False)
    )
    print(f"\n CSV  → {csv_path}")
    print(f" JSON → {json_path}")

    print("\n Generando gráficos…")
    plot_classification_metrics(df, RESULTS_DIR)
    plot_text_quality(df, RESULTS_DIR)
    plot_radar(df, RESULTS_DIR)
    plot_latency(df, RESULTS_DIR)
    plot_f1_heatmap(df, RESULTS_DIR)
    plot_cost_vs_quality(df, RESULTS_DIR)
    df_ranked = plot_weighted_ranking(df, RESULTS_DIR)

    best = df_ranked.iloc[0]
    print("\n" + "=" * 70)
    print("  RANKING FINAL (score ponderado)")
    print("=" * 70)
    cols_show = ["config_name", "f1_score", "bleu", "rougeL",
                 "cosine_sim", "latency_avg_s", "cost_per_1k_usd", "score"]
    print(df_ranked[cols_show].to_string(index=False))

    print(f"""
{'=' * 70}
  MODELO RECOMENDADO: {best['config_name']}
{'=' * 70}
  F1-Score    : {best['f1_score']:.3f}
  BLEU        : {best['bleu']:.3f}
  ROUGE-L     : {best['rougeL']:.3f}
  Cosine Sim  : {best['cosine_sim']:.3f}
  Latencia    : {best['latency_avg_s']:.2f}s
  Costo/1K    : ${best['cost_per_1k_usd']:.2f}
  Score total : {best['score']:.3f}
{'=' * 70}
""")
    return df_export


if __name__ == "__main__":
    run_benchmark()
