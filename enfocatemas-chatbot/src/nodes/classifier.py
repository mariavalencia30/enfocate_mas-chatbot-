import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.memory.state import ChatState

# Prompt del sistema para clasificación 
CLASSIFIER_SYSTEM_PROMPT = """
Eres un clasificador de intenciones para el chatbot de la institución 
educativa Enfócate Más. Analiza el mensaje del usuario y responde 
ÚNICAMENTE con una de estas palabras (sin puntuación, sin explicación):

  faq        → preguntas sobre la institución, programas, cursos, modalidades, 
               horarios, calendario, normas, información general, contacto, 
               admisiones, beneficios, servicios, cualquier duda informativa
  auth       → el usuario quiere iniciar sesión, pide sus credenciales o las valida
  pagos      → consultas sobre pagos, deudas, recibos, saldos pendientes
  matricula  → inscripciones, renovaciones, estado de matrícula
  soporte    → solo si es una queja, reclamo, reporte de error grave, o solicitud de agente humano

Responde solo con la palabra clave, nada más.
""".strip()


def intent_classifier_node(state: ChatState) -> dict:
    """Nodo LangGraph: clasifica la intención del último mensaje del usuario."""

    if state.get("auth_status") == "recovering":
        return {"intent": "auth"}

    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None
    )

    if last_ai and "¿Cómo te llamas?" in last_ai.content:
        return {"intent": "auth"}

    context = state.get("context", {})
    recovery = context.get("recovery", {}) if context else {}
    recovery_step = recovery.get("step", 0) if recovery else 0

    if recovery_step > 0:
        return {"intent": "auth"}

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )

    if not last_human:
        return {"intent": "soporte"}

    text_lower = last_human.content.lower()

    if state.get("auth_status") != "ok":
        if (
            "no me acuerdo" in text_lower
            or "olvide" in text_lower
            or "olvidé" in text_lower
            or "no puedo" in text_lower
            or "recuperar" in text_lower
            or "olvide" in text_lower
        ):
            return {"intent": "auth"}

    if state.get("auth_status") == "ok":
        import re

        if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", last_human.content):
            return {"intent": "pagos"}

        if any(
            palabra in text_lower
            for palabra in [
                "pago",
                "debo",
                "saldo",
                "asistencia",
                "asistí",
                "asistiendo",
                "asistio",
                "número",
                "numeros",
                "teléfono",
                "telefono",
                "registrado",
                "cuánto",
                "cuanto",
                "información",
                "info",
                "actualizar",
                "actualizalo",
                "actualizame",
                "cambiar",
                "correo",
                "email",
                "datos",
                "personal",
                "contraseña",
                "clave",
                "nombre",
                "hermano",
                "hermana",
                "clase",
                "clases",
                "colegio",
                "grupo",
                "van",
                "voy",
                "llevo",
            ]
        ):
            return {"intent": "pagos"}

    import re

    digits = re.sub(r"\D", "", last_human.content)
    is_only_phone = len(
        text_lower.replace(" ", "").replace("+", "").replace("-", "").replace(".", "")
    ) == len(digits)
    has_phone_pattern = (
        is_only_phone
        and len(digits) >= 10
        and ("3" in digits[-10:] or "3" in digits[-11:-1])
    )

    if has_phone_pattern:
        if state.get("auth_status") == "ok":
            return {"intent": "pagos"}
        return {"intent": "auth"}

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    result = llm.invoke(
        [
            SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(content=last_human.content),
        ]
    )

    raw_intent = result.content.strip().lower()

    valid_intents = {"faq", "auth", "pagos", "matricula", "soporte"}
    intent = raw_intent if raw_intent in valid_intents else "soporte"

    return {"intent": intent}
