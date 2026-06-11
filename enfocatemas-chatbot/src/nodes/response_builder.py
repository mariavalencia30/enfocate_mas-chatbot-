from langchain_core.messages import AIMessage
from src.memory.state import ChatState

MAX_WHATSAPP_CHARS = 1600


def response_builder_node(state: ChatState) -> dict:
    """
    Toma el último AIMessage del estado y lo formatea
    según el canal de comunicación.
    """
    channel = state.get("channel", "web")

    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None
    )

    if not last_ai:
        text = "Lo siento, no pude generar una respuesta. Por favor intenta de nuevo."
    else:
        text = last_ai.content

    # WhatsApp: límite de caracteres
    if channel == "whatsapp" and len(text) > MAX_WHATSAPP_CHARS:
        text = text[:MAX_WHATSAPP_CHARS - 3] + "..."

    return {"response_text": text}
