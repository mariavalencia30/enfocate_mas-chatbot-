from langchain_core.messages import HumanMessage, AIMessage
from src.memory.state import ChatState

ESCALATION_KEYWORDS = [
    "urgente", "queja", "reclamo", "problema grave", "error",
    "ayuda humana", "hablar con alguien", "hablar con una persona",
    "no funciona", "falla", "mal servicio"
]


def soporte_node(state: ChatState) -> dict:
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None
    )
    query = last_human.content.lower() if last_human else ""

    should_escalate = any(kw in query for kw in ESCALATION_KEYWORDS)

    if should_escalate:
        response = AIMessage(
            content=(
                " Entiendo que necesitas ayuda urgente. Voy a conectarte con "
                "un asesor de la institución.\n\n"
                " También puedes comunicarte directamente:\n"
                "• Teléfono: [número institucional]\n"
                "• Correo: [correo institucional]\n"
                "• Horario de atención: Lunes a Viernes, 7am–5pm"
            )
        )
        return {
            "escalate": True,
            "messages": state["messages"] + [response],
        }
    else:
        response = AIMessage(
            content=(
                "Hola 👋, soy el asistente virtual de *Enfócate Más*. "
                "Puedo ayudarte con:\n\n"
                "• 📚 Preguntas frecuentes sobre la institución\n"
                "• 🔐 Inicio de sesión\n"
                "• 💳 Consulta de pagos y saldos\n"
                "• 📝 Estado de matrícula\n\n"
                "¿Sobre qué tema necesitas información?"
            )
        )
        return {
            "escalate": False,
            "messages": state["messages"] + [response],
        }
