from typing import TypedDict, Optional
from langchain_core.messages import BaseMessage


class ChatState(TypedDict):
    """
    Estado conversacional compartido entre todos los nodos.
    LangGraph lo pasa de nodo en nodo, acumulando información.
    """

    # Historial de mensajes (memoria conversacional) 
    messages: list[BaseMessage]

    # Identificación de sesión 
    user_id: str          # ID interno del usuario (si autenticado)
    session_id: str       # UUID único por conversación
    channel: str          # "whatsapp" | "web"

    # Clasificación de intención 
    intent: str           # faq | auth | pagos | matricula | soporte

    # Estado de autenticación 
    auth_status: str      # "pending" | "ok" | "fail" | "locked"
    auth_attempts: int    # contador de intentos fallidos (máx. 3)

    # Contexto recuperado (datos de negocio) 
    context: dict      

    # Respuesta final formateada 
    response_text: str    

    # Escalación humana 
    escalate: bool        # True si debe derivarse a agente humano

    # Registro de la interacción 
    logs: list[dict]      # entradas de log de esta sesión
