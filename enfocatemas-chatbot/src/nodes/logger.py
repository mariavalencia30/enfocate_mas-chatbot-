import os
import json
from datetime import datetime
from src.memory.state import ChatState

LOG_FILE = os.path.join(os.path.dirname(__file__), "../../logs/interactions.jsonl")


def logger_node(state: ChatState) -> dict:
    """Registra la interacción actual en el archivo de logs JSONL."""

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": state.get("session_id", "unknown"),
        "user_id": state.get("user_id", "anonymous"),
        "channel": state.get("channel", "unknown"),
        "intent": state.get("intent", "unknown"),
        "auth_status": state.get("auth_status", "pending"),
        "escalated": state.get("escalate", False),
        "message_count": len(state.get("messages", [])),
    }

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[LOGGER NODE] Error escribiendo log: {e}")

    # Agregar al log interno del estado
    logs = state.get("logs", [])
    logs.append(entry)

    return {"logs": logs}
