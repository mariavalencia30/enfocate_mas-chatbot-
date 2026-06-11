import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from src.nodes.auth import _extract_credentials, _extract_phone
from src.nodes.response_builder import response_builder_node


class TestRegressionAuth:
    """Comportamientos de auth que NUNCA deben cambiar."""

    def test_hash_never_exposed_in_any_response(self):
        """CRÍTICO: El hash bcrypt jamás debe aparecer en ninguna respuesta."""
        from src.nodes.auth import auth_node

        state = {
            "messages": [HumanMessage(content="+57 3874290099")],
            "user_id": "", "session_id": "reg_001", "channel": "web",
            "intent": "auth", "auth_status": "pending", "auth_attempts": 0,
            "context": {}, "response_text": "", "escalate": False, "logs": [],
        }

        result = auth_node(state)

        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage):
                assert "$2b$" not in msg.content
                assert "hash" not in msg.content.lower()
                assert "CONTRASENA" not in msg.content

    def test_max_attempts_always_3(self, tmp_path):
        """El límite de intentos debe ser siempre 3."""
        from passlib.hash import bcrypt as bc
        import pandas as pd
        from src.nodes.auth import auth_node, MAX_ATTEMPTS

        assert MAX_ATTEMPTS == 3, "El límite de intentos cambió — revisar seguridad"

    def test_locked_state_is_permanent_per_session(self):
        """Una cuenta bloqueada no se desbloquea dentro de la misma sesión."""
        from src.nodes.auth import auth_node

        state = {
            "messages": [HumanMessage(content="usuario: test contraseña: pass")],
            "user_id": "", "session_id": "reg_lock", "channel": "web",
            "intent": "auth", "auth_status": "locked", "auth_attempts": 3,
            "context": {}, "response_text": "", "escalate": False, "logs": [],
        }

        result = auth_node(state)
        assert result.get("auth_status") == "locked"

    def test_credential_extraction_formats_stable(self):
        """Los formatos de extracción de credenciales son contratos estables."""
        from src.nodes.auth import _extract_credentials
        cases = [
            ("correo: juan@mail.com contraseña: 1234", ("juan@mail.com", "1234")),
            ("email: maria@test.com clave: secreto",     ("maria@test.com", "secreto")),
        ]
        for text, expected in cases:
            user, pwd = _extract_credentials(text)
            assert (user, pwd) == expected, f"Formato roto: '{text}'"


class TestRegressionResponseBuilder:
    """Contratos del response_builder que no deben cambiar."""

    def test_whatsapp_truncation_at_1600(self):
        """WhatsApp: respuestas siempre truncadas a 1600 chars."""
        long_text = "A" * 2000
        state = {
            "messages": [AIMessage(content=long_text)],
            "channel": "whatsapp",
            "user_id": "", "session_id": "", "intent": "",
            "auth_status": "", "auth_attempts": 0, "context": {},
            "response_text": "", "escalate": False, "logs": [],
        }
        result = response_builder_node(state)
        assert len(result["response_text"]) <= 1600

    def test_web_channel_no_truncation(self):
        """Web: respuestas largas no se truncan."""
        long_text = "B" * 2000
        state = {
            "messages": [AIMessage(content=long_text)],
            "channel": "web",
            "user_id": "", "session_id": "", "intent": "",
            "auth_status": "", "auth_attempts": 0, "context": {},
            "response_text": "", "escalate": False, "logs": [],
        }
        result = response_builder_node(state)
        assert len(result["response_text"]) == 2000

    def test_empty_messages_returns_fallback(self):
        """Sin mensajes AI, el builder debe retornar fallback, no lanzar excepción."""
        state = {
            "messages": [],
            "channel": "web",
            "user_id": "", "session_id": "", "intent": "",
            "auth_status": "", "auth_attempts": 0, "context": {},
            "response_text": "", "escalate": False, "logs": [],
        }
        result = response_builder_node(state)
        assert isinstance(result["response_text"], str)
        assert len(result["response_text"]) > 0


class TestRegressionIntentRouting:
    """El enrutamiento por intención no debe cambiar silenciosamente."""

    def test_valid_intents_unchanged(self):
        """Lista de intents válidos no debe cambiar sin revisión."""
        from src.graph import route_by_intent

        expected_routes = {
            "faq":       "faq_node",
            "auth":      "auth_node",
            "pagos":     "pagos_node",
            "matricula": "pagos_node",
            "soporte":   "soporte_node",
        }

        for intent, expected_node in expected_routes.items():
            state = {"intent": intent}
            assert route_by_intent(state) == expected_node, \
                f"Ruta para '{intent}' cambió de '{expected_node}'"

    def test_unknown_intent_defaults_to_soporte(self):
        """Intents desconocidos siempre van a soporte."""
        from src.graph import route_by_intent

        for unknown in ["certificado", "voz", "asistencia", "", "xyz"]:
            state = {"intent": unknown}
            assert route_by_intent(state) == "soporte_node"
