"""
tests/unit/test_auth_node.py
=============================
Pruebas unitarias para el módulo de autenticación.
Valida seguridad, lógica de intentos y no exposición de datos sensibles.
"""

import pytest
import os
import pandas as pd
from unittest.mock import patch
from langchain_core.messages import HumanMessage, AIMessage

from src.nodes.auth import auth_node, _extract_credentials, _extract_phone, _normalize_phone


#Fixtures 

@pytest.fixture
def base_state():
    return {
        "messages": [HumanMessage(content="+57 3871140239")],
        "user_id": "",
        "session_id": "session_auth_001",
        "channel": "web",
        "intent": "auth",
        "auth_status": "pending",
        "auth_attempts": 0,
        "context": {},
        "response_text": "",
        "escalate": False,
        "logs": [],
    }


#  Tests de extracción de credenciales 

class TestExtractCredentials:

    def test_extract_format_correo_contrasena(self):
        user, pwd = _extract_credentials("correo: juan@mail.com contraseña: 1234")
        assert user == "juan@mail.com"
        assert pwd == "1234"

    def test_extract_format_email_clave(self):
        user, pwd = _extract_credentials("email: maria@test.com clave: secreto")
        assert user == "maria@test.com"
        assert pwd == "secreto"

    def test_extract_no_credentials_fails(self):
        user, pwd = _extract_credentials("hola quiero entrar")
        assert user == "" or pwd == ""

    def test_extract_single_word_fails(self):
        user, pwd = _extract_credentials("juan")
        assert user == "" or pwd == ""


#  Tests de normalización de teléfono 

class TestPhoneNormalization:

    def test_extract_phone_10_digits(self):
        result = _extract_phone("3001234567")
        assert result == "+573001234567"

    def test_extract_phone_with_country_code(self):
        result = _extract_phone("+573001234567")
        assert result == "+573001234567"

    def test_extract_phone_with_spaces(self):
        result = _extract_phone("+57 300 123 4567")
        assert result == "+573001234567"

    def test_extract_phone_with_11_digits(self):
        result = _extract_phone("573001234567")
        assert result == "+573001234567"

    def test_normalize_phone_10_digits(self):
        result = _normalize_phone("3001234567")
        assert result == "+573001234567"

    def test_normalize_phone_with_57(self):
        result = _normalize_phone("573001234567")
        assert result == "+573001234567"


#  Tests del nodo completo 

class TestAuthNode:

    def test_already_authenticated(self, base_state):
        """Si ya está autenticado, debe confirmarlo sin re-validar."""
        base_state["auth_status"] = "ok"
        result = auth_node(base_state)
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) >= 1
        assert "sesión" in ai_msgs[-1].content.lower() or "ayudar" in ai_msgs[-1].content.lower()

    def test_locked_account_blocks(self, base_state):
        """Cuenta bloqueada no debe intentar validar."""
        base_state["auth_status"] = "locked"
        base_state["auth_attempts"] = 3
        result = auth_node(base_state)
        assert result.get("auth_status") == "locked"
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert any("bloqueado" in m.content.lower() for m in ai_msgs)

    def test_successful_phone_login(self, base_state):
        """Login con teléfono válido debe establecer auth_status ok."""
        result = auth_node(base_state)
        assert result.get("auth_status") == "ok"
        assert result.get("auth_attempts") == 0
        assert result.get("user_id") == "+573871140239"

    def test_failed_phone_login_increments_attempts(self, base_state):
        """Login con teléfono no registrado debe incrementar intentos."""
        base_state["messages"] = [HumanMessage(content="+57 9999999999")]
        result = auth_node(base_state)
        assert result.get("auth_attempts", 0) == 1
        assert result.get("auth_status") == "fail"

    def test_no_sensitive_data_in_response(self, base_state):
        """La respuesta NUNCA debe contener datos sensibles del CSV."""
        result = auth_node(base_state)
        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage):
                assert "$2b$" not in msg.content, "Hash expuesto en respuesta"
                assert "password_hash" not in msg.content.lower()

    def test_no_phone_prompts_for_phone(self, base_state):
        """Si no se detecta teléfono, debe solicitar el número."""
        base_state["messages"] = [HumanMessage(content="quiero entrar")]
        result = auth_node(base_state)
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) >= 1
        assert any("teléfono" in m.content.lower() for m in ai_msgs)