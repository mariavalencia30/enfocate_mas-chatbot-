import pytest
import os
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

from src.nodes.pagos import pagos_node, _get_pagos, _get_asistencia


ESTUDIANTES_CSV = os.path.join(os.path.dirname(__file__), "../../data/csv/datos_estudiantes.csv")


@pytest.fixture
def auth_state():
    return {
        "messages": [HumanMessage(content="¿cuánto debo de pensión?")],
        "user_id": "+573871140239",
        "session_id": "s_pagos_001",
        "channel": "web",
        "intent": "pagos",
        "auth_status": "ok",
        "auth_attempts": 0,
        "context": {},
        "response_text": "",
        "escalate": False,
        "logs": [],
    }


class TestPagosNode:

    def test_requires_auth(self):
        """Sin autenticación, debe pedir login."""
        state = {
            "messages": [HumanMessage(content="¿cuánto debo?")],
            "user_id": "",
            "session_id": "test",
            "channel": "web",
            "intent": "pagos",
            "auth_status": "pending",
            "auth_attempts": 0,
            "context": {},
            "response_text": "",
            "escalate": False,
            "logs": [],
        }
        result = pagos_node(state)
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert any("verificar" in m.content.lower() or "teléfono" in m.content.lower() for m in ai_msgs)

    def test_pagos_query_returns_data(self, auth_state):
        """Con auth OK y teléfono válido, debe devolver datos de pagos."""
        result = pagos_node(auth_state)
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        content = " ".join(m.content for m in ai_msgs)
        assert "pensión" in content.lower() or "Estado de Cuenta" in content

    def test_asistencia_query_returns_data(self, auth_state):
        """Consulta de asistencia debe devolver datos."""
        auth_state["messages"] = [HumanMessage(content="¿asistí a clases?")]
        result = pagos_node(auth_state)
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        content = " ".join(m.content for m in ai_msgs)
        assert "clases" in content.lower() or "Asistencia" in content

    def test_context_updated(self, auth_state):
        """El contexto debe actualizarse con el tipo de consulta."""
        result = pagos_node(auth_state)
        assert "context" in result
        assert result["context"].get("query_type") == "pagos/asistencia"


class TestGetPagos:

    def test_get_pagos_returns_format(self):
        """Debe formatear correctamente los pagos."""
        df = pd.read_csv(ESTUDIANTES_CSV)
        result = _get_pagos("+573871140239", df)
        assert "Pensión" in result or "pagado" in result.lower()

    def test_get_pagos_not_found(self):
        """Teléfono no existente debe devolver mensaje apropiado."""
        df = pd.read_csv(ESTUDIANTES_CSV)
        result = _get_pagos("+579999999999", df)
        assert "No se encontraron" in result


class TestGetAsistencia:

    def test_get_asistencia_returns_format(self):
        """Debe formatear correctamente la asistencia."""
        df = pd.read_csv(ESTUDIANTES_CSV)
        result = _get_asistencia("+573871140239", df)
        assert "clases" in result.lower()

    def test_get_asistencia_not_found(self):
        """Teléfono no existente debe devolver mensaje apropiado."""
        df = pd.read_csv(ESTUDIANTES_CSV)
        result = _get_asistencia("+579999999999", df)
        assert "No se encontraron" in result