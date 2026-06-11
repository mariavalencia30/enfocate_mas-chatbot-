import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage

from src.graph import build_graph
from src.memory.state import ChatState


#Fixtures 

@pytest.fixture
def usuarios_csv(tmp_path):
    df = pd.DataFrame([{
        "user_id": 1,
        "username": "test.user",
        "password_hash": "$2b$dummy",
        "activo": True,
    }])
    p = tmp_path / "usuarios.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def pagos_csv(tmp_path):
    df = pd.DataFrame([{
        "user_id": 1,
        "concepto": "Pensión Mayo",
        "monto": 350000,
        "fecha_vencimiento": "2025-05-15",
        "estado": "pendiente",
    }])
    p = tmp_path / "pagos.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def matriculas_csv(tmp_path):
    df = pd.DataFrame([{
        "user_id": 1,
        "periodo": "2025-1",
        "grado": "Noveno",
        "estado_matricula": "Matriculado",
        "fecha_limite": "2025-01-31",
    }])
    p = tmp_path / "matriculas.csv"
    df.to_csv(p, index=False)
    return str(p)


def make_initial_state(message: str, session_id: str = "integ_001") -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "user_id": "",
        "session_id": session_id,
        "channel": "web",
        "intent": "",
        "auth_status": "pending",
        "auth_attempts": 0,
        "context": {},
        "response_text": "",
        "escalate": False,
        "logs": [],
    }


#Tests de integración 

class TestConversationalFlow:

    @patch("src.nodes.classifier.ChatOpenAI")
    def test_faq_intent_routes_to_faq_node(self, mock_llm_class):
        """Mensajes de FAQ deben llegar al nodo FAQ y obtener respuesta."""

        # Clasificador retorna "faq"
        mock_clf = MagicMock()
        mock_clf.invoke.return_value = MagicMock(content="faq")
        mock_llm_class.return_value = mock_clf

        with patch("src.nodes.faq.ChatOpenAI") as mock_faq_llm, \
             patch("src.nodes.faq.Chroma") as mock_chroma, \
             patch("src.nodes.faq.OpenAIEmbeddings"):

            mock_doc = MagicMock()
            mock_doc.page_content = "El horario es de 7am a 5pm."
            mock_chroma.return_value.similarity_search.return_value = [mock_doc]

            mock_faq = MagicMock()
            mock_faq.invoke.return_value = AIMessage(content="El horario es lunes a viernes 7am-5pm.")
            mock_faq_llm.return_value = mock_faq

            graph = build_graph()
            state = make_initial_state("¿Cuál es el horario de atención?")
            config = {"configurable": {"thread_id": "integ_faq_001"}}
            result = graph.invoke(state, config=config)

        assert result["response_text"] != ""
        assert result["intent"] == "faq"

    @patch("src.nodes.classifier.ChatOpenAI")
    def test_auth_flow_success(self, mock_llm_class):
        """Login con teléfono válido debe resultar en auth_status ok."""

        mock_clf = MagicMock()
        mock_clf.invoke.return_value = MagicMock(content="auth")
        mock_llm_class.return_value = mock_clf

        graph = build_graph()
        state = make_initial_state(
            "+57 3871140239",
            session_id="integ_auth_001"
        )
        config = {"configurable": {"thread_id": "integ_auth_001"}}
        result = graph.invoke(state, config=config)

        assert result["auth_status"] == "ok"
        assert result["response_text"] != ""

    @patch("src.nodes.classifier.ChatOpenAI")
    def test_auth_flow_failure(self, mock_llm_class):
        """Login con teléfono no válido debe incrementar intentos."""

        mock_clf = MagicMock()
        mock_clf.invoke.return_value = MagicMock(content="auth")
        mock_llm_class.return_value = mock_clf

        graph = build_graph()
        state = make_initial_state(
            "+57 9999999999",
            session_id="integ_auth_fail_001"
        )
        config = {"configurable": {"thread_id": "integ_auth_fail_001"}}
        result = graph.invoke(state, config=config)

        assert result["auth_status"] == "fail"
        assert result.get("auth_attempts", 0) >= 1

    @patch("src.nodes.classifier.ChatOpenAI")
    def test_pagos_requires_auth(self, mock_llm_class):
        """Consulta de pagos sin login debe redirigir a autenticación."""

        mock_clf = MagicMock()
        mock_clf.invoke.return_value = MagicMock(content="pagos")
        mock_llm_class.return_value = mock_clf

        graph = build_graph()
        state = make_initial_state("¿cuánto debo?", session_id="integ_pagos_noauth")
        state["auth_status"] = "pending"
        config = {"configurable": {"thread_id": "integ_pagos_noauth"}}
        result = graph.invoke(state, config=config)

        assert "verificar" in result["response_text"].lower() or \
               "teléfono" in result["response_text"].lower()

    @patch("src.nodes.classifier.ChatOpenAI")
    def test_soporte_escalation(self, mock_llm_class):
        """Mensajes de queja deben escalar a soporte humano."""

        mock_clf = MagicMock()
        mock_clf.invoke.return_value = MagicMock(content="soporte")
        mock_llm_class.return_value = mock_clf

        graph = build_graph()
        state = make_initial_state("Tengo una queja urgente", session_id="integ_soporte_001")
        config = {"configurable": {"thread_id": "integ_soporte_001"}}
        result = graph.invoke(state, config=config)

        assert result["escalate"] is True

    @patch("src.nodes.classifier.ChatOpenAI")
    def test_logger_records_interaction(self, mock_llm_class):
        """El nodo logger debe registrar la interacción en el estado."""

        mock_clf = MagicMock()
        mock_clf.invoke.return_value = MagicMock(content="soporte")
        mock_llm_class.return_value = mock_clf

        with patch("src.nodes.logger.open", create=True) as mock_open:
            mock_open.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_open.return_value.__exit__ = MagicMock(return_value=False)

            graph = build_graph()
            state = make_initial_state("hola", session_id="integ_log_001")
            config = {"configurable": {"thread_id": "integ_log_001"}}
            result = graph.invoke(state, config=config)

        assert len(result.get("logs", [])) >= 1
        log_entry = result["logs"][-1]
        assert "timestamp" in log_entry
        assert "intent" in log_entry

    @patch("src.nodes.classifier.ChatOpenAI")
    def test_response_text_populated(self, mock_llm_class):
        """El campo response_text debe siempre estar poblado al final."""

        mock_clf = MagicMock()
        mock_clf.invoke.return_value = MagicMock(content="soporte")
        mock_llm_class.return_value = mock_clf

        graph = build_graph()
        state = make_initial_state("¿qué puedo hacer aquí?", session_id="integ_resp_001")
        config = {"configurable": {"thread_id": "integ_resp_001"}}
        result = graph.invoke(state, config=config)

        assert isinstance(result["response_text"], str)
        assert len(result["response_text"]) > 0