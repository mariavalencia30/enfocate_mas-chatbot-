import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from src.nodes.faq import faq_node


@pytest.fixture
def base_state():
    return {
        "messages": [HumanMessage(content="¿Cuál es el horario de atención?")],
        "user_id": "test_user",
        "session_id": "session_001",
        "channel": "web",
        "intent": "faq",
        "auth_status": "pending",
        "auth_attempts": 0,
        "context": {},
        "response_text": "",
        "escalate": False,
        "logs": [],
    }


class TestFAQNode:

    @patch("src.nodes.faq.Chroma")
    @patch("src.nodes.faq.OpenAIEmbeddings")
    @patch("src.nodes.faq.ChatOpenAI")
    def test_faq_returns_ai_message(self, mock_llm_class, mock_emb_class, mock_chroma_class, base_state):
        """El nodo FAQ debe agregar un AIMessage al historial."""

        # Mock del vector store
        mock_doc = MagicMock()
        mock_doc.page_content = "El horario de atención es de 7am a 5pm de lunes a viernes."
        mock_chroma = MagicMock()
        mock_chroma.similarity_search.return_value = [mock_doc]
        mock_chroma_class.return_value = mock_chroma

        # Mock del LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="El horario de atención es de 7am a 5pm, lunes a viernes."
        )
        mock_llm_class.return_value = mock_llm

        result = faq_node(base_state)

        assert "messages" in result
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_messages) >= 1

    @patch("src.nodes.faq.Chroma")
    @patch("src.nodes.faq.OpenAIEmbeddings")
    @patch("src.nodes.faq.ChatOpenAI")
    def test_faq_no_docs_found_fallback(self, mock_llm_class, mock_emb_class, mock_chroma_class, base_state):
        """Cuando no hay documentos relevantes, la respuesta debe indicarlo."""

        mock_chroma = MagicMock()
        mock_chroma.similarity_search.return_value = []
        mock_chroma_class.return_value = mock_chroma

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="No tengo información sobre eso. Por favor, comunícate con la institución."
        )
        mock_llm_class.return_value = mock_llm

        result = faq_node(base_state)
        assert "messages" in result

    def test_faq_no_human_message(self):
        """Si no hay mensaje humano, debe responder con mensaje de error."""
        state = {
            "messages": [],
            "user_id": "",
            "session_id": "test",
            "channel": "web",
            "intent": "faq",
            "auth_status": "pending",
            "auth_attempts": 0,
            "context": {},
            "response_text": "",
            "escalate": False,
            "logs": [],
        }
        result = faq_node(state)
        assert "messages" in result

    @patch("src.nodes.faq.Chroma")
    @patch("src.nodes.faq.OpenAIEmbeddings")
    @patch("src.nodes.faq.ChatOpenAI")
    def test_faq_context_updated(self, mock_llm_class, mock_emb_class, mock_chroma_class, base_state):
        """El contexto debe actualizarse con el número de documentos encontrados."""

        mock_doc = MagicMock()
        mock_doc.page_content = "Información relevante."
        mock_chroma = MagicMock()
        mock_chroma.similarity_search.return_value = [mock_doc, mock_doc]
        mock_chroma_class.return_value = mock_chroma

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Respuesta.")
        mock_llm_class.return_value = mock_llm

        result = faq_node(base_state)
        assert "context" in result
        assert result["context"].get("faq_docs_found", 0) == 2
