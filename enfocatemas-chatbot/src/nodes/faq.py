import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from src.memory.state import ChatState

# Directorio donde se almacena el índice vectorial
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "../../data/embeddings")

FAQ_SYSTEM_PROMPT = """
Eres el asistente virtual de la institución educativa Enfócate Más.
Responde en español, de forma clara, breve y amigable.
Analiza el contexto proporcionado y responde basándote en la información disponible.
Si la información del contexto está relacionada con la pregunta aunque no sea exacta,
proporciona una respuesta basada en esa información relacionada.
Solo di que no tienes información si el contexto NO tiene NADA que ver con la pregunta.
Incluye datos de contacto cuando sean relevantes: +57 302 8276259, www.enfocatemas.org
""".strip()


def faq_node(state: ChatState) -> dict:
    """
    Nodo LangGraph: responde preguntas frecuentes usando RAG.
    1. Recupera documentos relevantes del vector store.
    2. Genera respuesta contextualizada con el LLM.
    """

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )

    if not last_human:
        return {
            "context": {},
            "messages": state["messages"]
            + [AIMessage(content="No recibí tu pregunta. ¿Puedes repetirla?")],
        }

    query = last_human.content

    #  Búsqueda vectorial 
    try:
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        vector_store = Chroma(
            persist_directory=VECTOR_STORE_DIR,
            embedding_function=embeddings,
            collection_name="faq_docs",
        )
        docs = vector_store.similarity_search(query, k=5)
        context_text = "\n\n---\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        context_text = ""
        print(f"[FAQ NODE] Error en búsqueda vectorial: {e}")

    #  Generación de respuesta 
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt_with_context = (
        f"Contexto institucional:\n{context_text}\n\nPregunta del usuario: {query}"
    )

    response = llm.invoke(
        [
            SystemMessage(content=FAQ_SYSTEM_PROMPT),
            HumanMessage(content=prompt_with_context),
        ]
    )

    return {
        "context": {"faq_docs_found": len(docs) if context_text else 0},
        "messages": state["messages"] + [response],
    }
