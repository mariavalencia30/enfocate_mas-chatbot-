from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.nodes.classifier import intent_classifier_node
from src.nodes.faq import faq_node
from src.nodes.auth import auth_node
from src.nodes.pagos import pagos_node
from src.nodes.soporte import soporte_node
from src.nodes.response_builder import response_builder_node
from src.nodes.logger import logger_node
from src.memory.state import ChatState


def route_by_intent(
    state: ChatState,
) -> Literal["faq_node", "auth_node", "pagos_node", "soporte_node"]:
    """Arista condicional: decide a qué nodo enrutar."""
    intent = state.get("intent", "soporte")
    recovery_step = state.get("context", {}).get("recovery", {}).get("step", 0)

    if recovery_step > 0:
        return "auth_node"

    routes = {
        "faq": "faq_node",
        "auth": "auth_node",
        "pagos": "pagos_node",
        "matricula": "pagos_node",
        "soporte": "soporte_node",
    }
    return routes.get(intent, "soporte_node")


def build_graph() -> StateGraph:
    """Construye y compila el StateGraph."""

    memory = MemorySaver()
    builder = StateGraph(ChatState)

    builder.add_node("intent_classifier", intent_classifier_node)
    builder.add_node("faq_node", faq_node)
    builder.add_node("auth_node", auth_node)
    builder.add_node("pagos_node", pagos_node)
    builder.add_node("soporte_node", soporte_node)
    builder.add_node("response_builder", response_builder_node)
    builder.add_node("logger_node", logger_node)

    builder.set_entry_point("intent_classifier")

    builder.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {
            "faq_node": "faq_node",
            "auth_node": "auth_node",
            "pagos_node": "pagos_node",
            "soporte_node": "soporte_node",
        },
    )

    for node in ["faq_node", "auth_node", "pagos_node", "soporte_node"]:
        builder.add_edge(node, "response_builder")

    builder.add_edge("response_builder", "logger_node")
    builder.add_edge("logger_node", END)

    return builder.compile(checkpointer=memory)


graph = build_graph()
