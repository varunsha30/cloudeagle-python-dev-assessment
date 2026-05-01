"""
Assembles the three nodes into a compiled LangGraph StateGraph

Graph structure:
  START -> intent_node -> fetch_node -> synthesize_node -> END

Error handling lives inside each node via state fields
Downstream nodes check these and short-circuit
"""

from __future__ import annotations
from functools import lru_cache
from langgraph.graph import StateGraph, START, END
from app.agent.state import AgentState
from app.agent.nodes import intent_node, fetch_node, synthesize_node


def build_graph():
    builder = StateGraph(AgentState)

    # register the three nodes
    builder.add_node("intent_node", intent_node)
    builder.add_node("fetch_node", fetch_node)
    builder.add_node("synthesize_node", synthesize_node)

    # wire edges, simple linear pipeline
    builder.add_edge(START, "intent_node")
    # builder.add_edge("intent_node", "fetch_node")
    # builder.add_edge("fetch_node", "synthesize_node")
    builder.add_edge("synthesize_node", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_graph():
    """Return the singleton compiled graph"""
    return build_graph()