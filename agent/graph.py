"""
agent/graph.py

Pharmacy kiosk ReAct graph.

Topology (matches the reference pattern):

  START
    │
    ▼
  call_model ──(tool_calls?)──► tools
    ▲                              │
    └──────────────────────────────┘
    │
  __end__ (direct answer or post-tool answer)

The loop continues until the model produces a response with no tool calls.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from agent.state import PharmacyState
from agent.tools import TOOLS
from agent.nodes import call_model, route_model_output


def create_graph(checkpointer: MemorySaver | None = None):
    """
    Build and compile the pharmacy ReAct graph.

    checkpointer: MemorySaver (or SQLiteSaver) to persist conversation history.
                  Each customer session uses a unique thread_id via config.
    """
    builder = StateGraph(PharmacyState)

    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.set_entry_point("call_model")
    builder.add_conditional_edges("call_model", route_model_output)
    builder.add_edge("tools", "call_model")

    return builder.compile(checkpointer=checkpointer)


# Pre-compiled without checkpointer — picked up by LangGraph Studio
graph = create_graph()
