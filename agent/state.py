"""
agent/state.py

Pharmacy kiosk graph state.
Just messages — the add_messages reducer handles appending.
Cart lives in InventoryManager (singleton), not in graph state.
"""

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class PharmacyState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
