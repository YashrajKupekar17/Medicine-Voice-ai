"""
agent/tools.py

LangChain @tool definitions for the pharmacy kiosk ReAct agent.
All data comes from a single SQLite DB (rag/data/inventory.db):
  - inventory table  — stock, brands, prices
  - drug_info table  — uses, dosage, side effects (loaded from medicines.csv)
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from rag.inventory import InventoryManager

# ── Singleton ──────────────────────────────────────────────────────────────

_inventory: InventoryManager | None = None


def get_inventory() -> InventoryManager:
    global _inventory
    if _inventory is None:
        _inventory = InventoryManager()
    return _inventory


# ── Tool definitions ───────────────────────────────────────────────────────

@tool
def check_inventory(medicine_name: str) -> str:
    """
    Check if a medicine is available in stock.
    Returns available brands, prices, quantities, and schedule info.
    Call this whenever the customer asks if we have something.
    """
    return get_inventory().check_stock(medicine_name)


@tool
def drug_lookup(query: str) -> str:
    """
    Look up a medicine by symptom, use-case, or name.
    Returns dosage, uses, side effects, and available brands in stock.
    Call this for symptom questions or requests for drug information.
    """
    return get_inventory().drug_lookup(query)


@tool
def check_interaction(drug_a: str, drug_b: str) -> str:
    """Check for interactions between two drugs."""
    info_a = get_inventory().drug_lookup(drug_a)
    info_b = get_inventory().drug_lookup(drug_b)
    return f"{info_a}\n\n---\n\n{info_b}"


@tool
def find_generic(brand_or_query: str) -> str:
    """Find an affordable generic alternative to a brand-name medicine."""
    return get_inventory().drug_lookup(brand_or_query)


@tool
def add_to_cart(brand_name: str, quantity: Optional[int] = None) -> str:
    """
    Add a specific medicine brand to the customer's cart.
    Call this when the customer confirms they want to buy something.
    """
    return get_inventory().add_to_cart(brand_name, quantity if quantity is not None else 1)


@tool
def view_cart() -> str:
    """Show all items currently in the cart with a running total."""
    return get_inventory().view_cart()


@tool
def generate_receipt() -> str:
    """
    Generate the final receipt and close the session.
    Call this when the customer says: done / bill / checkout / receipt / that's all.
    """
    return get_inventory().generate_receipt()


# ── Tool registry — passed to bind_tools and ToolNode ─────────────────────

TOOLS = [
    check_inventory,
    drug_lookup,
    check_interaction,
    find_generic,
    add_to_cart,
    view_cart,
    generate_receipt,
]
