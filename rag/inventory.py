"""
rag/inventory.py

InventoryManager — handles all database operations and cart state.

Two responsibilities:
  1. READ  — check what's in stock (by medicine name or brand)
  2. WRITE — cart management (add, view, clear) + receipt generation

The cart lives in memory (not persisted to DB) — it's session state.
A new customer session calls clear_cart() to start fresh.

Why SQLite and not FAISS for inventory?
  FAISS is great for semantic similarity search over unstructured text.
  Inventory queries are exact: "do you have Dolo 650?" needs a precise
  stock count, not a similarity score. SQL is the right tool here.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "inventory.db"


class InventoryManager:
    def __init__(self, db_path: Path = DB_PATH):
        if not db_path.exists():
            raise FileNotFoundError(
                f"Inventory DB not found at {db_path}. "
                "Run: python rag/setup_inventory.py"
            )
        self._db_path = str(db_path)
        self.cart: list[dict] = []
        print(f"[Inventory] Connected to {db_path.name}")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Stock queries ──────────────────────────────────────────────────────

    def check_stock(self, medicine_name: str) -> str:
        """
        Search inventory by generic medicine name or brand name.
        Returns formatted string listing available brands, prices, stock.
        Promoted brands appear first.
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT brand_name, medicine_name, strength, form,
                       quantity, unit, price, schedule,
                       is_promoted, promoted_label
                FROM inventory
                WHERE LOWER(medicine_name) LIKE LOWER(?)
                   OR LOWER(brand_name)    LIKE LOWER(?)
                ORDER BY is_promoted DESC, quantity DESC
            """, (f"%{medicine_name}%", f"%{medicine_name}%")).fetchall()

        if not rows:
            return f"NOT_FOUND: No medicine matching '{medicine_name}' in our inventory."

        in_stock  = [r for r in rows if r["quantity"] > 0]
        out_stock = [r for r in rows if r["quantity"] == 0]

        lines = []

        if in_stock:
            lines.append(f"AVAILABLE ({len(in_stock)} option(s)):")
            for r in in_stock:
                label = f" [{r['promoted_label']}]" if r["is_promoted"] else ""
                sched = f" [Schedule {r['schedule']} - Prescription required]" if r["schedule"] == "H" else ""
                lines.append(
                    f"  • {r['brand_name']} {r['strength']} — "
                    f"₹{r['price']:.0f}/{r['unit']}"
                    f"{label}{sched} "
                    f"({r['quantity']} {r['unit']}s in stock)"
                )
        else:
            lines.append(f"OUT OF STOCK: No {medicine_name} available right now.")

        if out_stock:
            lines.append("Currently out of stock:")
            for r in out_stock:
                lines.append(f"  • {r['brand_name']} {r['strength']} (unavailable)")

        return "\n".join(lines)

    # Words the LLM appends to brand names that are not in the DB
    _FORM_WORDS = {
        "tablet", "tablets", "capsule", "capsules", "syrup", "sachet",
        "sachets", "strip", "strips", "bottle", "bottles", "injection",
        "cream", "gel", "drops", "suspension", "powder",
    }

    def get_brand_info(self, brand_name: str) -> dict | None:
        """
        Return single brand row as dict, or None if not found.
        Tries progressively cleaned search queries:
          1. Raw name (e.g. "Calpol 250 Syrup 250mg/5ml")
          2. Strip strength patterns like "650mg", "250mg/5ml", "10mg" (LLM appends them)
          3. Strip form words like "tablets", "capsules", "syrup"
        """
        import re

        def _search(query: str):
            with self._conn() as conn:
                return conn.execute(
                    "SELECT * FROM inventory WHERE LOWER(brand_name) LIKE LOWER(?)",
                    (f"%{query}%",),
                ).fetchone()

        # Pass 1: exact
        row = _search(brand_name)
        if row:
            return dict(row)

        # Pass 2: strip strength patterns (e.g. "250mg/5ml", "650mg", "10 mg")
        cleaned = re.sub(r"\b\d+\s*mg(?:/\d+\s*ml)?\b", "", brand_name, flags=re.IGNORECASE).strip()
        if cleaned and cleaned.lower() != brand_name.lower():
            row = _search(cleaned)
            if row:
                return dict(row)

        # Pass 3: strip form words (e.g. "tablets", "capsules", "syrup")
        words = [w for w in cleaned.lower().split() if w not in self._FORM_WORDS]
        short = " ".join(words).strip()
        if short and short != cleaned.lower():
            row = _search(short)
            if row:
                return dict(row)

        return None

    # ── Drug info lookup (replaces FAISS) ─────────────────────────────────

    _STOPWORDS = {
        "do", "you", "have", "for", "what", "anything", "any", "is", "the",
        "a", "an", "me", "i", "can", "get", "give", "about", "how", "take",
        "use", "uses", "need", "want", "help", "with", "good", "something",
        "please", "and", "or", "to", "of", "in", "my", "some", "tell",
        "ache",   # too generic — matches "headache/toothache" for any pain query
        "high",   # too generic — matches "high blood pressure" for any query with "high"
        "low",    # similarly generic
    }

    def drug_lookup(self, query: str) -> str:
        """
        Search drug_info by symptom/use-case keyword.
        Returns the best matching drug with dosage info and in-stock brands.
        Prefers medicines that are currently in stock.
        """
        words = [w.strip("?.!,") for w in query.lower().split()]
        keywords = [w for w in words if w not in self._STOPWORDS and len(w) > 2]

        if not keywords:
            return "Please describe the symptom or medicine name."

        # Expand keywords with common stem variants
        _ALIASES = {
            "allergy": "allergic", "allergies": "allergic",
            "acidity": "acid", "cold": "rhinitis",
            "pressure": "hypertension", "sugar": "diabetes",
            "pain": "pain", "fever": "fever",
        }
        expanded = []
        for kw in keywords:
            expanded.append(kw)
            if kw in _ALIASES:
                expanded.append(_ALIASES[kw])
        keywords = expanded

        with self._conn() as conn:
            for keyword in keywords:
                rows = conn.execute("""
                    SELECT d.generic_name, d.category, d.uses,
                           d.dosage_adult, d.side_effects,
                           SUM(CASE WHEN i.quantity > 0 THEN 1 ELSE 0 END) AS in_stock_count
                    FROM drug_info d
                    LEFT JOIN inventory i ON LOWER(i.medicine_name) = LOWER(d.generic_name)
                    WHERE LOWER(d.uses)         LIKE ?
                       OR LOWER(d.category)     LIKE ?
                       OR LOWER(d.generic_name) LIKE ?
                    GROUP BY d.generic_name
                    ORDER BY in_stock_count DESC, d.generic_name
                """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%")).fetchall()

                if rows:
                    r = rows[0]
                    lines = [
                        f"Drug: {r['generic_name']} ({r['category']})",
                        f"Uses: {r['uses']}",
                        f"Adult dosage: {r['dosage_adult']}",
                    ]
                    if r["side_effects"]:
                        lines.append(f"Side effects: {r['side_effects']}")

                    # Show available brands from inventory
                    brands = conn.execute("""
                        SELECT brand_name, strength, price, unit, is_promoted, promoted_label
                        FROM inventory
                        WHERE LOWER(medicine_name) = LOWER(?) AND quantity > 0
                        ORDER BY is_promoted DESC
                    """, (r["generic_name"],)).fetchall()

                    if brands:
                        brand_names = ", ".join(
                            f"{b['brand_name']}{' [' + b['promoted_label'] + ']' if b['is_promoted'] else ''}"
                            for b in brands
                        )
                        lines.append(f"Available brands: {brand_names}")
                    else:
                        lines.append("Note: currently out of stock in our pharmacy.")

                    return "\n".join(lines)

        # Fallback: search inventory medicine_name directly
        for keyword in keywords:
            rows = conn.execute("""
                SELECT DISTINCT medicine_name FROM inventory
                WHERE LOWER(medicine_name) LIKE ? AND quantity > 0
            """, (f"%{keyword}%",)).fetchall()
            if rows:
                med = rows[0]["medicine_name"]
                brands = conn.execute("""
                    SELECT brand_name, strength, price, unit FROM inventory
                    WHERE medicine_name = ? AND quantity > 0
                    ORDER BY is_promoted DESC
                """, (med,)).fetchall()
                brand_str = ", ".join(b["brand_name"] for b in brands)
                return f"Drug: {med}\nAvailable brands: {brand_str}"

        return f"No medicine found in our database for: {query}"

    # ── Cart operations ────────────────────────────────────────────────────

    def add_to_cart(self, brand_name: str, quantity: int = 1) -> str:
        """
        Add a brand to the current session cart.
        Validates stock before adding.
        """
        info = self.get_brand_info(brand_name)

        if info is None:
            return f"ERROR: '{brand_name}' not found in inventory."

        if info["quantity"] < quantity:
            if info["quantity"] == 0:
                return f"Sorry, {info['brand_name']} is currently out of stock."
            return (
                f"Only {info['quantity']} {info['unit']}(s) of "
                f"{info['brand_name']} available. "
                f"Cannot add {quantity}."
            )

        # Check if already in cart — update quantity
        for item in self.cart:
            if item["brand_name"].lower() == info["brand_name"].lower():
                item["quantity"] += quantity
                total = item["quantity"] * item["unit_price"]
                return (
                    f"Updated: {info['brand_name']} × {item['quantity']} "
                    f"= ₹{total:.0f}"
                )

        # New item
        self.cart.append({
            "brand_name":  info["brand_name"],
            "medicine":    info["medicine_name"],
            "quantity":    quantity,
            "unit_price":  info["price"],
            "unit":        info["unit"],
            "form":        info["form"],
            "schedule":    info["schedule"],
        })

        subtotal = info["price"] * quantity
        return (
            f"Added: {info['brand_name']} × {quantity} {info['unit']}(s) "
            f"= ₹{subtotal:.0f}\n"
            f"Cart total so far: ₹{self._cart_total():.0f}"
        )

    def view_cart(self) -> str:
        """Return formatted cart summary."""
        if not self.cart:
            return "Cart is empty."

        lines = ["Current cart:"]
        for item in self.cart:
            subtotal = item["quantity"] * item["unit_price"]
            lines.append(
                f"  • {item['brand_name']} × {item['quantity']} "
                f"{item['unit']}(s) — ₹{subtotal:.0f}"
            )
        lines.append(f"Total: ₹{self._cart_total():.0f}")
        return "\n".join(lines)

    def generate_receipt(self, shop_name: str = "MedAssist Pharmacy") -> str:
        """
        Generate the final receipt.
        Returns formatted text receipt.
        """
        if not self.cart:
            return "Cart is empty — nothing to bill."

        now = datetime.now().strftime("%d-%b-%Y  %H:%M")
        total = self._cart_total()
        separator = "=" * 36

        lines = [
            separator,
            f"  {shop_name}",
            f"  {now}",
            separator,
        ]

        for item in self.cart:
            subtotal = item["quantity"] * item["unit_price"]
            lines.append(f"{item['brand_name']} ({item['medicine']})")
            lines.append(
                f"  {item['quantity']} {item['unit']}(s) × "
                f"₹{item['unit_price']:.0f}   ₹{subtotal:.0f}"
            )

        lines.append("-" * 36)
        lines.append(f"TOTAL                       ₹{total:.0f}")
        lines.append(separator)

        # Schedule H warning
        h_items = [i["brand_name"] for i in self.cart if i["schedule"] == "H"]
        if h_items:
            lines.append(f"⚠ Prescription required for: {', '.join(h_items)}")
            lines.append("  Please show prescription at counter.")
            lines.append("-" * 36)

        lines.append("Show this receipt at the counter.")
        lines.append(separator)

        return "\n".join(lines)

    def clear_cart(self) -> None:
        """Reset cart for new customer session."""
        self.cart.clear()

    def _cart_total(self) -> float:
        return sum(i["quantity"] * i["unit_price"] for i in self.cart)
