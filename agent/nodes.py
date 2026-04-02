"""
agent/nodes.py

Two nodes for the pharmacy ReAct agent.

llama3.2:3b supports Ollama's native tool-calling API via bind_tools().
Architecture:
  1. call_model uses bind_tools() — LLM returns AIMessage with tool_calls set
  2. ToolNode sees tool_calls and executes natively
  3. Round 2: ToolMessage detected → format answer directly (no LLM), no hallucination
  4. Keyword overrides (symptom/checkout/purchase) catch cases where LLM skips tool calls
"""

from __future__ import annotations

import json
import uuid
from typing import Literal, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from agent.state import PharmacyState
from agent.tools import TOOLS

OLLAMA_MODEL = "llama3.2:3b"

# ── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a pharmacy kiosk assistant. Use the available tools to help customers.

Rules:
1. Customer asks about a medicine by name → call check_inventory
2. Customer describes a symptom (headache/fever/stomach ache/allergy/cough) → call drug_lookup
3. Customer wants to buy something → call add_to_cart
4. Customer says bill/done/checkout/receipt → call generate_receipt
5. Greeting only → respond directly without a tool call
6. NEVER invent medicine names, prices, or stock — only use tool results."""

# Symptom keywords — force drug_lookup if LLM skips it
SYMPTOM_KEYWORDS = {
    "headache", "fever", "pain", "cold", "cough", "acidity", "acid",
    "allergy", "allergic", "stomach", "nausea", "diarrhea", "vomiting",
    "diabetes", "diabetic", "blood pressure", "hypertension",
    "flu", "runny nose", "sore throat", "inflammation", "infection",
}

# Keywords that should always trigger generate_receipt (even if LLM hallucinates)
CHECKOUT_KEYWORDS = {
    "bill", "receipt", "checkout", "check out", "that's all", "thats all",
    "total", "done", "finish", "pay", "payment",
}

# Purchase-intent phrases → force add_to_cart with extracted medicine name
_PURCHASE_PHRASES = [
    "i'll also have", "i will also have", "also have",
    "i'll have", "i will have",
    "i'll take", "i will take",
    "i'll buy", "i will buy",
    "give me", "get me",
    "i want", "i need",
    "add one", "add two", "add a", "add an",
]

# "do you have" / "is there" patterns → force check_inventory
_HAVE_QUERY_PATTERNS = [
    "do you have", "do you carry", "is there", "is it available",
    "do you stock", "have you got",
]

# Contextual add phrases — "add it", "yes please" referring to last recommendation
_CONTEXTUAL_ADD_PHRASES = [
    "add it", "add that", "add them", "add the", "please add",
    "yes add", "yeah add", "yes please", "yeah please",
    "add to cart", "put it in", "put that in",
]

# Tools that don't need LLM Round 2 — their output IS the answer
_DIRECT_ANSWER_TOOLS = {"check_inventory", "drug_lookup", "add_to_cart",
                        "view_cart", "generate_receipt",
                        "check_interaction", "find_generic"}

# ── LLM singleton ──────────────────────────────────────────────────────────

_llm: ChatOllama | None = None
_llm_with_tools = None


def get_llm():
    """Return ChatOllama instance with tools bound (native tool calling)."""
    global _llm, _llm_with_tools
    if _llm is None:
        try:
            from openai import OpenAI
            client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            available = [m.id for m in client.models.list().data]
            if OLLAMA_MODEL not in available:
                print(f"[LLM] WARNING: '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
            else:
                print(f"[LLM] Connected. Model: {OLLAMA_MODEL}")
        except Exception:
            raise ConnectionError(
                "Cannot connect to Ollama at http://localhost:11434.\n"
                "Start Ollama: open the app or run 'ollama serve'"
            )
        _llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1)
        _llm_with_tools = _llm.bind_tools(TOOLS)
    return _llm_with_tools


# ── Nodes ──────────────────────────────────────────────────────────────────

def call_model(state: PharmacyState) -> Dict[str, List]:
    """
    Round 1: invoke LLM with bind_tools() — native tool_calls in response.
    Round 2: ToolMessage detected → format answer directly (no LLM).
    Keyword overrides: safety net when LLM skips tool calls.
    """
    messages = list(state["messages"])
    last_msg  = messages[-1] if messages else None

    # ── Round 2: tool(s) just ran — collect all ToolMessages from this batch ───
    if isinstance(last_msg, ToolMessage):
        tool_results = _collect_tool_results(messages)
        tool_name    = _last_tool_name(messages)
        if len(tool_results) == 1:
            answer = _format_direct(tool_name, tool_results[0])
        else:
            # Multiple tools ran (e.g. multi-item add) — combine, one trailing prompt
            parts = []
            for r in tool_results:
                formatted = _format_direct("add_to_cart", r)
                parts.append(formatted.replace("\nAnything else?", ""))
            answer = "\n".join(parts) + "\nAnything else?"
        return {"messages": [AIMessage(content=answer)]}

    # ── Round 1: native tool calling ──────────────────────────────────────────
    all_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = get_llm().invoke(all_messages)

    if response.tool_calls:
        # Normalize each tool call's args (fix schema quirks like {"Dolo": "string"})
        clean_calls = []
        for tc in response.tool_calls:
            clean_args = _normalize_args(tc["name"], tc.get("args") or {})
            # Expand list brand_name → multiple individual calls
            for expanded_tc in _expand_list_args(tc["name"], clean_args, tc):
                clean_calls.append(expanded_tc)
        print(f"[LLM] Tool call: {clean_calls[0]['name']}({clean_calls[0]['args']})")
        response = AIMessage(content=response.content, tool_calls=clean_calls)
        return {"messages": [response]}

    # LLM returned no tool call — apply keyword overrides
    lower = _last_human_text(messages).lower()
    print(f"[LLM] No tool call, content: {response.content[:80]!r}")

    # Checkout → generate_receipt
    if any(kw in lower for kw in CHECKOUT_KEYWORDS):
        print(f"[LLM] Forcing generate_receipt")
        return {"messages": [_make_tool_call_message("", {"tool": "generate_receipt", "args": {}})]}

    # Symptom → drug_lookup
    matched = next((kw for kw in SYMPTOM_KEYWORDS if kw in lower), None)
    if matched:
        full_query = _last_human_text(messages)
        print(f"[LLM] Forcing drug_lookup: '{matched}'")
        return {"messages": [_make_tool_call_message("", {"tool": "drug_lookup", "args": {"query": full_query}})]}

    # "Do you have X" → check_inventory
    if any(pat in lower for pat in _HAVE_QUERY_PATTERNS):
        med = _extract_after_have_query(lower)
        if med:
            print(f"[LLM] Forcing check_inventory: '{med}'")
            return {"messages": [_make_tool_call_message("", {"tool": "check_inventory", "args": {"medicine_name": med}})]}

    # Purchase intent → add_to_cart
    med_name, qty = _extract_purchase_target(lower)
    if med_name:
        print(f"[LLM] Forcing add_to_cart: '{med_name}' x{qty}")
        return {"messages": [_make_tool_call_message("", {"tool": "add_to_cart", "args": {"brand_name": med_name, "quantity": qty}})]}

    # Contextual add ("add it", "yes please") → add last recommended brand
    if any(pat in lower for pat in _CONTEXTUAL_ADD_PHRASES):
        brand = _extract_last_recommended_brand(messages)
        if brand:
            print(f"[LLM] Forcing contextual add_to_cart: '{brand}'")
            return {"messages": [_make_tool_call_message("", {"tool": "add_to_cart", "args": {"brand_name": brand, "quantity": 1}})]}

    return {"messages": [response]}


def route_model_output(state: PharmacyState) -> Literal["tools", "__end__"]:
    """Tool calls pending → execute them. Final answer → end."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "__end__"


# ── Helpers ────────────────────────────────────────────────────────────────

_KNOWN_TOOLS = {
    "check_inventory", "drug_lookup", "add_to_cart", "view_cart",
    "generate_receipt", "check_interaction", "find_generic",
}

# Positional parameter names — used to convert list args and function-call notation
_TOOL_PARAMS_LIST: dict[str, list[str]] = {
    "check_inventory":  ["medicine_name"],
    "drug_lookup":      ["query"],
    "add_to_cart":      ["brand_name", "quantity"],
    "view_cart":        [],
    "generate_receipt": [],
    "check_interaction":["drug1", "drug2"],
    "find_generic":     ["brand_name"],
}


def _normalize_args(tool_name: str, args: dict) -> dict:
    """
    Fix args dict when LLM uses wrong key names or wrong value types.

    Cases handled:
      {"Dolo 650": 1}           → brand_name="Dolo 650", quantity=1  (wrong key name)
      {"brand_name": {"Dolo": "string"}} → brand_name="Dolo"         (dict-as-value)
      {"brand_name": "X", "quantity": "1"} → quantity=1              (string int)
    """
    params = _TOOL_PARAMS_LIST.get(tool_name, [])
    if not params:
        return args

    # First pass: fix dict-as-value ({"Dolo": "string"} → "Dolo")
    fixed: dict = {}
    for k, v in args.items():
        if isinstance(v, dict):
            # Use the first key of the dict as the actual value
            keys = list(v.keys())
            fixed[k] = keys[0] if keys else v
        elif isinstance(v, str) and v.lower() in ("none", "null", ""):
            pass  # drop invalid string values, will use defaults
        else:
            fixed[k] = v
    args = fixed

    # Second pass: coerce types
    for param in params:
        if param not in args:
            continue
        val = args[param]
        if param == "quantity":
            try:
                args[param] = int(val)
            except (TypeError, ValueError):
                args[param] = 1

    # Third pass: drop extra keys not in params (e.g. LLM passes {"cart": []} to view_cart)
    args = {k: v for k, v in args.items() if k in params}

    # If still some keys aren't valid params, treat keys+values as positional
    if all(k in params for k in args):
        return args
    all_vals = list(args.keys()) + list(args.values())
    result: dict = {}
    vi = 0
    for param in params:
        if param in args:
            result[param] = args[param]
        elif vi < len(all_vals):
            result[param] = all_vals[vi]
            vi += 1
    return result


def _expand_list_args(tool_name: str, args: dict, original_tc: dict) -> list[dict]:
    """
    If add_to_cart receives brand_name as a list, expand into multiple tool calls.
    e.g. brand_name=['Dolo 650', 'Crocin 500'] → two separate add_to_cart calls.
    For all other cases, returns [original_tc_with_clean_args].
    """
    if tool_name == "add_to_cart":
        brand = args.get("brand_name")
        qty   = args.get("quantity", 1) or 1
        if isinstance(brand, list):
            return [
                {**original_tc,
                 "id":   f"call_{uuid.uuid4().hex[:8]}",
                 "args": {"brand_name": b, "quantity": qty}}
                for b in brand if b
            ]
    return [{**original_tc, "args": args}]


def _parse_action(text: str) -> dict | None:
    """
    Parse phi3:mini tool call output. phi3:mini uses at least 6 different formats:

      F1: Action: {"tool": "check_inventory", "args": {"medicine_name": "X"}}
      F2: Action: add_to_cart {"brand_name": "X", "quantity": 1}
      F3: {"tool": "generate_receipt"}
      F4: Action: generate_receipt
      F5: Action: check_inventory, args: {"medicine_name": "X"}
      F6: ACTION: add_to_cart(Dolo,1)   ← function-call notation (case-insensitive)

    Strategy:
      Pass 1 — scan for any JSON object that contains a "tool" key (F1, F3).
      Pass 2 — regex "Action: tool_name [anything up to {] {json}" (F2, F5).
      Pass 3 — function-call notation "ACTION: tool_name(args)" (F6).
      Pass 4 — bare "Action: tool_name" with no JSON (F4).
    """
    import re

    if "```" in text:
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))

    # Pass 1: any JSON object with a "tool" key
    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            break
        try:
            data, end_pos = json.JSONDecoder().raw_decode(text, start)
            if isinstance(data, dict) and data.get("tool") in _KNOWN_TOOLS:
                raw_args = data.get("args", {})
                # Convert list args ["X", 1] → named dict {"brand_name": "X", "quantity": 1}
                if isinstance(raw_args, list):
                    params = _TOOL_PARAMS_LIST.get(data["tool"], [])
                    raw_args = {params[i]: v for i, v in enumerate(raw_args) if i < len(params)}
                if not isinstance(raw_args, dict):
                    raw_args = {}
                return {"tool": data["tool"], "args": _normalize_args(data["tool"], raw_args)}
            pos = end_pos
        except json.JSONDecodeError:
            pos = start + 1

    # Pass 2: "Action: tool_name [anything except {] {args_json}"
    # [^{]* skips ", args: " or " " or any other separator before the JSON
    m = re.search(r"[Aa][Cc][Tt][Ii][Oo][Nn]:\s*(\w+)[^{(]*(\{.*)", text, re.DOTALL)
    if m:
        tool_name = m.group(1)
        if tool_name in _KNOWN_TOOLS:
            args_text = m.group(2)
            try:
                args, _ = json.JSONDecoder().raw_decode(args_text)
                if isinstance(args, dict):
                    return {"tool": tool_name, "args": args}
            except json.JSONDecodeError:
                pass
            return {"tool": tool_name, "args": {}}

    # Pass 3: function-call notation "ACTION: tool_name(arg1, arg2, ...)"
    m = re.search(r"[Aa][Cc][Tt][Ii][Oo][Nn]:\s*(\w+)\(([^)]*)\)", text)
    if m:
        tool_name = m.group(1)
        if tool_name in _KNOWN_TOOLS:
            raw_args = m.group(2).strip()
            args = _parse_positional_args(tool_name, raw_args)
            return {"tool": tool_name, "args": args}

    # Pass 4: bare "Action: tool_name" with no JSON at all
    m = re.search(r"[Aa][Cc][Tt][Ii][Oo][Nn]:\s*(\w+)", text)
    if m and m.group(1) in _KNOWN_TOOLS:
        return {"tool": m.group(1), "args": {}}

    return None


# Positional parameter names for each tool (for function-call notation parsing)
def _parse_positional_args(tool_name: str, raw_args: str) -> dict:
    """Convert comma-separated positional args to a named-param dict."""
    if not raw_args:
        return {}
    params = _TOOL_PARAMS_LIST.get(tool_name, [])
    parts  = [p.strip() for p in raw_args.split(",")]
    result = {}
    for i, param in enumerate(params):
        if i >= len(parts):
            break
        val = parts[i].strip("'\" ")
        # Try to coerce to int or float
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        result[param] = val
    return result


def _make_tool_call_message(content: str, tool_call: dict) -> AIMessage:
    """
    Wrap a parsed tool call into an AIMessage with tool_calls set,
    so LangGraph's ToolNode can execute it natively.
    """
    return AIMessage(
        content=content,
        tool_calls=[{
            "id":   f"call_{uuid.uuid4().hex[:8]}",
            "name": tool_call["tool"],
            "args": tool_call["args"],
            "type": "tool_call",
        }],
    )


def _extract_answer(text: str) -> str:
    """Pull text after 'Answer:'. Falls back to stripping Thought/Action lines."""
    for line in text.split("\n"):
        if line.strip().lower().startswith("answer:"):
            return line.strip()[len("answer:"):].strip()
    kept = [
        l.strip() for l in text.split("\n")
        if l.strip() and not l.strip().lower().startswith(("thought:", "action:"))
    ]
    return " ".join(kept)


def _last_human_text(messages: list) -> str:
    return next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )


def _last_tool_name(messages: list) -> str:
    """Return the name of the most recently called tool."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            return msg.tool_calls[0]["name"]
    return ""


def _collect_tool_results(messages: list) -> list[str]:
    """Return contents of all ToolMessages at the tail (since last non-ToolMessage)."""
    results = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            results.insert(0, msg.content)
        else:
            break
    return results


def _extract_last_recommended_brand(messages: list) -> str | None:
    """
    Look at the last AI answer and extract the first brand name mentioned.
    Handles patterns like:
      "We have Digene Syrup, Gelusil in stock."
      "AVAILABLE: • Dolo 650 650mg — ₹32..."
    """
    import re
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            content = msg.content
            # "We have X [label], Y in stock"
            m = re.search(r"We have ([A-Za-z0-9][A-Za-z0-9 \-]+?)(?:\s*[\[,]|\s+in stock)", content)
            if m:
                return m.group(1).strip()
            # AVAILABLE bullet "• Brand Name 500mg"
            m = re.search(r"•\s+([A-Za-z][A-Za-z0-9 \-]+?)\s+\d", content)
            if m:
                return m.group(1).strip()
    return None


_QTY_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _extract_purchase_target(lower_text: str) -> tuple[str, int] | tuple[None, None]:
    """
    Extract (medicine_name, quantity) from a purchase-intent utterance.
    Returns (None, None) if no purchase phrase found.
    """
    import re
    for phrase in sorted(_PURCHASE_PHRASES, key=len, reverse=True):
        if phrase in lower_text:
            after = lower_text[lower_text.index(phrase) + len(phrase):].strip()
            # Extract quantity word or digit
            qty = 1
            m = re.match(r"^(\d+)\s+", after)
            if m:
                qty = int(m.group(1))
                after = after[m.end():]
            else:
                m = re.match(r"^(one|two|three|four|five)\s+", after)
                if m:
                    qty = _QTY_WORDS[m.group(1)]
                    after = after[m.end():]
                else:
                    # Skip "a"/"an"
                    after = re.sub(r"^(?:a|an)\s+", "", after)
            # Strip trailing filler
            after = re.sub(r"\s+(?:please|for me|now|to cart)\.?$", "", after)
            name = after.strip().rstrip(".,!")
            if name and len(name) > 1:
                return name, qty
    return None, None


def _extract_after_have_query(lower_text: str) -> str | None:
    """Extract medicine name from 'do you have X' / 'is there X' queries."""
    import re
    for pat in sorted(_HAVE_QUERY_PATTERNS, key=len, reverse=True):
        if pat in lower_text:
            after = lower_text[lower_text.index(pat) + len(pat):].strip()
            after = re.sub(r"^(?:any|a|an|some)\s+", "", after)
            after = re.sub(r"\s+(?:in stock|available|please|\?|now).*$", "", after)
            name = after.strip().rstrip(".,!?")
            if name and len(name) > 1:
                return name
    return None


def _format_direct(tool_name: str, observation: str) -> str:
    """Format tool output without LLM involvement."""
    if tool_name == "check_inventory":
        if observation.startswith("NOT_FOUND"):
            return observation.replace("NOT_FOUND: ", "Sorry, ") + " Try a different name."
        return observation

    if tool_name == "drug_lookup":
        # Extract drug name and available brands, format as a concise suggestion
        lines = observation.split("\n")
        drug_name = lines[0].replace("Drug: ", "").split("(")[0].strip() if lines else "the medicine"
        brands_line = next((l for l in lines if l.startswith("Available brands:")), None)
        if brands_line:
            brands = brands_line.replace("Available brands: ", "")
            return f"For that I recommend {drug_name}. We have {brands} in stock. Would you like to add one to your cart?"
        if "out of stock" in observation.lower():
            return f"For that I recommend {drug_name}, but it's currently out of stock."
        return f"For that I recommend {drug_name}. Would you like me to check if we have it in stock?"

    if tool_name == "add_to_cart":
        if observation.startswith("ERROR:"):
            # Make error message natural instead of reading raw "ERROR: '...' not found"
            import re
            m = re.search(r"'(.+?)' not found", observation)
            name = m.group(1) if m else "that medicine"
            return f"Sorry, I couldn't find {name} in our inventory. Could you check the name?"
        if observation.lower().startswith("error invoking tool"):
            # ToolNode validation error (e.g. Pydantic) — hide technical details
            import re
            m = re.search(r"brand_name.*?'([^']+)'", observation)
            if not m:
                m = re.search(r"kwargs\s*\{[^}]*'brand_name':\s*'([^']+)'", observation)
            name = m.group(1) if m else "that medicine"
            return f"Sorry, I couldn't add {name} to your cart. Could you say the exact brand name?"
        return observation + "\nAnything else?"

    # generate_receipt, view_cart, check_interaction, find_generic
    return observation
