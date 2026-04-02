"""
agent/kiosk.py

PharmacyKiosk — the entry point for the pharmacy assistant.

Replaces agent/orchestrator.py. The graph handles all routing logic;
this class is just a thin shell that:
  1. Feeds text/audio into the graph
  2. Extracts the answer from state
  3. Translates + speaks it (voice mode)
  4. Manages session lifecycle (reset)

Session memory:
  LangGraph MemorySaver persists conversation history across turns.
  Each customer session gets a unique thread_id. Calling reset() creates
  a new thread_id, which starts a fresh history while keeping the same
  graph instance.
"""

from __future__ import annotations

import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from .graph import create_graph


class PharmacyKiosk:
    def __init__(self, text_only: bool = False):
        print("[Kiosk] Initializing PharmacyKiosk...")

        checkpointer = MemorySaver()
        self.graph   = create_graph(checkpointer=checkpointer)
        self._session_id = self._new_session_id()

        if not text_only:
            from audio.stt import WhisperSTT
            from audio.tts import PiperTTS
            self.stt = WhisperSTT()
            self.tts = PiperTTS()
        else:
            self.stt = None
            self.tts = None

        self._text_only = text_only
        print("[Kiosk] Ready.\n")

    # ── Core: text in, text out ────────────────────────────────────────────

    def process_text(self, transcript: str, language: str = "en") -> str:
        transcript = transcript.strip()
        if not transcript:
            return ""

        config = {"configurable": {"thread_id": self._session_id}}
        state = self.graph.invoke(
            {"messages": [HumanMessage(content=transcript)]},
            config=config,
        )
        # Last AIMessage without tool_calls is the final answer
        last_ai = next(
            (m for m in reversed(state["messages"])
             if isinstance(m, AIMessage) and not m.tool_calls),
            None,
        )
        return (last_ai.content if last_ai else "").strip()

    # ── Voice: used by pipeline.py ─────────────────────────────────────────

    def process_audio(self, audio_array) -> tuple[str, str]:
        if self.stt is None:
            raise RuntimeError("STT not loaded. Initialize with text_only=False.")

        transcript, language = self.stt.transcribe(audio_array)
        if not transcript.strip():
            return "", language

        from audio.translator import translate
        english_answer = self.process_text(transcript, language)
        spoken_answer  = translate(_to_speech(english_answer), language)
        tts_language   = language if spoken_answer != english_answer else "en"
        return spoken_answer, tts_language

    def speak(self, text: str, language: str = "en") -> None:
        if self.tts is None:
            raise RuntimeError("TTS not loaded. Initialize with text_only=False.")
        self.tts.speak(text, language)

    # ── Session ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reset for a new customer.
        New thread_id = fresh conversation history.
        clear_cart()  = empty the in-memory cart.
        """
        from agent.tools import get_inventory
        self._session_id = self._new_session_id()
        get_inventory().clear_cart()
        print(f"[Kiosk] Session reset — new customer. (session={self._session_id})")

    @staticmethod
    def _new_session_id() -> str:
        return f"session-{uuid.uuid4().hex[:8]}"


# ── TTS text formatter ────────────────────────────────────────────────────

def _to_speech(text: str) -> str:
    """
    Convert structured tool output to natural, speakable text for TTS.
    Printed output keeps the formatted version; only voice gets this.
    """
    import re

    # Inventory result
    if text.startswith("AVAILABLE"):
        names = re.findall(r"•\s+([\w\s\-]+?)\s+[\d]+mg", text)
        if not names:
            names = re.findall(r"•\s+([\w\s\-]+?)\s+—", text)
        if names:
            listed = ", ".join(n.strip() for n in names)
            return f"Yes, we have {len(names)} options: {listed}. Which one would you like?"
        return "Yes, we have it in stock."

    if text.startswith("NOT_FOUND") or text.startswith("Sorry,"):
        return re.sub(r"NOT_FOUND:\s*", "Sorry, ", text)

    # Receipt
    if "MedAssist Pharmacy" in text:
        items = re.findall(r"([\w\s]+?)\s+\([\w\s]+\)\n\s+(\d+)\s+\w+\(s\)\s+×\s+₹(\d+)", text)
        total = re.search(r"TOTAL\s+₹([\d,]+)", text)
        parts = [f"{name.strip()}, {qty} at {price} rupees" for name, qty, price in items]
        total_str = f"Total {total.group(1)} rupees." if total else ""
        return "Receipt ready. " + " ".join(parts) + " " + total_str

    # Add to cart
    if "Added:" in text:
        return text.replace("₹", " rupees ").replace("×", "").replace("\n", ". ")

    # Generic: just remove symbols TTS can't handle
    text = text.replace("₹", " rupees ").replace("×", " times ").replace("•", "")
    text = re.sub(r"[=\-]{3,}", "", text)
    return re.sub(r"\n+", ". ", text).strip()


# ── Text-mode entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    kiosk = PharmacyKiosk(text_only=True)
    print("Pharmacy Kiosk — Text Mode")
    print("'reset' = new customer  |  'quit' = exit\n")

    while True:
        try:
            query = input("Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "reset":
            kiosk.reset()
            print("--- New customer session ---\n")
            continue

        answer = kiosk.process_text(query)
        print(f"Kiosk: {answer}\n")
