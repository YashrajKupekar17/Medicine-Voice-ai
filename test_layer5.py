"""
test_layer5.py

Tests for the orchestrator and memory — no audio hardware needed.
All tests run in text_only=True mode.

Run: python test_layer5.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Memory tests (no Ollama needed) ───────────────────────────────────────

def test_memory_add_and_get():
    print("--- Test: Memory stores and retrieves turns ---")
    from agent.memory import ConversationMemory

    mem = ConversationMemory(max_turns=3)
    mem.add("What is paracetamol?", "Paracetamol is a pain reliever.")
    mem.add("What is the dose?",    "The dose is 500-1000mg every 4-6 hours.")

    msgs = mem.get_messages()
    assert len(msgs) == 4, "2 turns = 4 messages (user + assistant each)"
    assert msgs[0]["role"]    == "user"
    assert msgs[1]["role"]    == "assistant"
    assert "paracetamol" in msgs[0]["content"].lower()
    print(f"  2 turns → {len(msgs)} messages ✓\n")


def test_memory_sliding_window():
    print("--- Test: Memory evicts oldest turn when full ---")
    from agent.memory import ConversationMemory

    mem = ConversationMemory(max_turns=2)
    mem.add("Turn 1 question", "Turn 1 answer")
    mem.add("Turn 2 question", "Turn 2 answer")
    mem.add("Turn 3 question", "Turn 3 answer")  # should evict Turn 1

    assert mem.turn_count == 2, "Should only keep 2 turns"
    msgs = mem.get_messages()
    combined = " ".join(m["content"] for m in msgs)
    assert "Turn 1" not in combined, "Turn 1 should have been evicted"
    assert "Turn 2" in combined
    assert "Turn 3" in combined
    print("  Oldest turn correctly evicted at max_turns=2 ✓\n")


def test_memory_clear():
    print("--- Test: Memory clears on reset ---")
    from agent.memory import ConversationMemory

    mem = ConversationMemory()
    mem.add("question", "answer")
    assert mem.turn_count == 1
    mem.clear()
    assert mem.turn_count == 0
    assert mem.get_messages() == []
    print("  Memory clears correctly ✓\n")


# ── Orchestrator tests (needs Ollama) ─────────────────────────────────────

def test_orchestrator_init():
    print("--- Test: Orchestrator initializes in text_only mode ---")
    from agent.orchestrator import MedAssistant
    assistant = MedAssistant(text_only=True)
    assert assistant.stt is None, "STT should not load in text_only mode"
    assert assistant.tts is None, "TTS should not load in text_only mode"
    assert assistant.llm  is not None
    assert assistant.memory is not None
    print("  Initialized in text_only mode (no STT/TTS loaded) ✓\n")
    return assistant


def test_smalltalk_bypass(assistant):
    print("--- Test: Small talk bypasses LLM ---")
    # Small talk should return instantly — no LLM call needed
    import time

    t0 = time.time()
    response = assistant.process_text("hello")
    elapsed = time.time() - t0

    assert "ask" in response.lower() or "medicine" in response.lower()
    assert elapsed < 0.5, f"Small talk should be instant, took {elapsed:.2f}s"
    print(f"  'hello' → '{response}' in {elapsed:.3f}s ✓\n")


def test_drug_query(assistant):
    print("--- Test: Drug query goes through LLM + RAG ---")
    answer = assistant.process_text("What is the dose of paracetamol for fever?")
    assert isinstance(answer, str) and len(answer) > 20
    assert any(w in answer.lower() for w in ["mg", "dose", "kg", "milligram", "500"])
    print(f"  Answer: '{answer[:90]}...' ✓\n")


def test_memory_in_conversation(assistant):
    print("--- Test: Memory persists across turns ---")
    assistant.reset()

    # Turn 1
    answer1 = assistant.process_text("Tell me about metformin.")
    assert "metformin" in answer1.lower() or "diabetes" in answer1.lower()

    # Turn 2 — "its" refers to metformin from context
    answer2 = assistant.process_text("What are its side effects?")
    assert isinstance(answer2, str) and len(answer2) > 10

    assert assistant.memory.turn_count == 2, "Should have 2 turns stored"
    print(f"  Turn 1: '{answer1[:60]}...'")
    print(f"  Turn 2: '{answer2[:60]}...'")
    print(f"  Memory has {assistant.memory.turn_count} turns ✓\n")


def test_session_reset(assistant):
    print("--- Test: Session reset clears memory ---")
    assistant.reset()
    assert assistant.memory.turn_count == 0
    print("  Memory cleared after reset ✓\n")


def test_interaction_query(assistant):
    print("--- Test: Drug interaction query ---")
    answer = assistant.process_text("Can I give aspirin to a child for fever?")
    assert isinstance(answer, str) and len(answer) > 20
    # Should warn about Reye syndrome (it's in the CSV)
    print(f"  Answer: '{answer[:100]}...' ✓\n")


if __name__ == "__main__":
    print("=" * 55)
    print("Layer 5 Tests — Orchestrator + Memory")
    print("=" * 55 + "\n")

    # Memory tests — always run
    print("PART A: Memory (no Ollama needed)")
    print("-" * 40)
    test_memory_add_and_get()
    test_memory_sliding_window()
    test_memory_clear()
    print("Part A complete ✓\n")

    # Orchestrator tests — needs Ollama
    print("PART B: Orchestrator (needs Ollama)")
    print("-" * 40)
    try:
        assistant = test_orchestrator_init()
        test_smalltalk_bypass(assistant)
        test_drug_query(assistant)
        test_memory_in_conversation(assistant)
        test_session_reset(assistant)
        test_interaction_query(assistant)
        print("Part B complete ✓")
    except ConnectionError as e:
        print(f"  SKIPPED — {e}")

    print("\n" + "=" * 55)
    print("Layer 5 tests done.")
    print("=" * 55)
    print()
    print("Next: run the interactive text demo:")
    print("  python agent/orchestrator.py")
