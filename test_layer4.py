"""
test_layer4.py

Layer 4 tests split into two parts:

Part A — Tools (no Ollama needed)
  Tests that the tool functions call the retriever correctly
  and return valid drug info strings.

Part B — LLM integration (needs Ollama running)
  Tests the full tool-calling loop: query → tool call → retrieval → answer
  Skipped automatically if Ollama is not running.

Run: python test_layer4.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Part A: Tool functions (no Ollama needed) ──────────────────────────────

def test_drug_lookup_tool():
    print("--- Test: drug_lookup tool ---")
    from agent.tools import drug_lookup

    result = drug_lookup("paracetamol fever dose")
    assert isinstance(result, str),     "Result must be a string"
    assert "Paracetamol" in result,     "Should retrieve Paracetamol"
    assert "dosage" in result.lower() or "dose" in result.lower(), \
        "Should contain dosage info"
    print(f"  drug_lookup('paracetamol fever dose') → Paracetamol found ✓")
    print(f"  First 80 chars: '{result[:80]}' ✓\n")


def test_check_interaction_tool():
    print("--- Test: check_interaction tool ---")
    from agent.tools import check_interaction

    result = check_interaction("ibuprofen", "aspirin")
    assert "Ibuprofen" in result, "Should contain Ibuprofen profile"
    assert "Aspirin" in result,   "Should contain Aspirin profile"
    print("  check_interaction('ibuprofen', 'aspirin') → both drugs found ✓\n")


def test_find_generic_tool():
    print("--- Test: find_generic tool ---")
    from agent.tools import find_generic

    result = find_generic("Crocin")    # brand name for Paracetamol
    assert "Paracetamol" in result,    "Should match Paracetamol via brand name"
    assert "Generic available" in result
    print("  find_generic('Crocin') → Paracetamol (generic info) ✓\n")


def test_execute_tool_dispatch():
    print("--- Test: execute_tool dispatches correctly ---")
    from agent.tools import execute_tool

    result = execute_tool("drug_lookup", {"query": "metformin diabetes"})
    assert "Metformin" in result, "Should return Metformin info"

    result = execute_tool("unknown_tool", {"query": "x"})
    assert "Unknown tool" in result, "Should handle unknown tool gracefully"

    print("  execute_tool dispatches and handles unknowns ✓\n")


def test_tool_schemas_valid():
    print("--- Test: Tool schemas have correct structure ---")
    from agent.tools import TOOL_SCHEMAS

    assert len(TOOL_SCHEMAS) == 3, "Should have exactly 3 tools"
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert len(fn["description"]) > 20, "Description must be meaningful for LLM"

    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert "drug_lookup"       in names
    assert "check_interaction" in names
    assert "find_generic"      in names
    print(f"  All 3 schemas valid: {names} ✓\n")


# ── Part B: LLM integration (needs Ollama) ────────────────────────────────

def test_llm_connection():
    print("--- Test: Ollama connection ---")
    from agent.llm import OllamaLLM
    llm = OllamaLLM()
    print("  Ollama connected ✓\n")
    return llm


def test_llm_drug_query(llm):
    print("--- Test: LLM answers drug query with tool call ---")
    answer = llm.chat("What is the dose of paracetamol for a child with fever?")
    assert isinstance(answer, str) and len(answer) > 20, "Should return a real answer"
    # The answer should contain dosage info retrieved from RAG
    assert any(word in answer.lower() for word in ["mg", "dose", "kg", "milligram"]), \
        "Answer should contain dosage information"
    print(f"  Answer: '{answer[:100]}...' ✓\n")


def test_llm_interaction_query(llm):
    print("--- Test: LLM handles drug interaction query ---")
    answer = llm.chat("Can I take ibuprofen and aspirin together?")
    assert isinstance(answer, str) and len(answer) > 20
    print(f"  Answer: '{answer[:100]}...' ✓\n")


def test_llm_generic_query(llm):
    print("--- Test: LLM finds generic alternative ---")
    answer = llm.chat("Is there a cheaper alternative to Crocin?")
    assert isinstance(answer, str) and len(answer) > 20
    print(f"  Answer: '{answer[:100]}...' ✓\n")


if __name__ == "__main__":
    print("=" * 55)
    print("Layer 4 Tests — LLM + Tools")
    print("=" * 55 + "\n")

    # Part A — always runs
    print("PART A: Tool functions (no Ollama needed)")
    print("-" * 40)
    test_drug_lookup_tool()
    test_check_interaction_tool()
    test_find_generic_tool()
    test_execute_tool_dispatch()
    test_tool_schemas_valid()
    print("Part A complete ✓\n")

    # Part B — only if Ollama is running
    print("PART B: LLM integration (needs Ollama)")
    print("-" * 40)
    try:
        llm = test_llm_connection()
        test_llm_drug_query(llm)
        test_llm_interaction_query(llm)
        test_llm_generic_query(llm)
        print("Part B complete ✓")
    except ConnectionError as e:
        print(f"  SKIPPED — {e}")
        print("  Install Ollama and run: ollama pull phi3:mini")

    print("\n" + "=" * 55)
    print("Layer 4 tests done.")
    print("=" * 55)
