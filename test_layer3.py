"""
test_layer3.py — Tests for the RAG pipeline (Layer 3)
Run: python test_layer3.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_retriever_loads():
    print("--- Test: Retriever loads index ---")
    from rag.retriever import DrugRetriever
    r = DrugRetriever()
    assert r.index.ntotal == 23, "Should have 23 drug vectors"
    assert len(r.chunks) == 23,  "Should have 23 text chunks"
    print(f"  Index: {r.index.ntotal} vectors ✓\n")


def test_correct_drug_retrieved():
    print("--- Test: Correct drug returned for exact query ---")
    from rag.retriever import DrugRetriever
    r = DrugRetriever()

    results = r.retrieve("paracetamol dose for fever", top_k=1)
    assert len(results) == 1
    assert "Paracetamol" in results[0], "Top result should be Paracetamol"
    print("  'paracetamol dose for fever' → Paracetamol ✓")

    results = r.retrieve("metformin for diabetes side effects", top_k=1)
    assert "Metformin" in results[0], "Top result should be Metformin"
    print("  'metformin for diabetes side effects' → Metformin ✓\n")


def test_interaction_query():
    print("--- Test: Drug interaction query returns both drugs ---")
    from rag.retriever import DrugRetriever
    r = DrugRetriever()

    results = r.retrieve("ibuprofen and aspirin together", top_k=2)
    combined = " ".join(results)
    assert "Ibuprofen" in combined, "Ibuprofen should be in results"
    assert "Aspirin" in combined,   "Aspirin should be in results"
    print("  'ibuprofen and aspirin together' → both drugs retrieved ✓\n")


def test_semantic_match():
    print("--- Test: Semantic matching (not just keyword) ---")
    from rag.retriever import DrugRetriever
    r = DrugRetriever()

    # "sugar disease" is not in the CSV — test semantic understanding
    results = r.retrieve("medicine for sugar disease", top_k=1)
    assert "Metformin" in results[0], "Should semantically match diabetes → Metformin"
    print("  'medicine for sugar disease' → Metformin (semantic match) ✓\n")


def test_top_k():
    print("--- Test: top_k controls result count ---")
    from rag.retriever import DrugRetriever
    r = DrugRetriever()

    assert len(r.retrieve("antibiotic", top_k=1)) == 1
    assert len(r.retrieve("antibiotic", top_k=3)) == 3
    print("  top_k=1 → 1 result, top_k=3 → 3 results ✓\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Layer 3 Tests — RAG")
    print("=" * 50 + "\n")

    test_retriever_loads()
    test_correct_drug_retrieved()
    test_interaction_query()
    test_semantic_match()
    test_top_k()

    print("=" * 50)
    print("All Layer 3 tests passed.")
    print("=" * 50)
