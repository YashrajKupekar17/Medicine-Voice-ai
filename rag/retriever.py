"""
rag/retriever.py

Loads the pre-built FAISS index and retrieves relevant drug chunks
for a given query at runtime.

Usage:
    retriever = DrugRetriever()
    chunks = retriever.retrieve("what is the dose of paracetamol for fever")
    # → returns top-3 most relevant drug text chunks

This is what the agent tools (Layer 4) will call.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import MODELS_DIR, DATA_DIR

INDEX_PATH  = Path(DATA_DIR) / "faiss.index"
CHUNKS_PATH = Path(DATA_DIR) / "chunks.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class DrugRetriever:
    def __init__(self):
        print("[Retriever] Loading FAISS index and embedding model...")

        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Index not found at {INDEX_PATH}. Run 'python rag/ingest.py' first."
            )

        # Load FAISS index (just memory-mapped — very fast)
        self.index = faiss.read_index(str(INDEX_PATH))

        # Load text chunks — parallel list to the index
        # index position 0 → chunks[0], position 1 → chunks[1], etc.
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks: list[str] = pickle.load(f)

        # Load same embedding model used during ingestion
        # Must be identical — different models produce incompatible vector spaces
        self.model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODELS_DIR)

        print(f"[Retriever] Ready. Index has {self.index.ntotal} drug vectors.")

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """
        Find the top_k most relevant drug chunks for a given query.

        How it works:
          1. Embed the query → 384-dim float32 vector
          2. FAISS finds the top_k index positions with highest dot product
             (= cosine similarity, since vectors are normalized)
          3. Return the text chunks at those positions

        top_k=3 means: return the 3 most relevant drug profiles.
        The LLM in Layer 4 will read these 3 chunks and answer from them.
        """
        # Embed query — shape: (1, 384)
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # FAISS search — returns (scores, indices) each shape (1, top_k)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue    # FAISS returns -1 for empty slots
            results.append(self.chunks[idx])
            print(f"[Retriever] Match (score={score:.3f}): {self.chunks[idx].split(chr(10))[0]}")

        return results


if __name__ == "__main__":
    r = DrugRetriever()

    test_queries = [
        "what is the dose of paracetamol for fever in children",
        "can I take ibuprofen and aspirin together",
        "what antibiotic for urinary tract infection",
        "affordable generic medicine for diabetes",
        "side effects of metformin",
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: '{query}'")
        print(f"{'='*50}")
        results = r.retrieve(query, top_k=2)
        for i, chunk in enumerate(results, 1):
            print(f"\n[Result {i}]\n{chunk}")
