"""
rag/ingest.py

One-time ingestion pipeline:
  medicines.csv  →  text chunks  →  embeddings  →  FAISS index (saved to disk)

Run once:  python rag/ingest.py
Re-run only when you update medicines.csv.

Why this is separate from retrieval:
  Ingestion is slow (embedding 23 drugs takes a few seconds).
  Retrieval must be fast (<100ms per query).
  By saving the FAISS index to disk, we pay the ingestion cost once,
  then the retriever just loads the pre-built index instantly.
"""

from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import MODELS_DIR, DATA_DIR

# ── Paths ──────────────────────────────────────────────────────────────────
MEDICINES_CSV  = Path(DATA_DIR) / "medicines.csv"
INDEX_PATH     = Path(DATA_DIR) / "faiss.index"   # the FAISS binary index
CHUNKS_PATH    = Path(DATA_DIR) / "chunks.pkl"    # the raw text chunks (parallel to index)

# ── Embedding model ────────────────────────────────────────────────────────
# all-MiniLM-L6-v2:
#   - 80MB download, cached after first run
#   - 384-dimensional embeddings
#   - Fast on CPU (~5ms per sentence)
#   - Good semantic understanding ("dose" ≈ "dosage" ≈ "how much to take")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_csv(path: Path) -> list[dict]:
    """Read medicines.csv into a list of dicts, one per row."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_chunk(row: dict) -> str:
    """
    Convert one CSV row into a single rich text chunk.

    Why one chunk per drug (not splitting into dosage/interactions/etc.)?
    - Our database is small (~20-50 drugs)
    - A drug query is usually about one drug — returning the full drug profile
      gives the LLM all the context it needs in one hit
    - Splitting would require more chunks, more index entries, more complexity
    - For a database of 10,000+ drugs, you'd split by field

    Why natural language instead of JSON?
    - Embeddings are trained on natural language text
    - "The dose is 500mg" embeds more similarly to "What is the dosage?"
      than {"dose": "500mg"} does
    - The LLM also reads the retrieved chunk — natural language is cleaner
    """
    return (
        f"Drug: {row['name']} "
        f"(also known as {row['brand_names'].replace('|', ',')})\n"
        f"Category: {row['category']}\n"
        f"Uses: {row['uses']}\n"
        f"Adult dosage: {row['dosage_adult']}\n"
        f"Child dosage: {row['dosage_child']}\n"
        f"Side effects: {row['side_effects']}\n"
        f"Do not use if (contraindications): {row['contraindications']}\n"
        f"Drug interactions: {row['interactions']}\n"
        f"Generic available: {row['generic_available']} | "
        f"Price: {row['price_category']}"
    )


def embed(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Convert a list of text strings into a 2D float32 numpy array.
    Shape: (len(texts), 384)

    normalize_embeddings=True:
      Makes every vector unit length (magnitude = 1).
      This means cosine similarity = dot product, which FAISS can compute
      much faster than full cosine similarity.
    """
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index from embeddings.

    IndexFlatIP = Flat index with Inner Product (dot product) similarity.
    Because our embeddings are normalized, dot product = cosine similarity.

    Why "Flat"?
    - Exact search — checks every vector, no approximation
    - Fine for our size (23 drugs = 23 vectors)
    - For 100,000+ vectors, use IndexIVFFlat (approximate but fast)

    Why not L2 distance?
    - L2 finds geometrically close vectors
    - IP/cosine finds semantically similar vectors — better for text
    """
    dim = embeddings.shape[1]   # 384
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def ingest():
    print("=" * 50)
    print("RAG Ingestion Pipeline")
    print("=" * 50)

    # Step 1: Load CSV
    print(f"\n[1/4] Loading {MEDICINES_CSV.name}...")
    rows = load_csv(MEDICINES_CSV)
    print(f"      Loaded {len(rows)} drugs")

    # Step 2: Build text chunks
    print("\n[2/4] Building text chunks...")
    chunks = [make_chunk(row) for row in rows]
    print(f"      Built {len(chunks)} chunks")
    print(f"\n      Sample chunk:\n      {'-'*40}")
    for line in chunks[0].split("\n"):
        print(f"      {line}")
    print(f"      {'-'*40}")

    # Step 3: Embed
    print(f"\n[3/4] Loading embedding model '{EMBEDDING_MODEL}'...")
    print("      First run downloads ~80MB, cached after that.")
    model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODELS_DIR)
    embeddings = embed(chunks, model)
    print(f"      Embeddings shape: {embeddings.shape}")  # (23, 384)
    print(f"      dtype: {embeddings.dtype}")

    # Step 4: Build and save FAISS index
    print("\n[4/4] Building FAISS index and saving to disk...")
    index = build_index(embeddings)
    faiss.write_index(index, str(INDEX_PATH))

    # Save chunks in parallel — we need them to map index positions back to text
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"      Index saved  → {INDEX_PATH}")
    print(f"      Chunks saved → {CHUNKS_PATH}")
    print(f"      Index size: {index.ntotal} vectors × {embeddings.shape[1]} dims")

    print("\nIngestion complete. Run retriever.py to test queries.")


if __name__ == "__main__":
    ingest()
