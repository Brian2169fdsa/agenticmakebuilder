"""
TF-IDF Embedding Engine — Lightweight Similarity Search

Provides cosine-similarity search over project briefs without external deps.
Uses term-frequency inverse-document-frequency vectors stored in a JSON file.

Public functions:
    embed_document(doc_id, text, metadata, store_path) — add/update a document
    find_similar(query_text, top_n, store_path) — cosine similarity search
    load_store(store_path) — load embedding store from disk
    save_store(store, store_path) — persist embedding store to disk

Deterministic. No network calls. No external dependencies.
"""

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

DEFAULT_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.json")

# Common stop words to exclude from TF-IDF
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "and", "but", "or",
    "not", "no", "so", "if", "than", "too", "very", "just", "about", "up", "out",
    "that", "this", "it", "i", "me", "my", "we", "our", "they", "them", "their",
    "she", "he", "her", "his", "you", "your", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "only", "same", "then",
    "when", "where", "how", "what", "which", "who", "whom", "while", "are",
})


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, stripping punctuation."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _term_frequency(tokens: list[str]) -> dict[str, float]:
    """Compute normalized term frequency for a token list."""
    counts = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {term: count / total for term, count in counts.items()}


def load_store(store_path: str = DEFAULT_STORE_PATH) -> dict:
    """Load the embedding store from disk."""
    path = Path(store_path)
    if not path.exists():
        return {"version": 1, "vocab": {}, "documents": []}
    with open(path, "r") as f:
        return json.load(f)


def save_store(store: dict, store_path: str = DEFAULT_STORE_PATH):
    """Persist the embedding store to disk."""
    path = Path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(store, f, indent=2)


def _rebuild_idf(store: dict):
    """Recompute IDF values across all documents."""
    n_docs = len(store["documents"])
    if n_docs == 0:
        store["vocab"] = {}
        return

    # Count how many documents each term appears in
    doc_freq = Counter()
    for doc in store["documents"]:
        unique_terms = set(doc.get("tf", {}).keys())
        doc_freq.update(unique_terms)

    # IDF = log(N / df) + 1 (smoothed)
    store["vocab"] = {
        term: math.log(n_docs / df) + 1
        for term, df in doc_freq.items()
    }


def _tfidf_vector(tf: dict, vocab: dict) -> dict[str, float]:
    """Compute TF-IDF vector from term frequencies and global IDF."""
    return {term: freq * vocab.get(term, 1.0) for term, freq in tf.items()}


def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Compute cosine similarity between two sparse vectors."""
    # Dot product
    common_terms = set(vec_a.keys()) & set(vec_b.keys())
    if not common_terms:
        return 0.0

    dot = sum(vec_a[t] * vec_b[t] for t in common_terms)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


def embed_document(doc_id: str, text: str, metadata: dict = None,
                   store_path: str = DEFAULT_STORE_PATH) -> dict:
    """Add or update a document in the embedding store.

    Args:
        doc_id: Unique identifier (e.g. project_id).
        text: Full text to embed (brief + outcome description).
        metadata: Optional dict stored alongside the embedding.
        store_path: Path to the JSON store file.

    Returns:
        dict with doc_id, token_count, vocab_size.
    """
    store = load_store(store_path)

    tokens = _tokenize(text)
    tf = _term_frequency(tokens)

    # Remove existing doc with same id
    store["documents"] = [d for d in store["documents"] if d.get("id") != doc_id]

    store["documents"].append({
        "id": doc_id,
        "text_preview": text[:200],
        "tf": tf,
        "metadata": metadata or {},
    })

    # Rebuild IDF across all docs
    _rebuild_idf(store)

    save_store(store, store_path)

    return {
        "doc_id": doc_id,
        "token_count": len(tokens),
        "vocab_size": len(store["vocab"]),
        "total_documents": len(store["documents"]),
    }


def find_similar(query_text: str, top_n: int = 3,
                 store_path: str = DEFAULT_STORE_PATH) -> list[dict]:
    """Find the top N most similar documents to a query string.

    Args:
        query_text: The search query.
        top_n: Number of results to return.
        store_path: Path to the JSON store file.

    Returns:
        List of dicts with id, score, text_preview, metadata.
    """
    store = load_store(store_path)

    if not store["documents"]:
        return []

    query_tokens = _tokenize(query_text)
    query_tf = _term_frequency(query_tokens)
    query_vec = _tfidf_vector(query_tf, store["vocab"])

    results = []
    for doc in store["documents"]:
        doc_vec = _tfidf_vector(doc.get("tf", {}), store["vocab"])
        score = _cosine_similarity(query_vec, doc_vec)
        if score > 0:
            results.append({
                "id": doc["id"],
                "score": round(score, 4),
                "text_preview": doc.get("text_preview", ""),
                "metadata": doc.get("metadata", {}),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]
