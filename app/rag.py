# app/rag.py
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama

from app.utils.chunking import chunk_paragraphs

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BOOK = BASE_DIR / "data" / "meditations.txt"

# ----- Loading & Chunking -----

def load_chunks(path: Optional[Union[str, Path]] = None) -> List[str]:
    book_path = Path(path) if path else DEFAULT_BOOK
    if not book_path.is_absolute():
        book_path = (BASE_DIR / book_path).resolve()
    if not book_path.exists():
        raise FileNotFoundError(f"Book file not found at: {book_path}")
    text = book_path.read_text(encoding="utf-8")
    chunks = chunk_paragraphs(text, overlap=1)  # 3-paragraph chunks w/ small overlap
    return chunks

# ----- Embeddings & Index -----

_embedder: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_index_ids: Optional[np.ndarray] = None
_chunks: List[str] = []

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        # small, fast, good quality
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def _normalize_rows(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms

def build_index(chunks: List[str]) -> Tuple[faiss.Index, np.ndarray]:
    """
    Build a cosine-similarity FAISS index using inner product on L2-normalized vectors.
    Returns (index, ids) where ids map FAISS positions -> chunk indices.
    """
    embedder = _get_embedder()
    embs = embedder.encode(chunks, batch_size=64, show_progress_bar=True)
    embs = embs.astype("float32")
    embs = _normalize_rows(embs)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine if inputs are normalized
    index.add(embs)
    ids = np.arange(len(chunks), dtype=np.int64)
    return index, ids

def init_store(book_path: Optional[Union[str, Path]] = None) -> None:
    """Load chunks and build FAISS index on startup."""
    global _chunks, _index, _index_ids
    _chunks = load_chunks(book_path)
    _index, _index_ids = build_index(_chunks)
    print(f"[rag] Loaded {_index.ntotal} embeddings for {_index_ids.size} chunks.")

# ----- Retrieval -----

def retrieve_chunks(query: str, k: int = 5) -> List[Dict]:
    """
    Search top-k similar chunks via FAISS and return [{id, text}, ...].
    """
    if _index is None or _index_ids is None or not _chunks:
        raise RuntimeError("Index not initialized. Call init_store() on startup.")

    q_emb = _get_embedder().encode([query]).astype("float32")
    q_emb = _normalize_rows(q_emb)

    # FAISS returns distances (inner product) and indices
    D, I = _index.search(q_emb, k)
    top = []
    for pos, score in zip(I[0], D[0]):
        if pos == -1:
            continue
        chunk_id = int(_index_ids[pos])
        top.append({"id": chunk_id, "text": _chunks[chunk_id], "score": float(score)})
    return top

# ----- Generation -----

def generate_answer(question: str, top_chunks: List[Dict]) -> str:
    context = "\n\n---\n\n".join([c["text"] for c in top_chunks])

    prompt = f"""You are a helpful assistant answering questions using only the provided context from *Meditations* by Marcus Aurelius.

Context:
---
{context}
---

Instructions:
- Only use information from the context above.
- If the answer is not in the context, reply: "The book does not say directly."
- Be concise and faithful to the text.

Question: {question}
Answer:"""

    resp = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp["message"]["content"]

def query_book(question: str, k: int = 5) -> Tuple[str, List[Dict]]:
    top = retrieve_chunks(question, k=k)
    answer = generate_answer(question, top)
    return answer, top