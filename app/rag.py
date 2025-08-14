# app/rag.py
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from app.utils.chunking import chunk_paragraphs
from app.models.llm_groq import chat as groq_chat, stream_chat as groq_stream

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BOOK = BASE_DIR / "data" / "meditations.txt"

# Global index + chunks (loaded on demand)
_index: Optional[faiss.Index] = None
_index_ids: Optional[np.ndarray] = None
_chunks: List[str] = []

# ----- Embeddings -----
_embedder: Optional[SentenceTransformer] = None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def _normalize_rows(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms

# ----- FAISS index load (on-demand) -----
def _build_index() -> None:
    global _index, _index_ids, _chunks
    if _index is not None and _index_ids is not None:
        return  # already loaded

    book_path = DEFAULT_BOOK
    if not book_path.exists():
        raise FileNotFoundError(f"Book file not found: {book_path}")

    text = book_path.read_text(encoding="utf-8")
    _chunks = chunk_paragraphs(text, overlap=1)

    embedder = _get_embedder()
    embs = embedder.encode(_chunks, batch_size=64, show_progress_bar=True).astype("float32")
    embs = _normalize_rows(embs)

    dim = embs.shape[1]
    _index = faiss.IndexFlatIP(dim)
    _index.add(embs)
    _index_ids = np.arange(len(_chunks), dtype=np.int64)
    print(f"[rag] Loaded {_index.ntotal} embeddings for {_index_ids.size} chunks.")

# ----- Retrieval -----
def retrieve_chunks(query: str, k: int = 5) -> List[Dict]:
    _build_index()  # ensure index is loaded
    q_emb = _get_embedder().encode([query]).astype("float32")
    q_emb = _normalize_rows(q_emb)

    D, I = _index.search(q_emb, k)
    top = []
    for pos, score in zip(I[0], D[0]):
        if pos == -1:
            continue
        chunk_id = int(_index_ids[pos])
        top.append({"id": chunk_id, "text": _chunks[chunk_id], "score": float(score)})
    return top

# ----- Prompt & Generation -----
def build_prompt(question: str, top_chunks: List[Dict]) -> str:
    context = "\n\n---\n\n".join([c["text"] for c in top_chunks])
    return f"""You are a helpful assistant answering questions using only the provided context from *Meditations* by Marcus Aurelius.

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

def generate_answer(question: str, top_chunks: List[Dict]) -> str:
    prompt = build_prompt(question, top_chunks)
    return groq_chat(prompt)

def stream_answer(question: str, top_chunks: List[Dict]):
    prompt = build_prompt(question, top_chunks)
    for token in groq_stream(prompt):
        yield token

def query_book(question: str, k: int = 5) -> (str, List[Dict]):
    top = retrieve_chunks(question, k=k)
    answer = generate_answer(question, top)
    return answer, top
