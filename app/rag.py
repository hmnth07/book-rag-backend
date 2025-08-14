import os
from typing import List, Dict, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ---- Embeddings ----
# Use same model here and in scripts/build_index.py
_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embedding_model = SentenceTransformer(_EMBED_MODEL_NAME)

# ---- Groq client ----
_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not _GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment variables.")
_groq = Groq(api_key=_GROQ_API_KEY)

# ---- Chunking ----
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Simple overlapping character-based chunking.
    Tune sizes as needed; keep consistent with index build.
    """
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start += chunk_size - overlap
    return chunks

# ---- Build FAISS (used by the build script; not used at runtime on Render) ----
def build_faiss_index(chunks: List[str]) -> faiss.Index:
    """
    Build a FAISS index (L2) over normalized embeddings.
    """
    embeddings = _embedding_model.encode(
        chunks, convert_to_numpy=True, normalize_embeddings=True
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index

# ---- Retrieval ----
def retrieve_chunks(query: str, chunks: List[str], index: faiss.Index, k: int = 5) -> List[Dict]:
    """
    Search top-k similar chunks. Returns [{id, text, score}, ...]
    """
    q_emb = _embedding_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    D, I = index.search(q_emb, k)
    out: List[Dict] = []
    for pos, dist in zip(I[0], D[0]):
        if pos == -1:
            continue
        out.append({
            "id": int(pos),
            "text": chunks[pos],
            "score": float(dist)
        })
    return out

# ---- Prompting ----
def _build_prompt(question: str, top_chunks: List[Dict]) -> str:
    context = "\n\n---\n\n".join([c["text"] for c in top_chunks])
    return f"""You are a helpful assistant answering questions using only the provided context from *Meditations* by Marcus Aurelius.

Context:
---
{context}
---

Instructions:
- Only use information from the context above.
- If the answer is not in the context, reply exactly: "The book does not say directly."
- Be concise and faithful to the text.

Question: {question}
Answer:"""

# ---- Generation (non-streaming) ----
def generate_answer(question: str, top_chunks: List[Dict]) -> str:
    prompt = _build_prompt(question, top_chunks)
    resp = _groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return resp.choices[0].message.content

# ---- Generation (streaming) ----
def stream_answer(question: str, top_chunks: List[Dict]):
    prompt = _build_prompt(question, top_chunks)
    stream = _groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

# ---- High-level convenience (non-streaming) ----
def query_book(question: str, chunks: List[str], index: faiss.Index, k: int = 5) -> Tuple[str, List[Dict]]:
    top = retrieve_chunks(question, chunks, index, k=k)
    answer = generate_answer(question, top)
    return answer, top
