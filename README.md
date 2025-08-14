# Book RAG Backend (Groq + Llama + FAISS)

FastAPI backend that answers questions about a book using:

- FAISS retrieval (prebuilt index)
- HuggingFace embeddings (`all-MiniLM-L6-v2`)
- Groq Llama (`llama-3.1-8b-instant`)
- SSE streaming (`/ask-stream`)

## Setup

1. Install deps

```bash
pip install -r requirements.txt
```
