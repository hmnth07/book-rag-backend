# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
import json

from app.rag import init_store, query_book, retrieve_chunks
import ollama

app = FastAPI(title="Book RAG Backend (FAISS)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.on_event("startup")
def on_startup():
    # builds embeddings + FAISS index in memory
    init_store("data/meditations.txt")

def sse_event(event: str, data) -> str:
    """Format a Server-Sent Event line."""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

@app.get("/ask-stream")
def ask_stream(question: str):
    """
    SSE stream:
      - meta: { chunks: [{id, text}, ...] }
      - token: { text: "..." }  (small deltas)
      - done: {}
    """
    def gen():
        try:
            # 1) Retrieve relevant chunks first
            top = retrieve_chunks(question, k=5)

            # 2) Send metadata so the UI can show Sources immediately
            meta = {"chunks": [{"id": c["id"], "text": c["text"]} for c in top]}
            yield sse_event("meta", meta)

            # 3) Build prompt from chunks
            context = "\n\n---\n\n".join([c["text"] for c in top])
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

            # 4) Stream tokens from Ollama
            for part in ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": prompt}],
                stream=True,  # <- streaming on
            ):
                delta = part.get("message", {}).get("content", "")
                if delta:
                    yield sse_event("token", {"text": delta})

            # 5) Done
            yield sse_event("done", {})

        except Exception as e:
            # Send error to client then close
            yield sse_event("error", {"message": str(e)})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # CORS is handled by middleware; extra header helps some proxies:
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


@app.post("/ask")
def ask(req: QuestionRequest):
    try:
        answer, chunks = query_book(req.question, k=5)
        # Only return id+text for the frontend (score optional)
        return {"answer": answer, "chunks": [{"id": c["id"], "text": c["text"]} for c in chunks]}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
