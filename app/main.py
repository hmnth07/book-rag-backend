from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import faiss
import pickle
from typing import List, Dict
from app.rag import retrieve_chunks, query_book, stream_answer

app = FastAPI(title="Book RAG Backend (FAISS + Groq Llama)")

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

# Store loaded data on app.state
@app.on_event("startup")
def _load_index_and_chunks():
    try:
        app.state.faiss_index = faiss.read_index("data/faiss_index.bin")
    except Exception as e:
        raise RuntimeError(
            "Failed to read data/faiss_index.bin. "
            "Run `python -m scripts.build_index` first."
        ) from e

    try:
        with open("data/chunks.pkl", "rb") as f:
            app.state.chunks = pickle.load(f)
    except Exception as e:
        raise RuntimeError(
            "Failed to read data/chunks.pkl. "
            "Run `python -m scripts.build_index` first."
        ) from e

def sse_event(event: str, data) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

@app.get("/ask-stream")
def ask_stream(question: str):
    def gen():
        try:
            top = retrieve_chunks(question, app.state.chunks, app.state.faiss_index, k=5)
            # Send supporting chunks first
            yield sse_event("meta", {"chunks": [{"id": c["id"], "text": c["text"]} for c in top]})
            # Stream tokens
            for delta in stream_answer(question, top):
                yield sse_event("token", {"text": delta})
            yield sse_event("done", {})
        except Exception as e:
            yield sse_event("error", {"message": str(e)})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

@app.post("/ask")
def ask(req: QuestionRequest):
    try:
        answer, chunks = query_book(req.question, app.state.chunks, app.state.faiss_index, k=5)
        return {"answer": answer, "chunks": [{"id": c["id"], "text": c["text"]} for c in chunks]}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
