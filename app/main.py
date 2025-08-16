# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import os

from app.rag import query_book, retrieve_chunks, stream_answer, _build_index

from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="Book RAG Backend (FAISS + Groq)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://marcusaurelius-chat.vercel.app",
        "http://localhost:5137"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

def sse_event(event: str, data) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

@app.on_event("startup")
async def startup_event():
    """Preload FAISS index so first request is fast."""
    try:
        _build_index()
        print("[startup] FAISS index loaded successfully.")
    except Exception as e:
        print(f"[startup] Failed to build index: {e}")

@app.get("/ask-stream")
def ask_stream(question: str):
    def gen():
        try:
            # Send immediate "ping" so Render doesn’t timeout
            yield sse_event("ping", {"status": "connected"})

            # retrieve top-k chunks
            top = retrieve_chunks(question, k=5)
            yield sse_event("meta", {"chunks": [{"id": c["id"], "text": c["text"]} for c in top]})

            # stream tokens from Groq
            for token in stream_answer(question, top):
                yield sse_event("token", {"text": token})

            yield sse_event("done", {})
        except Exception as e:
            yield sse_event("error", {"message": str(e)})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # prevents buffering
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

@app.post("/ask")
def ask(req: QuestionRequest):
    try:
        answer, chunks = query_book(req.question, k=5)
        return {
            "answer": answer,
            "chunks": [{"id": c["id"], "text": c["text"]} for c in chunks]
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
