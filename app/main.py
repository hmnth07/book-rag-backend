# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import os
import asyncio
import logging

from app.rag import query_book, retrieve_chunks, stream_answer

from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Book RAG Backend (FAISS + Groq)")

# CORS
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

# ---- SSE Helper ----
def sse_event(event: str, data) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

# ---- Stream endpoint ----
@app.get("/ask-stream")
async def ask_stream(question: str):
    async def event_generator():
        try:
            # retrieve top-k chunks
            top = retrieve_chunks(question, k=5)
            yield sse_event("meta", {"chunks": [{"id": c["id"], "text": c["text"]} for c in top]})

            # stream tokens
            loop = asyncio.get_event_loop()
            for token in await loop.run_in_executor(None, lambda: list(stream_answer(question, top))):
                yield sse_event("token", {"text": token})

            yield sse_event("done", {})
        except Exception as e:
            logging.exception("Streaming error:")
            yield sse_event("error", {"message": str(e)})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Transfer-Encoding": "chunked",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)

# ---- Non-stream endpoint ----
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
        logging.exception("Ask endpoint error:")
        raise HTTPException(status_code=500, detail=str(e))
