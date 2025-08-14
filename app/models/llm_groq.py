# app/models/llm_groq.py
import os
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # choose your Groq model

_client = None

def client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client

def ask_groq(prompt: str, model: str = None) -> str:
    """Ask a question using Groq API."""
    groq_client = client()
    model_to_use = model or MODEL
    
    response = groq_client.chat.completions.create(
        model=model_to_use,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def stream_chat(prompt: str):
    """
    Yields incremental text chunks (delta strings).
    """
    stream = client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        stream=True,
    )
    for chunk in stream:
        # Each chunk may contain partial delta content
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def chat(prompt: str) -> str:
    """Non-streaming call (for /ask)."""
    resp = client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        stream=False,
    )
    return resp.choices[0].message.content
