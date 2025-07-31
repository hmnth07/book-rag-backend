# Book RAG Backend

FastAPI backend for the Book Q&A Bot using a local Ollama (Mistral) model.

🚀 To run:

```bash
cd book-rag-backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```

If you get "Address already in use" error, use a different port:

```bash
uvicorn app.main:app --reload --port 8001
```
