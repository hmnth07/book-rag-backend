# Build and save FAISS index + chunks locally (commit them for Render).
# Run from repo root:
#   python -m scripts.build_index

import os
import pickle
import faiss
from app.rag import chunk_text, build_faiss_index

DATA_DIR = "data"
SRC_PATH = os.path.join(DATA_DIR, "meditations.txt")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")

def main():
    if not os.path.exists(SRC_PATH):
        raise FileNotFoundError(f"Missing source text at {SRC_PATH}")

    os.makedirs(DATA_DIR, exist_ok=True)

    with open(SRC_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=900, overlap=150)
    print(f"Chunked {SRC_PATH} into {len(chunks)} chunks")

    index = build_faiss_index(chunks)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Saved {INDEX_PATH} and {CHUNKS_PATH}")

if __name__ == "__main__":
    main()
