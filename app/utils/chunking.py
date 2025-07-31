def chunk_paragraphs(text, overlap=1):
    paragraphs = text.split('\n\n')
    chunks = []
    for i in range(0, len(paragraphs), max(1, len(paragraphs) // 200)):
        chunk = "\n\n".join(paragraphs[i:i + 3])
        if i > 0 and overlap:
            chunk = "\n\n".join(paragraphs[i - overlap:i + 3])
        chunks.append(chunk.strip())
    return chunks
