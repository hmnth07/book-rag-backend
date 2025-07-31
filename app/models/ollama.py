import ollama

def ask_ollama(prompt: str) -> str:
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']
