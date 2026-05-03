"""Generate embeddings via nomic-embed-text through the Ollama API."""
import requests


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


def embed_text(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def embed_batch(texts: list[str]) -> list[list[float]]:
    return [embed_text(t) for t in texts]
