"""Generate embeddings via nomic-embed-text through the Ollama API."""
import time

import requests


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


def embed_text(text: str, retries: int = 5) -> list[float]:
    for attempt in range(retries):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except (requests.HTTPError, requests.ConnectionError):
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def embed_batch(texts: list[str]) -> list[list[float]]:
    embeddings = [embed_text(t) for t in texts]
    time.sleep(0.5)  # brief pause between batches to avoid overwhelming Ollama
    return embeddings
