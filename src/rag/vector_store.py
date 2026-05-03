"""Build and persist a ChromaDB collection of Spider schema documents."""
import json
from pathlib import Path

import chromadb

from src.rag.embedder import embed_batch


CHROMA_DIR = str(Path(__file__).parents[2] / "chroma_db")
COLLECTION_NAME = "spider_schemas"
SCHEMA_DOCS_PATH = Path(__file__).parents[2] / "data" / "processed" / "schema_docs.jsonl"


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_DIR)


def build_collection(batch_size: int = 50) -> chromadb.Collection:
    """Embed all schema docs and store in ChromaDB. Idempotent — deletes existing collection first."""
    client = get_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    docs = []
    with open(SCHEMA_DOCS_PATH, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    print(f"Embedding {len(docs)} schema documents in batches of {batch_size}...")
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        texts = [d["text"] for d in batch]
        embeddings = embed_batch(texts)
        collection.add(
            ids=[d["doc_id"] for d in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"db_id": d["db_id"], "table_name": d["table_name"]} for d in batch],
        )
        print(f"  {min(i + batch_size, len(docs))}/{len(docs)}")

    print(f"Collection '{COLLECTION_NAME}' built with {collection.count()} documents.")
    return collection


def load_collection() -> chromadb.Collection:
    client = get_client()
    return client.get_collection(COLLECTION_NAME)


if __name__ == "__main__":
    build_collection()
