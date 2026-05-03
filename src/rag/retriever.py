"""Retrieve top-k relevant schema documents for a given question."""
from src.rag.embedder import embed_text
from src.rag.vector_store import load_collection


def retrieve_schemas(question: str, db_id: str | None = None, top_k: int = 3) -> list[str]:
    """Return top-k schema text snippets relevant to the question.

    If db_id is provided, restrict results to that database.
    """
    collection = load_collection()
    query_embedding = embed_text(question)

    where = {"db_id": db_id} if db_id else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return results["documents"][0]


def build_schema_context(question: str, db_id: str | None = None, top_k: int = 3) -> str:
    """Return a formatted schema context string ready for prompt injection."""
    schemas = retrieve_schemas(question, db_id=db_id, top_k=top_k)
    return "\n\n".join(schemas)
