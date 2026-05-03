"""FastAPI serving layer: /query, /health, /databases endpoints."""
import sqlite3
from pathlib import Path

import requests
import sqlparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag.retriever import build_schema_context


OLLAMA_URL = "http://localhost:11434"
MODEL = "phi3-nl2sql"
DB_DIR = Path(__file__).parents[2] / "data" / "databases"

SYSTEM_PROMPT = (
    "You are an expert SQL assistant. "
    "Given a database schema and a natural language question, generate a correct SQL query. "
    "Output only the SQL with no explanation."
)

app = FastAPI(title="NL2SQL Assistant", version="1.0.0")


class QueryRequest(BaseModel):
    question: str
    db_id: str | None = None
    execute: bool = False


class QueryResponse(BaseModel):
    sql: str
    db_id: str | None
    results: list | None = None
    error: str | None = None


@app.get("/health")
def health():
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False
    return {"status": "ok", "ollama": ollama_ok, "model": MODEL}


@app.get("/databases")
def list_databases():
    dbs = []
    for p in sorted(DB_DIR.iterdir()):
        if p.is_dir() and (p / f"{p.name}.sqlite").exists():
            dbs.append(p.name)
        elif p.suffix == ".sqlite":
            dbs.append(p.stem)
    return {"databases": dbs}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    schema_ctx = build_schema_context(req.question, db_id=req.db_id)
    prompt = f"{SYSTEM_PROMPT}\n\nSchema:\n{schema_ctx}\n\nQuestion: {req.question}\n\nSQL:"

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=60,
        )
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {e}")

    raw_sql = resp.json()["response"].strip()
    formatted_sql = sqlparse.format(raw_sql, reindent=True, keyword_case="upper")

    results = None
    error = None

    if req.execute and req.db_id:
        db_path = DB_DIR / req.db_id / f"{req.db_id}.sqlite"
        if not db_path.exists():
            db_path = DB_DIR / f"{req.db_id}.sqlite"

        if db_path.exists():
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(formatted_sql)
                rows = cursor.fetchmany(100)
                col_names = [d[0] for d in cursor.description] if cursor.description else []
                results = {"columns": col_names, "rows": [list(r) for r in rows]}
                conn.close()
            except Exception as e:
                error = str(e)
        else:
            error = f"Database '{req.db_id}' not found"

    return QueryResponse(sql=formatted_sql, db_id=req.db_id, results=results, error=error)
