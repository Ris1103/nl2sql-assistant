"""Extract schema text documents from Spider SQLite databases for ChromaDB ingestion."""
import json
import sqlite3
from pathlib import Path


DB_DIR = Path(__file__).parents[2] / "data" / "databases"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


def extract_schema(db_path: Path) -> dict:
    """Return a dict with db_id, tables list, and a human-readable schema string."""
    db_id = db_path.stem
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cursor.fetchall()]

    schema_parts = [f"Database: {db_id}"]
    table_docs = []

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        cols = cursor.fetchall()

        cursor.execute(f"PRAGMA foreign_key_list({table});")
        fks = cursor.fetchall()

        col_descriptions = []
        for col in cols:
            # col: (cid, name, type, notnull, dflt_value, pk)
            pk_marker = " [PK]" if col[5] else ""
            col_descriptions.append(f"{col[1]} {col[2]}{pk_marker}")

        fk_descriptions = []
        for fk in fks:
            # fk: (id, seq, table, from, to, ...)
            fk_descriptions.append(f"{fk[3]} -> {fk[2]}.{fk[4]}")

        table_text = f"Table: {table}\n  Columns: {', '.join(col_descriptions)}"
        if fk_descriptions:
            table_text += f"\n  Foreign keys: {', '.join(fk_descriptions)}"

        schema_parts.append(table_text)
        table_docs.append({
            "db_id": db_id,
            "table_name": table,
            "doc_id": f"{db_id}__{table}",
            "text": f"Database: {db_id}\n{table_text}",
        })

    conn.close()

    return {
        "db_id": db_id,
        "schema_text": "\n".join(schema_parts),
        "table_docs": table_docs,
    }


def extract_all() -> list[dict]:
    """Walk DB_DIR, extract schemas from all .sqlite files, save to JSONL."""
    db_files = sorted(DB_DIR.rglob("*.sqlite"))
    if not db_files:
        print(f"No .sqlite files found in {DB_DIR}")
        print("Run the Spider download first, then copy databases/ folder.")
        return []

    all_docs = []
    for db_path in db_files:
        try:
            result = extract_schema(db_path)
            all_docs.extend(result["table_docs"])
        except Exception as e:
            print(f"  Warning: failed on {db_path.name}: {e}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "schema_docs.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc) + "\n")

    print(f"Extracted {len(all_docs)} table documents from {len(db_files)} databases -> {out_path}")
    return all_docs


if __name__ == "__main__":
    extract_all()
