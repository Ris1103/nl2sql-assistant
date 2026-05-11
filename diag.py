import json, requests, sqlite3
from pathlib import Path
from src.rag.retriever import build_schema_context
from src.evaluation.metrics import exact_match, execute_query

OLLAMA_URL = "http://localhost:11434"
MODEL = "phi3-nl2sql"
VAL_FILE = Path("D:/Work/Projects02/nl2sql-assistant/data/processed/val_instruct.jsonl")
DB_DIR = Path("D:/Work/Projects02/nl2sql-assistant/data/databases")

records = []
with open(VAL_FILE, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))
        if len(records) >= 5:
            break

for i, rec in enumerate(records):
    question = rec["input"].split("Question:")[-1].strip() if "Question:" in rec["input"] else rec["input"]
    db_id = rec["input"].split("Database:")[1].split("\n")[0].strip() if "Database:" in rec["input"] else "unknown"
    schema_ctx = build_schema_context(question, db_id=db_id)
    enriched_input = f"Schema:\n{schema_ctx}\n\nQuestion: {question}"
    prompt = f"{rec['instruction']}\n\n{enriched_input}\n\nSQL:"

    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        timeout=60,
    )
    predicted = resp.json()["response"].strip()

    em = exact_match(predicted, rec["output"])
    ok_pred, pred_rows = execute_query(db_id, predicted)
    ok_gold, gold_rows = execute_query(db_id, rec["output"])
    exec_match = ok_pred and ok_gold and set(pred_rows) == set(gold_rows)

    db_path = DB_DIR / db_id / f"{db_id}.sqlite"
    db_exists = db_path.exists()

    print(f"--- Example {i+1} (db={db_id}, db_exists={db_exists}) ---")
    print(f"GOLD:      {rec['output']}")
    print(f"PREDICTED: {predicted[:300]}")
    print(f"exact_match={em}  exec_ok_pred={ok_pred}  exec_ok_gold={ok_gold}  exec_match={exec_match}")
    if not ok_pred:
        # try executing to get error
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute(predicted)
        except Exception as e:
            print(f"  pred error: {e}")
    print()
