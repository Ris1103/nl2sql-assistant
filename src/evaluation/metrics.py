"""Evaluation metrics: exact match accuracy and execution accuracy."""
import sqlite3
from pathlib import Path


DB_DIR = Path(__file__).parents[2] / "data" / "databases" / "spider_data" / "spider_data" / "database"


def exact_match(predicted: str, gold: str) -> bool:
    return predicted.strip().lower() == gold.strip().lower()


def execute_query(db_id: str, sql: str) -> tuple[bool, list | None]:
    """Execute sql on the Spider SQLite database. Returns (success, results)."""
    db_path = DB_DIR / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        # try flat layout
        db_path = DB_DIR / f"{db_id}.sqlite"
    if not db_path.exists():
        return False, None

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = [tuple(r) for r in cursor.fetchall()]
        conn.close()
        return True, rows
    except Exception:
        return False, None


def execution_accuracy(predicted: str, gold: str, db_id: str) -> bool:
    """Return True if predicted and gold SQL return the same result set."""
    ok_pred, pred_rows = execute_query(db_id, predicted)
    ok_gold, gold_rows = execute_query(db_id, gold)

    if not ok_pred or not ok_gold:
        return False

    return set(pred_rows) == set(gold_rows)


def compute_metrics(predictions: list[dict]) -> dict:
    """
    predictions: list of {predicted, gold, db_id}
    Returns dict with exact_match_accuracy and execution_accuracy.
    """
    em_total = ex_total = 0
    n = len(predictions)

    for p in predictions:
        if exact_match(p["predicted"], p["gold"]):
            em_total += 1
        if execution_accuracy(p["predicted"], p["gold"], p["db_id"]):
            ex_total += 1

    return {
        "exact_match_accuracy": em_total / n if n else 0.0,
        "execution_accuracy": ex_total / n if n else 0.0,
        "n_examples": n,
    }
