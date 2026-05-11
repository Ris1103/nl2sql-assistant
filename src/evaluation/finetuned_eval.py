"""Post-training evaluation of the fine-tuned phi3-nl2sql model via Ollama, logged to MLflow."""
import json
import time
from pathlib import Path

import mlflow
import requests

from src.evaluation.metrics import compute_metrics
from src.rag.retriever import build_schema_context


OLLAMA_URL = "http://localhost:11434"
MODEL = "phi3-nl2sql"  # Ollama model name after GGUF export
VAL_FILE = Path(__file__).parents[2] / "data" / "processed" / "val_instruct.jsonl"
MLFLOW_TRACKING_URI = (Path(__file__).parents[2] / "mlruns").as_uri()  # as_uri() avoids Windows drive letter being misread as URI scheme
EXPERIMENT_NAME = "nl2sql-finetuning"
USE_RAG = True
MAX_SAMPLES = 200


def generate_sql(instruction: str, input_text: str, use_rag: bool = True) -> tuple[str, float]:
    if use_rag:
        question = input_text.split("Question:")[-1].strip() if "Question:" in input_text else input_text
        db_id_line = input_text.split("Database:")[1].split("\n")[0].strip() if "Database:" in input_text else None
        schema_ctx = build_schema_context(question, db_id=db_id_line)
        enriched_input = f"Schema:\n{schema_ctx}\n\nQuestion: {question}"
    else:
        enriched_input = input_text

    prompt = f"{instruction}\n\n{enriched_input}\n\nSQL:"
    start = time.time()
    for attempt in range(3):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
                timeout=300,
            )
            response.raise_for_status()
            break
        except (requests.ReadTimeout, requests.HTTPError):
            if attempt == 2:
                raise
            time.sleep(3)
    elapsed = (time.time() - start) * 1000
    return response.json()["response"].strip(), elapsed


def run_eval():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    records = []
    with open(VAL_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if MAX_SAMPLES and len(records) >= MAX_SAMPLES:
                break

    print(f"Evaluating {MODEL} on {len(records)} examples (RAG={USE_RAG})...")

    predictions = []
    latencies = []

    with mlflow.start_run(run_name=f"finetuned-{'rag' if USE_RAG else 'no-rag'}"):
        mlflow.log_param("model", MODEL)
        mlflow.log_param("n_examples", len(records))
        mlflow.log_param("use_rag", USE_RAG)

        for i, rec in enumerate(records):
            predicted, latency_ms = generate_sql(rec["instruction"], rec["input"], USE_RAG)
            latencies.append(latency_ms)
            db_id = rec["input"].split("Database:")[1].split("\n")[0].strip() if "Database:" in rec["input"] else "unknown"
            predictions.append({"predicted": predicted, "gold": rec["output"], "db_id": db_id})
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(records)}")

        metrics = compute_metrics(predictions)
        metrics["avg_latency_ms"] = sum(latencies) / len(latencies)
        mlflow.log_metrics(metrics)

        print("\nFine-tuned Model Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    run_eval()
