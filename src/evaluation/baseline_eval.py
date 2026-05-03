"""Zero-shot baseline evaluation of Phi-3 mini on Spider validation set, logged to MLflow."""
import json
import time
from pathlib import Path

import mlflow
import requests

from src.evaluation.metrics import compute_metrics


OLLAMA_URL = "http://localhost:11434"
MODEL = "phi3:mini"
VAL_FILE = Path(__file__).parents[2] / "data" / "processed" / "val_instruct.jsonl"
MLFLOW_TRACKING_URI = str(Path(__file__).parents[2] / "mlruns")
EXPERIMENT_NAME = "nl2sql-baseline"
MAX_SAMPLES = None  # set to int to limit for quick runs


def generate_sql(instruction: str, input_text: str) -> tuple[str, float]:
    prompt = f"{instruction}\n\n{input_text}\n\nSQL:"
    start = time.time()
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        timeout=60,
    )
    response.raise_for_status()
    elapsed = (time.time() - start) * 1000
    return response.json()["response"].strip(), elapsed


def load_val_data(max_samples: int | None = None) -> list[dict]:
    records = []
    with open(VAL_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if max_samples and len(records) >= max_samples:
                break
    return records


def run_baseline():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    records = load_val_data(MAX_SAMPLES)
    print(f"Evaluating on {len(records)} examples with {MODEL}...")

    predictions = []
    latencies = []

    with mlflow.start_run(run_name="zero-shot-phi3-mini"):
        mlflow.log_param("model", MODEL)
        mlflow.log_param("n_examples", len(records))
        mlflow.log_param("temperature", 0)

        for i, rec in enumerate(records):
            predicted, latency_ms = generate_sql(rec["instruction"], rec["input"])
            latencies.append(latency_ms)

            # Extract db_id and gold SQL from the input/output fields
            db_id = rec["input"].split("Database:")[1].split("\n")[0].strip() if "Database:" in rec["input"] else "unknown"
            predictions.append({
                "predicted": predicted,
                "gold": rec["output"],
                "db_id": db_id,
            })

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(records)}")

        metrics = compute_metrics(predictions)
        metrics["avg_latency_ms"] = sum(latencies) / len(latencies)

        mlflow.log_metrics(metrics)
        print("\nBaseline Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    run_baseline()
