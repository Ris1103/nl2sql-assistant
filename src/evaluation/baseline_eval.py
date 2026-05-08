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
MLFLOW_TRACKING_URI = (Path(__file__).parents[2] / "mlruns").as_uri()  # as_uri() avoids Windows drive letter being misread as URI scheme
EXPERIMENT_NAME = "nl2sql-baseline"
MAX_SAMPLES = 100  # set to None to run the full 1,034-example val set


def generate_sql(instruction: str, input_text: str, timeout: int = 300) -> tuple[str, float]:
    prompt = f"{instruction}\n\n{input_text}\n\nSQL:"
    start = time.time()
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        # num_gpu=20 offloads 20 layers to GTX 1650 (4GB VRAM); num_ctx=2048 keeps KV cache small enough to fit
        json={"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0, "num_gpu": 20, "num_ctx": 2048}},
        timeout=timeout,  # cold model load on first call can exceed 60s
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
            try:
                predicted, latency_ms = generate_sql(rec["instruction"], rec["input"])
            except Exception as e:
                print(f"  WARNING: skipping example {i}: {e}")
                predicted, latency_ms = "", 0.0
            latencies.append(latency_ms)

            # Extract db_id and gold SQL from the input/output fields
            db_id = rec["input"].split("Database:")[1].split("\n")[0].strip() if "Database:" in rec["input"] else "unknown"
            predictions.append({
                "predicted": predicted,
                "gold": rec["output"],
                "db_id": db_id,
            })

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(records)}", flush=True)

        metrics = compute_metrics(predictions)
        metrics["avg_latency_ms"] = sum(latencies) / len(latencies)

        mlflow.log_metrics(metrics)
        print("\nBaseline Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    run_baseline()
