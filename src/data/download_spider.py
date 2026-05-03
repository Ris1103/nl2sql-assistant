"""Download Spider NL2SQL dataset from HuggingFace and save raw splits."""
import json
import os
from pathlib import Path

from datasets import load_dataset


RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "spider"


def download():
    print("Downloading Spider dataset from HuggingFace...")
    dataset = load_dataset("spider")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation"):
        out_path = RAW_DIR / f"{split}.jsonl"
        records = list(dataset[split])
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"  Saved {len(records)} records → {out_path}")

    # Save tables metadata
    tables_path = RAW_DIR / "tables.json"
    if hasattr(dataset["train"], "info") and dataset["train"].info.dataset_name:
        pass  # tables.json comes from the raw HF cache; copy if available

    print("Done. Raw Spider data in:", RAW_DIR)


if __name__ == "__main__":
    download()
