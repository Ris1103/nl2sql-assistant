"""Convert raw Spider JSONL splits into instruction-tuning format."""
import json
from pathlib import Path


RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "spider"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

SYSTEM_PROMPT = (
    "You are an expert SQL assistant. "
    "Given a database schema and a natural language question, generate a correct SQL query."
)


def format_schema(db_id: str, tables: list[dict]) -> str:
    """Build a compact schema string from Spider tables metadata."""
    lines = [f"Database: {db_id}"]
    for tbl in tables:
        col_strs = []
        for i, col_name in enumerate(tbl["column_names_original"]):
            if tbl["column_names"][i][0] == -1:
                continue
            col_type = tbl["column_types"][i] if i < len(tbl["column_types"]) else "text"
            col_strs.append(f"{col_name} ({col_type})")
        lines.append(f"Table: {tbl['table_names_original'][0] if tbl['table_names_original'] else 'unknown'}")
        lines.append("  Columns: " + ", ".join(col_strs))
    return "\n".join(lines)


def build_instruction_record(record: dict) -> dict:
    schema_text = f"Database: {record['db_id']}"
    if record.get("db_table_names"):
        for tbl in record["db_table_names"]:
            schema_text += f"\nTable: {tbl}"

    input_text = f"Schema:\n{schema_text}\n\nQuestion: {record['question']}"
    return {
        "instruction": SYSTEM_PROMPT,
        "input": input_text,
        "output": record["query"],
    }


def process_split(split_name: str) -> int:
    src = RAW_DIR / f"{split_name}.jsonl"
    out_name = "val_instruct.jsonl" if split_name == "validation" else "train_instruct.jsonl"
    dst = PROCESSED_DIR / out_name

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            instruct_rec = build_instruction_record(rec)
            fout.write(json.dumps(instruct_rec) + "\n")
            count += 1

    print(f"  {split_name}: {count} records → {dst}")
    return count


def main():
    print("Preprocessing Spider dataset...")
    for split in ("train", "validation"):
        process_split(split)
    print("Done.")


if __name__ == "__main__":
    main()
