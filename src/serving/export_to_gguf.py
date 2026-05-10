"""Merge LoRA adapter into base model, export to GGUF, and register with Ollama.

Run this in WSL2 after training completes:
  python -m src.serving.export_to_gguf

Requires llama.cpp cloned to ~/llama.cpp and its Python deps installed.
"""
import os
import subprocess
import sys
from pathlib import Path


ADAPTER_DIR = Path(__file__).parents[2] / "adapters" / "phi3-nl2sql-v1"
MERGED_DIR = Path(__file__).parents[2] / "adapters" / "phi3-nl2sql-merged"
GGUF_DIR = Path(__file__).parents[2] / "adapters" / "gguf"
LLAMA_CPP_DIR = Path.home() / "llama.cpp"
OLLAMA_MODEL_NAME = "phi3-nl2sql"

MODELFILE_TEMPLATE = """\
FROM {gguf_path}
SYSTEM "You are an expert SQL assistant. Given a database schema and a natural language question, generate a correct SQL query. Output only the SQL with no explanation."
PARAMETER temperature 0
PARAMETER stop "<|end|>"
"""


def merge_adapter():
    """Merge LoRA weights into base model using Unsloth."""
    from unsloth import FastLanguageModel  # type: ignore

    print(f"Loading adapter from {ADAPTER_DIR}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER_DIR),
        max_seq_length=2048,
        load_in_4bit=True,
    )
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Merging and saving to {MERGED_DIR}...")
    model.save_pretrained_merged(str(MERGED_DIR), tokenizer, save_method="merged_16bit")
    print("Merge complete.")


def convert_to_gguf():
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        convert_script = LLAMA_CPP_DIR / "convert.py"

    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    gguf_f16 = GGUF_DIR / "phi3-nl2sql-f16.gguf"

    print("Converting to GGUF f16...")
    subprocess.run(
        [sys.executable, str(convert_script), str(MERGED_DIR), "--outfile", str(gguf_f16), "--outtype", "f16"],
        check=True,
    )

    # Quantize to Q4_K_M
    gguf_q4 = GGUF_DIR / "phi3-nl2sql-q4_k_m.gguf"
    # llama.cpp renamed the binary to llama-quantize in newer versions
    quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "quantize"
    if not quantize_bin.exists():
        quantize_bin = LLAMA_CPP_DIR / "quantize"

    print("Quantizing to Q4_K_M...")
    subprocess.run([str(quantize_bin), str(gguf_f16), str(gguf_q4), "Q4_K_M"], check=True)
    print(f"GGUF saved: {gguf_q4}")
    return gguf_q4


def create_ollama_model(gguf_path: Path):
    modelfile_path = GGUF_DIR / "Modelfile"
    modelfile_path.write_text(MODELFILE_TEMPLATE.format(gguf_path=str(gguf_path)))
    print(f"Modelfile written to {modelfile_path}")
    print(f"Creating Ollama model '{OLLAMA_MODEL_NAME}'...")
    subprocess.run(["ollama", "create", OLLAMA_MODEL_NAME, "-f", str(modelfile_path)], check=True)
    print(f"Done. Test with: ollama run {OLLAMA_MODEL_NAME}")


if __name__ == "__main__":
    merge_adapter()
    gguf_path = convert_to_gguf()
    create_ollama_model(gguf_path)
