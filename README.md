# NL2SQL Assistant

Full lifecycle LLM engineering project: fine-tuned Phi-3 mini 3.8B + QLoRA + RAG schema retrieval + MLflow LLMOps.

## Stack

| Layer | Technology |
|-------|-----------|
| Base model | Phi-3 mini 3.8B (`unsloth/Phi-3-mini-4k-instruct-bnb-4bit`) |
| Fine-tuning | QLoRA via Unsloth + TRL SFTTrainer (WSL2) |
| Embeddings | nomic-embed-text via Ollama |
| Vector store | ChromaDB (cosine similarity) |
| Benchmark | Spider NL2SQL (7K train / 1,034 val) |
| LLMOps | MLflow (experiments, metrics, artifacts) |
| Serving | FastAPI + Ollama (GGUF Q4_K_M) |
| Frontend | Streamlit |
| Report | ReportLab PDF |

## Hardware Requirements

- GPU: GTX 1650 4GB VRAM (or equivalent)
- RAM: 16GB
- Training must run in **WSL2** — Unsloth's Triton kernels require Linux

## Quick Start

### Phase 1 — Data Pipeline

```bash
# In WSL2
pip install -r requirements-train.txt
python -m src.data.download_spider       # ~7K train + 1,034 val examples
python -m src.data.preprocess            # → data/processed/*.jsonl
python -m src.data.schema_extractor      # → data/processed/schema_docs.jsonl
```

### Phase 2 — Baseline Evaluation

```powershell
# Windows PowerShell (after: ollama pull phi3:mini && ollama pull nomic-embed-text)
pip install -r requirements.txt
python -m src.rag.vector_store           # Build ChromaDB collection (~200 schema docs)
python -m src.evaluation.baseline_eval  # Logs to MLflow: nl2sql-baseline
```

### Phase 3 — QLoRA Fine-tuning

```bash
# WSL2 only
python -m src.training.trainer           # ~6-8 hrs, peak VRAM ~3.5GB
# Adapter saved to: adapters/phi3-nl2sql-v1/
```

### Phase 4 — Export to Ollama

```bash
# WSL2
python -m src.serving.export_to_gguf
# Creates: ollama model phi3-nl2sql (Q4_K_M)
```

### Phase 5 — Post-training Evaluation

```powershell
python -m src.evaluation.finetuned_eval  # Logs to MLflow: nl2sql-finetuning
```

### Phase 6 — Serve

```powershell
# Terminal 1
ollama serve
# Terminal 2
mlflow ui --backend-store-uri ./mlruns --port 5000
# Terminal 3
uvicorn src.serving.api:app --port 8000 --reload
# Terminal 4
streamlit run src/serving/ui.py --server.port 8501
```

### Phase 7 — Generate Report

```powershell
python -m src.reporting.generate_report --output nl2sql_project_report.pdf
```

## Service URLs

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| MLflow UI | http://localhost:5000 |
| Ollama | http://localhost:11434 |

## Expected Results

| Metric | Baseline (zero-shot) | Fine-tuned + RAG |
|--------|---------------------|------------------|
| Exact Match Accuracy | ~50% | ~70–75% |
| Execution Accuracy | ~55% | ~72–78% |

## Training Notes

- `bf16: false` and `fp16: false` — GTX 1650 Turing arch does not support bf16; Unsloth handles quantization internally
- Effective batch size: 4 (batch_size=1 × grad_accum=4)
- LoRA rank: 8, alpha: 16, targeting all attention + MLP projection layers
- Optimizer: `adamw_8bit` for VRAM efficiency
