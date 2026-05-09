"""Training configuration dataclass loaded from configs/training_config.yaml."""
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    model: str = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    output_dir: str = "./adapters/phi3-nl2sql-v1"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means train for full num_train_epochs
    warmup_steps: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    bf16: bool = False
    fp16: bool = False
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2

    train_file: str = "./data/processed/train_instruct.jsonl"
    val_file: str = "./data/processed/val_instruct.jsonl"

    mlflow_experiment: str = "nl2sql-finetuning"
    mlflow_tracking_uri: str = "./mlruns"

    lora: LoraConfig = field(default_factory=LoraConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        lora_raw = raw.pop("lora", {})
        training_raw = raw.pop("training", {})
        data_raw = raw.pop("data", {})
        mlflow_raw = raw.pop("mlflow", {})

        cfg = cls(
            model=raw.get("model", cls.model),
            max_seq_length=raw.get("max_seq_length", cls.max_seq_length),
            load_in_4bit=raw.get("load_in_4bit", cls.load_in_4bit),
            **{k: v for k, v in training_raw.items() if hasattr(cls, k)},
            train_file=data_raw.get("train_file", cls.train_file),
            val_file=data_raw.get("val_file", cls.val_file),
            mlflow_experiment=mlflow_raw.get("experiment_name", cls.mlflow_experiment),
            mlflow_tracking_uri=mlflow_raw.get("tracking_uri", cls.mlflow_tracking_uri),
        )
        cfg.lora = LoraConfig(**{k: v for k, v in lora_raw.items() if hasattr(LoraConfig, k)})
        return cfg
