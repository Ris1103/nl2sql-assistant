"""QLoRA fine-tuning with Unsloth + TRL SFTTrainer. Run this in WSL2."""
import json
from pathlib import Path

# unsloth must be imported before trl/transformers to apply its optimizations
import unsloth  # noqa: F401
import mlflow
import yaml
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

from src.training.callbacks import MLflowTrainingCallback
from src.training.config import TrainingConfig


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "training_config.yaml"


def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def format_prompt(example: dict) -> dict:
    text = (
        f"<|user|>\n{example['instruction']}\n\n{example['input']}<|end|>\n"
        f"<|assistant|>\n{example['output']}<|end|>"
    )
    return {"text": text}


def train(config_path: str = str(CONFIG_PATH)):
    # Import Unsloth here — only available in WSL2/Linux
    from unsloth import FastLanguageModel  # type: ignore

    cfg = TrainingConfig.from_yaml(config_path)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    print(f"Loading model: {cfg.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        use_gradient_checkpointing=cfg.gradient_checkpointing,
    )

    train_ds = load_jsonl(cfg.train_file).map(format_prompt)
    val_ds = load_jsonl(cfg.val_file).map(format_prompt)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.eval_steps,  # must be a multiple of eval_steps when load_best_model_at_end=True
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    with mlflow.start_run(run_name="phi3-qlora-nl2sql"):
        # Log all hyperparams
        mlflow.log_params({
            "model": cfg.model,
            "lora_r": cfg.lora.r,
            "lora_alpha": cfg.lora.lora_alpha,
            "epochs": cfg.num_train_epochs,
            "batch_size": cfg.per_device_train_batch_size,
            "grad_accum": cfg.gradient_accumulation_steps,
            "learning_rate": cfg.learning_rate,
            "optimizer": cfg.optim,
        })

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_length,
            args=training_args,
            callbacks=[MLflowTrainingCallback()],
        )

        print("Starting training...")
        trainer_result = trainer.train()

        mlflow.log_metrics({
            "train_loss": trainer_result.training_loss,
            "train_runtime_hours": trainer_result.metrics.get("train_runtime", 0) / 3600,
        })

        # Save adapter
        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        mlflow.log_artifacts(cfg.output_dir, artifact_path="adapter")
        print(f"Adapter saved to {cfg.output_dir}")


if __name__ == "__main__":
    train()
