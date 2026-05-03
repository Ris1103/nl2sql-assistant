"""MLflow callback for per-step loss logging during SFT training."""
import mlflow
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class MLflowTrainingCallback(TrainerCallback):
    """Log train/eval loss to MLflow at every logging step."""

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        metrics = {}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                metrics[k] = v
        if metrics:
            mlflow.log_metrics(metrics, step=step)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics:
            step = state.global_step
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step)
