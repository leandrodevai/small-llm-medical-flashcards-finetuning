"""LoRA/QLoRA training entry points."""

from __future__ import annotations

import os
from pathlib import Path

from medical_flashcards.data import load_prepared_dataset
from medical_flashcards.models import load_causal_lm


def train_lora(config: dict) -> dict:
    """Train a LoRA adapter from a project configuration.

    Args:
        config: Nested training config loaded from YAML or W&B.

    Returns:
        Evaluation metrics from the final trainer evaluation.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import EarlyStoppingCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    training_cfg = config["training"]
    project_cfg = config["project"]

    seed = int(dataset_cfg.get("seed", 42))
    set_seed(seed)
    sample_sizes = dataset_cfg.get("sample_sizes", {})

    dataset = load_prepared_dataset(
        dataset_id=dataset_cfg["dataset_id"],
        revision=dataset_cfg.get("revision"),
        seed=seed,
        train_size=sample_sizes.get("train"),
        validation_size=sample_sizes.get("validation"),
        test_size=sample_sizes.get("test"),
        system_prompt=dataset_cfg["system_prompt"],
    )

    model, tokenizer = load_causal_lm(
        model_cfg["id"],
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        prepare_for_kbit=bool(model_cfg.get("load_in_4bit", True)),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )

    lora_r = int(lora_cfg["r"])
    lora_alpha = resolve_lora_alpha(lora_cfg)
    lora_cfg["alpha"] = lora_alpha
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=float(lora_cfg["dropout"]),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get("target_modules", "all-linear"),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    run_name = build_run_name(config)
    output_dir = Path(project_cfg["output_dir"]) / run_name
    callbacks = []
    patience = training_cfg.get("early_stopping_patience")
    if patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(patience)))

    sft_config = {
        "output_dir": str(output_dir),
        "completion_only_loss": bool(training_cfg.get("completion_only_loss", True)),
        "num_train_epochs": float(training_cfg["num_train_epochs"]),
        "per_device_train_batch_size": int(training_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training_cfg["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(training_cfg["gradient_accumulation_steps"]),
        "learning_rate": float(training_cfg["learning_rate"]),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.0)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
        "lr_scheduler_type": training_cfg.get("lr_scheduler_type", "cosine"),
        "logging_steps": int(training_cfg.get("logging_steps", 10)),
        "eval_strategy": "steps",
        "eval_steps": int(training_cfg["eval_steps"]),
        "save_steps": int(training_cfg["save_steps"]),
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "optim": training_cfg.get("optim", "paged_adamw_8bit"),
        "packing": bool(training_cfg.get("packing", False)),
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "report_to": ["wandb"] if _wandb_enabled() else [],
        "load_best_model_at_end": bool(
            training_cfg.get("load_best_model_at_end", True)
        ),
        "run_name": run_name,
    }
    if training_cfg.get("metric_for_best_model"):
        sft_config["metric_for_best_model"] = training_cfg["metric_for_best_model"]
    if "greater_is_better" in training_cfg:
        sft_config["greater_is_better"] = bool(training_cfg["greater_is_better"])

    args = SFTConfig(**sft_config)

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir / "final_adapter"))
    return {key: float(value) for key, value in metrics.items()}


def build_run_name(config: dict) -> str:
    """Build a stable run name from model and LoRA settings."""
    model_name = config["model"]["id"].replace("/", "_")
    return (
        f"{config['project'].get('run_prefix', 'qlora')}"
        f"_lr{config['training']['learning_rate']}"
        f"_r{config['lora']['r']}_{model_name}"
    )


def resolve_lora_alpha(lora_cfg: dict) -> int:
    """Resolve LoRA alpha, using 2x rank for `None` or `auto`."""
    alpha = lora_cfg.get("alpha")
    if alpha is None or alpha == "auto":
        return 2 * int(lora_cfg["r"])
    return int(alpha)


def _wandb_enabled() -> bool:
    """Return whether training should report metrics to W&B."""
    return bool(os.getenv("WANDB_API_KEY")) and os.getenv("WANDB_DISABLED") != "true"
