"""Train a LoRA run from YAML config and optional dotted overrides."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from medical_flashcards.config import (
    apply_dotted_overrides,
    flatten_dict,
    load_yaml,
    resolve_dataset_config,
)
from medical_flashcards.train import resolve_lora_alpha, train_lora
from medical_flashcards.wandb_utils import merge_wandb_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted override, e.g. training.learning_rate=0.0001",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None, *, base_dir: str | Path | None = None
) -> None:
    """Run training locally or through a W&B sweep agent."""
    args = parse_args(argv)
    config_base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
    config = load_yaml(args.config)
    config = apply_dotted_overrides(config, dict(parse_override(v) for v in args.override))
    config = resolve_dataset_config(config, base_dir=config_base_dir)
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    use_wandb = not args.no_wandb and bool(os.getenv("WANDB_API_KEY"))
    if use_wandb:
        import wandb

        with wandb.init(
            project=config["project"]["wandb_project"],
            config=flatten_dict(config),
        ):
            config = merge_wandb_config(config, wandb.config)
            config = resolve_dataset_config(config, base_dir=config_base_dir)
            wandb.config.update(
                {"lora.alpha": resolve_lora_alpha(config["lora"])},
                allow_val_change=True,
            )
            metrics = train_lora(config)
            wandb.log(metrics)
    else:
        metrics = train_lora(config)

    print(metrics)


def parse_override(raw: str) -> tuple[str, object]:
    """Parse a dotted CLI override into key and typed value."""
    key, value = raw.split("=", 1)
    return key, parse_scalar(value)


def parse_scalar(value: str) -> object:
    """Parse basic scalar values from CLI strings."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
