"""Create a W&B sweep from a YAML sweep config."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import wandb

from medical_flashcards.config import load_yaml


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for sweep creation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_config")
    parser.add_argument("--project", default="medical-flashcards-lora")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Create the sweep and print the W&B sweep ID."""
    args = parse_args(argv)
    sweep_id = wandb.sweep(load_yaml(args.sweep_config), project=args.project)
    print(sweep_id)
