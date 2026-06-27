"""Create a W&B sweep from a YAML sweep config."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import wandb

from medical_flashcards.config import load_yaml


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for sweep creation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_config")
    parser.add_argument("--project", default="medical-flashcards-lora")
    return parser.parse_args()


def main() -> None:
    """Create the sweep and print the W&B sweep ID."""
    args = parse_args()
    sweep_id = wandb.sweep(load_yaml(args.sweep_config), project=args.project)
    print(sweep_id)


if __name__ == "__main__":
    main()
