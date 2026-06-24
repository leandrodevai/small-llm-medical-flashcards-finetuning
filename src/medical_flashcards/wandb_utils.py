"""Weights & Biases configuration helpers."""

from __future__ import annotations

from typing import Any

from medical_flashcards.config import apply_dotted_overrides, flatten_dict


def merge_wandb_config(base_config: dict[str, Any], wandb_config) -> dict[str, Any]:
    """Apply W&B sweep overrides to a nested project config."""
    raw = dict(wandb_config)
    flat_base = flatten_dict(base_config)
    overrides = {
        key: value
        for key, value in raw.items()
        if key in flat_base and flat_base[key] != value
    }
    return apply_dotted_overrides(base_config, overrides)
