"""Configuration loading and override helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary using dotted keys."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        dotted_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, dotted_key))
        else:
            flat[dotted_key] = value
    return flat


def apply_dotted_overrides(
    config: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Return a copy of the config with dotted-key overrides applied.

    Args:
        config: Base nested configuration.
        overrides: Mapping such as {"training.learning_rate": 0.0001}.

    Returns:
        A new config dictionary. The input config is not mutated.
    """
    updated = deepcopy(config)
    for dotted_key, value in overrides.items():
        if value is None or "." not in dotted_key:
            continue

        target = updated
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return updated


def resolve_dataset_config(
    config: dict[str, Any], *, base_dir: str | Path
) -> dict[str, Any]:
    """Resolve the dataset section from an inline config or external file.

    Args:
        config: Full training config. The `dataset` section may include
            `config_path` plus inline overrides.
        base_dir: Directory used to resolve relative `config_path` values.

    Returns:
        A copied config with the referenced dataset config merged in.
    """
    updated = deepcopy(config)
    dataset_cfg = updated.get("dataset", {})
    config_path = dataset_cfg.get("config_path")

    if config_path:
        dataset_path = Path(config_path)
        if not dataset_path.is_absolute():
            dataset_path = Path(base_dir) / dataset_path
        base_dataset_cfg = load_yaml(dataset_path)
        overrides = {
            key: value for key, value in dataset_cfg.items() if key != "config_path"
        }
        dataset_cfg = _deep_merge(base_dataset_cfg, overrides)

    updated["dataset"] = dataset_cfg
    return updated


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dictionaries, with override values taking precedence."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
