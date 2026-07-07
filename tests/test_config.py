from medical_flashcards.config import (
    apply_dotted_overrides,
    flatten_dict,
    resolve_dataset_config,
)
from medical_flashcards.train import resolve_lora_alpha


def test_flatten_dict_uses_dotted_keys():
    assert flatten_dict({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}


def test_apply_dotted_overrides_updates_nested_values():
    config = {"training": {"learning_rate": 0.001}}
    updated = apply_dotted_overrides(config, {"training.learning_rate": 0.0002})
    assert updated["training"]["learning_rate"] == 0.0002
    assert config["training"]["learning_rate"] == 0.001


def test_resolve_lora_alpha_defaults_to_twice_rank():
    assert resolve_lora_alpha({"r": 16, "alpha": None}) == 32
    assert resolve_lora_alpha({"r": 8, "alpha": "auto"}) == 16
    assert resolve_lora_alpha({"r": 8, "alpha": 32}) == 32


def test_resolve_dataset_config_loads_referenced_file(tmp_path):
    dataset_config = tmp_path / "dataset.yaml"
    dataset_config.write_text(
        "\n".join(
            [
                "dataset_id: fixed-dataset",
                "system_prompt: prompt",
                "sample_sizes:",
                "  train: 4096",
                "  validation: 512",
            ]
        ),
        encoding="utf-8",
    )

    resolved = resolve_dataset_config(
        {"dataset": {"config_path": "dataset.yaml"}}, base_dir=tmp_path
    )

    assert resolved["dataset"]["dataset_id"] == "fixed-dataset"
    assert resolved["dataset"]["system_prompt"] == "prompt"
    assert resolved["dataset"]["sample_sizes"]["train"] == 4096
    assert resolved["dataset"]["sample_sizes"]["validation"] == 512
