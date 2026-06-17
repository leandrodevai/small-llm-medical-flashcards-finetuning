"""Dataset loading and preparation utilities."""

from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset

from medical_flashcards.prompts import to_chat_example


def load_fixed_dataset(
    dataset_id: str,
    revision: str | None = None,
) -> DatasetDict:
    """Load the fixed Hugging Face dataset split used by this project."""
    kwargs = {"revision": revision} if revision else {}
    return load_dataset(dataset_id, **kwargs)


def prepare_dataset(
    dataset: DatasetDict,
    *,
    system_prompt: str,
    seed: int = 42,
    train_size: int | None = None,
    validation_size: int | None = None,
    test_size: int | None = None,
) -> DatasetDict:
    """Prepare all dataset splits for SFT.

    Args:
        dataset: Raw Hugging Face dataset with train/validation/test splits.
        seed: Seed used when subsampling a split.
        train_size: Optional cap for the train split.
        validation_size: Optional cap for the validation split.
        test_size: Optional cap for the test split.
        system_prompt: System prompt prepended to each instruction.

    Returns:
        DatasetDict with chat-style `prompt` and `completion` columns.
    """
    sizes = {
        "train": train_size,
        "validation": validation_size,
        "test": test_size,
    }

    prepared = {}
    for split_name, split in dataset.items():
        prepared[split_name] = prepare_split(
            split,
            size=sizes.get(split_name),
            seed=seed,
            system_prompt=system_prompt,
        )
    return DatasetDict(prepared)


def prepare_split(
    split: Dataset,
    *,
    system_prompt: str,
    size: int | None = None,
    seed: int = 42,
) -> Dataset:
    """Subsample and convert one split to chat format."""
    selected = _select_size(split, size, seed)
    return _to_chat_dataset(selected, system_prompt)


def load_prepared_dataset(
    *,
    dataset_id: str,
    system_prompt: str,
    revision: str | None = None,
    seed: int = 42,
    train_size: int | None = None,
    validation_size: int | None = None,
    test_size: int | None = None,
) -> DatasetDict:
    """Load the fixed dataset and prepare it for training or evaluation."""
    dataset = load_fixed_dataset(dataset_id=dataset_id, revision=revision)
    return prepare_dataset(
        dataset,
        seed=seed,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        system_prompt=system_prompt,
    )


def _select_size(split: Dataset, size: int | None, seed: int) -> Dataset:
    """Return a shuffled subset when a size cap is configured."""
    if size is None:
        return split
    return split.shuffle(seed=seed).select(range(min(size, len(split))))


def _to_chat_dataset(split: Dataset, system_prompt: str) -> Dataset:
    """Convert raw flashcard columns to chat columns when needed."""
    if {"prompt", "completion"}.issubset(split.column_names):
        return split
    return split.map(
        lambda example: to_chat_example(example, system_prompt),
        remove_columns=split.column_names,
    )
