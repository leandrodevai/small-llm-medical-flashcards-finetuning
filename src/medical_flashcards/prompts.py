"""Prompt and chat-template helpers for the flashcard task."""

from __future__ import annotations


def to_chat_example(example: dict, system_prompt: str) -> dict:
    """Convert a raw flashcard row into a prompt-completion conversational style."""
    return {
        "prompt": [
            {
                "role": "system",
                "content": system_prompt + example["instruction"],
            },
            {
                "role": "user",
                "content": example["input"],
            },
        ],
        "completion": [
            {
                "role": "assistant",
                "content": example["output"],
            }
        ],
    }


def get_reference(example: dict) -> dict[str, str]:
    """Extract the normalized reference answer from a chat example."""
    from medical_flashcards.metrics import clean_answer

    completion = example["completion"][0]["content"]
    return {"reference_answer": clean_answer(completion)}


def get_question(example: dict) -> str:
    """Return the user question from a chat example."""
    for message in example["prompt"]:
        if message["role"] == "user":
            return message["content"]
    return ""


def build_generation_prompt(tokenizer, example: dict) -> str:
    """Render a chat example as a generation prompt for a tokenizer."""
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "add_special_tokens": False,
    }
    try:
        return tokenizer.apply_chat_template(
            example["prompt"], enable_thinking=False, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(example["prompt"], **kwargs)
