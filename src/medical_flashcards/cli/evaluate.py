"""Evaluate one or more fine-tuned models on a fixed split."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from dotenv import load_dotenv

from medical_flashcards.config import load_yaml
from medical_flashcards.data import load_fixed_dataset, prepare_split
from medical_flashcards.evaluate import (
    add_similarity_and_summarize,
    evaluate_model,
    write_predictions_and_leaderboard,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for model evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--model-id", action="append", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output-dir", default="outputs/finetuning_evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--skip-reference-logprobs", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate requested models and update the leaderboard."""
    load_dotenv()
    args = parse_args(argv)
    dataset_cfg = load_yaml(args.dataset_config)
    dataset = load_fixed_dataset(
        dataset_id=dataset_cfg["dataset_id"], revision=dataset_cfg.get("revision")
    )
    eval_dataset = prepare_split(
        dataset[args.split],
        seed=dataset_cfg.get("seed", 42),
        size=args.eval_size,
        system_prompt=dataset_cfg["system_prompt"],
    )

    for model_id in args.model_id:
        print(f"Evaluating {model_id}")
        predictions, telemetry = evaluate_model(
            model_id,
            eval_dataset,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            load_in_4bit=args.load_in_4bit,
            score_reference=not args.skip_reference_logprobs,
        )
        predictions, metrics = add_similarity_and_summarize(
            predictions,
            telemetry,
            embedding_model_id=dataset_cfg["embedding_model_id"],
        )
        write_predictions_and_leaderboard(args.output_dir, model_id, predictions, metrics)
        print(
            f'  token_f1={metrics["token_f1_mean"]:.3f} '
            f'answer_similarity={metrics["answer_similarity_mean"]:.3f}'
        )
