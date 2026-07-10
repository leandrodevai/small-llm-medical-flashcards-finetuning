"""Run baseline evaluation for all configured models."""

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
    """Parse CLI arguments for baseline evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--skip-reference-logprobs", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate configured baseline models and update output artifacts."""
    load_dotenv()
    args = parse_args(argv)
    dataset_cfg = load_yaml(args.dataset_config)
    models_cfg = load_yaml(args.models_config)

    dataset = load_fixed_dataset(
        dataset_id=dataset_cfg["dataset_id"], revision=dataset_cfg.get("revision")
    )
    eval_dataset = prepare_split(
        dataset["validation"],
        seed=dataset_cfg.get("seed", 42),
        size=args.eval_size,
        system_prompt=dataset_cfg["system_prompt"],
    )

    for model_id in models_cfg["baseline_models"]:
        print(f"Evaluating {model_id}")
        try:
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
            write_predictions_and_leaderboard(
                args.output_dir, model_id, predictions, metrics
            )
            print(
                f'  token_f1={metrics["token_f1_mean"]:.3f} '
                f'answer_similarity={metrics["answer_similarity_mean"]:.3f}'
            )
        except Exception as exc:
            print(f"  skipped: {exc!r}")
