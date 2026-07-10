"""Evaluation loop and prediction artifact helpers."""

from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from medical_flashcards.metrics import (
    add_answer_similarity,
    clean_answer,
    summarize_predictions,
    token_f1,
)
from medical_flashcards.models import (
    cuda_peak_memory_mb,
    load_causal_lm,
    release_model,
    reset_cuda_peak_memory,
    synchronize_cuda,
)
from medical_flashcards.prompts import (
    build_generation_prompt,
    get_question,
    get_reference,
)


def evaluate_model(
    model_id: str,
    eval_dataset,
    *,
    batch_size: int = 32,
    max_new_tokens: int = 512,
    load_in_4bit: bool = False,
    score_reference: bool = True,
    trust_remote_code: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Generate predictions and telemetry for one model.

    Args:
        model_id: Hugging Face model identifier or local model path.
        eval_dataset: Prepared dataset split with `prompt` and `completion` columns.
        batch_size: Number of examples per generation batch.
        max_new_tokens: Generation cap for each answer.
        load_in_4bit: Load the model with NF4 quantization.
        score_reference: Compute log-probability metrics for reference answers.
        trust_remote_code: Forwarded to Transformers loaders.

    Returns:
        Predictions DataFrame and telemetry dictionary.
    """
    model, tokenizer = load_causal_lm(
        model_id,
        load_in_4bit=load_in_4bit,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    rows = []
    batch_stats = []
    reference_logprob_seconds = float("nan")
    reset_cuda_peak_memory()

    try:
        for start in tqdm(range(0, len(eval_dataset), batch_size), desc=model_id):
            examples = [
                eval_dataset[i]
                for i in range(start, min(start + batch_size, len(eval_dataset)))
            ]
            generations, stats = generate_batch(
                model, tokenizer, examples, max_new_tokens=max_new_tokens
            )
            batch_stats.append(stats)
            for index, (example, generation, generated_tokens) in enumerate(
                zip(examples, generations, stats["generated_tokens_per_example"]),
                start=start,
            ):
                reference = get_reference(example)
                generated_answer = clean_answer(generation)
                reference_answer = reference["reference_answer"]
                rows.append(
                    {
                        "model_id": model_id,
                        "example_index": index,
                        "question": get_question(example),
                        **reference,
                        "generated_answer": generated_answer,
                        "raw_generation": generation,
                        "generated_tokens": generated_tokens,
                        "token_f1": token_f1(generated_answer, reference_answer),
                    }
                )

        predictions = pd.DataFrame(rows)
        if score_reference:
            reference_scores, reference_logprob_seconds = score_reference_answers(
                model, tokenizer, eval_dataset
            )
            predictions = predictions.merge(
                reference_scores, on="example_index", how="left"
            )
    finally:
        memory_stats = cuda_peak_memory_mb()
        release_model(model, tokenizer)

    total_generation_seconds = sum(s["inference_seconds"] for s in batch_stats)
    total_generation_tokens = sum(s["generated_tokens"] for s in batch_stats)
    total_inference_seconds = total_generation_seconds + (
        reference_logprob_seconds if math.isfinite(reference_logprob_seconds) else 0.0
    )
    telemetry = {
        "total_inference_seconds": total_inference_seconds,
        "generation_inference_seconds": total_generation_seconds,
        "generation_tokens": total_generation_tokens,
        "generation_tokens_per_second": total_generation_tokens
        / total_generation_seconds
        if total_generation_seconds > 0
        else float("nan"),
        "reference_logprob_inference_seconds": reference_logprob_seconds,
        "reference_logprob_examples_per_second": len(eval_dataset)
        / reference_logprob_seconds
        if reference_logprob_seconds > 0
        else float("nan"),
        **memory_stats,
    }
    return predictions, telemetry


def generate_batch(model, tokenizer, examples: list[dict], *, max_new_tokens: int):
    """Generate answers for a batch of prepared examples."""
    import torch

    prompts = [build_generation_prompt(tokenizer, example) for example in examples]
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, add_special_tokens=True
    ).to(model.device)

    synchronize_cuda()
    start_time = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    synchronize_cuda()
    inference_seconds = time.perf_counter() - start_time

    new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
    generated_tokens_per_example = [
        _generated_token_count(row, tokenizer) for row in new_tokens
    ]
    generated_token_count = sum(generated_tokens_per_example)
    stats = {
        "generated_tokens": generated_token_count,
        "inference_seconds": inference_seconds,
        "tokens_per_second": generated_token_count / inference_seconds
        if inference_seconds > 0
        else float("nan"),
        "generated_tokens_per_example": generated_tokens_per_example,
    }
    generations = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return generations, stats


def reference_answer_logprobs(model, tokenizer, example: dict) -> dict:
    """Score the reference answer under the model."""
    import torch

    prompt = build_generation_prompt(tokenizer, example)
    reference_answer = get_reference(example)["reference_answer"]
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=True
    ).input_ids[0]
    target_ids = tokenizer(
        reference_answer, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]

    if target_ids.numel() == 0:
        return {
            "reference_answer_logprob_sum": float("nan"),
            "reference_answer_nll_mean": float("nan"),
            "reference_answer_token_perplexity": float("nan"),
            "reference_answer_tokens": 0,
        }

    input_ids = torch.cat([prompt_ids, target_ids]).unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(input_ids).to(model.device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    prompt_length = prompt_ids.shape[0]
    target_logits = logits[
        0, prompt_length - 1 : prompt_length - 1 + target_ids.shape[0], :
    ]
    target_ids = target_ids.to(model.device)
    log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)
    token_logprobs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    logprob_sum = float(token_logprobs.sum().item())
    nll_mean = float(-token_logprobs.mean().item())

    return {
        "reference_answer_logprob_sum": logprob_sum,
        "reference_answer_nll_mean": nll_mean,
        "reference_answer_token_perplexity": math.exp(nll_mean)
        if math.isfinite(nll_mean)
        else float("nan"),
        "reference_answer_tokens": int(target_ids.shape[0]),
    }


def score_reference_answers(
    model, tokenizer, eval_dataset
) -> tuple[pd.DataFrame, float]:
    """Compute reference-answer scores for a full evaluation split."""
    rows = []
    synchronize_cuda()
    start_time = time.perf_counter()
    for index, example in enumerate(
        tqdm(eval_dataset, desc="reference logprob", leave=False)
    ):
        rows.append(
            {
                "example_index": index,
                **reference_answer_logprobs(model, tokenizer, example),
            }
        )
    synchronize_cuda()
    return pd.DataFrame(rows), time.perf_counter() - start_time


def add_similarity_and_summarize(
    predictions: pd.DataFrame,
    telemetry: dict,
    *,
    embedding_model_id: str,
) -> tuple[pd.DataFrame, dict]:
    """Add embedding similarity and summarize predictions for reporting."""
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(embedding_model_id, device="cpu")
    predictions = add_answer_similarity(predictions, embedder)
    return predictions, summarize_predictions(predictions, telemetry)


def write_predictions_and_leaderboard(
    output_dir: str | Path,
    model_id: str,
    predictions: pd.DataFrame,
    metrics: dict,
) -> None:
    """Write per-model predictions and update the output leaderboard."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = model_id.replace("/", "__") + ".csv"
    predictions.to_csv(output_path / filename, index=False)

    leaderboard_path = output_path / "leaderboard.csv"
    leaderboard = pd.read_csv(leaderboard_path) if leaderboard_path.exists() else None
    row = pd.DataFrame([metrics])
    leaderboard = row if leaderboard is None else pd.concat([leaderboard, row])
    leaderboard = (
        leaderboard.drop_duplicates(subset=["model_id"], keep="last")
        .sort_values(["answer_similarity_mean", "token_f1_mean"], ascending=False)
        .reset_index(drop=True)
    )
    leaderboard.to_csv(leaderboard_path, index=False)


def _generated_token_count(row, tokenizer) -> int:
    """Count generated tokens up to EOS or excluding padding."""
    if tokenizer.eos_token_id is not None:
        eos_positions = (row == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            return int(eos_positions[0].item()) + 1
    if tokenizer.pad_token_id is not None:
        return int(row.ne(tokenizer.pad_token_id).sum().item())
    return row.shape[0]
