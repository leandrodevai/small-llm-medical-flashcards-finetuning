"""Lightweight text metrics used for model comparison."""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

ANSWER_PREFIX_RE = re.compile(r"^\s*(?:answer|assistant)\s*:\s*", flags=re.IGNORECASE)
ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def clean_answer(text: str | None) -> str:
    """Strip common assistant-style answer prefixes."""
    text = (text or "").strip()
    return ANSWER_PREFIX_RE.sub("", text).strip()


def normalize_answer(text: str | None) -> str:
    """Normalize text for token-overlap scoring."""
    text = (text or "").lower()
    text = ARTICLE_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def answer_tokens(text: str | None) -> list[str]:
    """Tokenize normalized answer text."""
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def token_f1(prediction: str | None, reference: str | None) -> float:
    """Compute token-level F1 after lightweight normalization."""
    prediction_tokens = answer_tokens(prediction)
    reference_tokens = answer_tokens(reference)
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def add_answer_similarity(predictions: pd.DataFrame, embedder) -> pd.DataFrame:
    """Add cosine similarity between generated and reference answers.

    Args:
        predictions: DataFrame with `generated_answer` and `reference_answer`.
        embedder: SentenceTransformer-like object with an `encode` method.

    Returns:
        A copy of `predictions` with an `answer_similarity` column.
    """
    predictions = predictions.copy()
    predictions["answer_similarity"] = float("nan")
    valid = (
        predictions["generated_answer"].notna()
        & predictions["reference_answer"].notna()
        & predictions["generated_answer"].str.len().gt(0)
        & predictions["reference_answer"].str.len().gt(0)
    )
    if valid.any():
        generated = embedder.encode(
            predictions.loc[valid, "generated_answer"].tolist(),
            normalize_embeddings=True,
        )
        reference = embedder.encode(
            predictions.loc[valid, "reference_answer"].tolist(),
            normalize_embeddings=True,
        )
        predictions.loc[valid, "answer_similarity"] = (generated * reference).sum(
            axis=1
        )
    return predictions


def summarize_predictions(
    predictions: pd.DataFrame, telemetry: dict | None = None
) -> dict:
    """Aggregate prediction-level metrics into one leaderboard row."""
    telemetry = telemetry or {}
    summary = {
        "model_id": predictions["model_id"].iat[0],
        "n": len(predictions),
        "token_f1_mean": predictions["token_f1"].mean(),
        "token_f1_std": predictions["token_f1"].std(),
        "answer_similarity_mean": predictions["answer_similarity"].mean()
        if "answer_similarity" in predictions
        else float("nan"),
        "answer_similarity_std": predictions["answer_similarity"].std()
        if "answer_similarity" in predictions
        else float("nan"),
    }

    if "reference_answer_nll_mean" in predictions:
        summary["reference_answer_nll_mean"] = predictions[
            "reference_answer_nll_mean"
        ].mean()
        summary["reference_answer_nll_std"] = predictions[
            "reference_answer_nll_mean"
        ].std()
    if "reference_answer_token_perplexity" in predictions:
        summary["reference_answer_token_perplexity_mean"] = predictions[
            "reference_answer_token_perplexity"
        ].mean()
        summary["reference_answer_token_perplexity_std"] = predictions[
            "reference_answer_token_perplexity"
        ].std()
        summary["reference_answer_token_perplexity_median"] = predictions[
            "reference_answer_token_perplexity"
        ].median()
    if "reference_answer_tokens" in predictions:
        summary["reference_answer_tokens_mean"] = predictions[
            "reference_answer_tokens"
        ].mean()
        summary["reference_answer_tokens_std"] = predictions[
            "reference_answer_tokens"
        ].std()
    return {**summary, **telemetry}
