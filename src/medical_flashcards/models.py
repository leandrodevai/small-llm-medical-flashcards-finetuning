"""Model loading and CUDA memory helpers."""

from __future__ import annotations

import gc


def load_tokenizer(model_id: str, *, trust_remote_code: bool = True):
    """Load a tokenizer with left padding for batched generation."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_causal_lm(
    model_id: str,
    *,
    load_in_4bit: bool = False,
    prepare_for_kbit: bool = False,
    trust_remote_code: bool = True,
):
    """Load a causal language model and tokenizer.

    Args:
        model_id: Hugging Face model identifier.
        load_in_4bit: Load weights with bitsandbytes NF4 quantization.
        prepare_for_kbit: Apply PEFT preparation for k-bit training.
        trust_remote_code: Forwarded to Transformers loaders.

    Returns:
        Tuple of `(model, tokenizer)`.
    """
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(model_id, trust_remote_code=trust_remote_code)
    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    kwargs = {
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not torch.cuda.is_available():
        model.to("cpu")
    if prepare_for_kbit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def release_model(model, tokenizer) -> None:
    """Release model references and clear CUDA cache when available."""
    import torch

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_device_ids() -> list[int]:
    """Return visible CUDA device IDs."""
    import torch

    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


def synchronize_cuda() -> None:
    """Synchronize all visible CUDA devices."""
    import torch

    for device_id in cuda_device_ids():
        torch.cuda.synchronize(device_id)


def reset_cuda_peak_memory() -> None:
    """Reset peak CUDA memory counters for visible devices."""
    import torch

    if not torch.cuda.is_available():
        return
    synchronize_cuda()
    for device_id in cuda_device_ids():
        torch.cuda.reset_peak_memory_stats(device_id)


def cuda_peak_memory_mb() -> dict[str, float]:
    """Return peak allocated and reserved CUDA memory in MB."""
    import torch

    if not torch.cuda.is_available():
        return {
            "vram_peak_allocated_mb": float("nan"),
            "vram_peak_reserved_mb": float("nan"),
        }
    synchronize_cuda()
    allocated = sum(torch.cuda.max_memory_allocated(i) for i in cuda_device_ids())
    reserved = sum(torch.cuda.max_memory_reserved(i) for i in cuda_device_ids())
    return {
        "vram_peak_allocated_mb": allocated / 1024**2,
        "vram_peak_reserved_mb": reserved / 1024**2,
    }
