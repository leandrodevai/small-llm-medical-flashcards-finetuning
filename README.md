# Small LLM Fine-Tuning for Medical Flashcards

An end-to-end case study on selecting, fine-tuning, and evaluating 1B-4B language models for medical flashcard answer generation on consumer hardware.

The goal is not to build a clinical system. It is to show a typical ML workflow: establish a baseline, measure the cost of quantization, select candidates, tune them with QLoRA, and evaluate the final models on held-out data.

## Results at a glance

Eleven instruction-tuned models were benchmarked before selecting SmolLM2 1.7B and Llama 3.2 1B for fine-tuning. Both adapters were selected using validation loss and evaluated once on a fixed 128-example test sample.

| Model | Setup | Token F1 | Cosine similarity | Token F1 gain |
| --- | --- | ---: | ---: | ---: |
| SmolLM2 1.7B | Base | 0.292 | 0.803 | - |
| SmolLM2 1.7B | QLoRA | **0.454** | **0.838** | **+0.162** |
| Llama 3.2 1B | Base | 0.235 | 0.773 | - |
| Llama 3.2 1B | QLoRA | **0.441** | **0.831** | **+0.206** |

Across the full baseline, 4-bit NF4 quantization reduced average peak reserved VRAM from **16.7 GB to 7.4 GB** while mean Token F1 changed from 0.235 to 0.234 and cosine similarity from 0.776 to 0.772.

These metrics measure similarity to a reference answer. They do not establish medical correctness or clinical safety.

## Work design

1. Explore the dataset, prompt format, and LoRA feasibility in notebooks.
2. Benchmark 11 models on the same fixed validation sample in BF16 and 4-bit NF4.
3. Select a strong 1.7B baseline and a smaller 1B candidate.
4. Run QLoRA sweeps using only train and validation data.
5. Select checkpoints by validation loss and evaluate them once on the held-out test sample.
6. Move the reusable workflow into configuration-driven CLI commands.

## Baseline results

The baseline uses 128 fixed validation examples for every model.

| Model | BF16 F1 | NF4 F1 | BF16 cosine | NF4 cosine |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2 1.7B Instruct | 0.292 | **0.332** | 0.803 | **0.809** |
| Qwen3 4B Instruct | 0.243 | 0.230 | 0.802 | 0.798 |
| SmolLM3 3B | 0.268 | 0.271 | 0.797 | 0.793 |
| Qwen2.5 3B Instruct | 0.277 | 0.270 | 0.793 | 0.793 |
| Llama 3.2 3B Instruct | 0.262 | 0.258 | 0.788 | 0.790 |
| Qwen3 1.7B | 0.234 | 0.218 | 0.782 | 0.773 |
| Qwen2.5 1.5B Instruct | 0.258 | 0.252 | 0.773 | 0.772 |
| Llama 3.2 1B Instruct | 0.232 | 0.241 | 0.769 | 0.763 |
| MedGemma 1.5 4B IT | 0.199 | 0.193 | 0.756 | 0.746 |
| Gemma 3 4B IT | 0.166 | 0.164 | 0.741 | 0.744 |
| Gemma 3 1B IT | 0.158 | 0.149 | 0.734 | 0.717 |

The main result was operational: NF4 cut average peak reserved VRAM by about **56%** with less than 0.005 absolute change in either mean proxy metric. Complete predictions and telemetry are stored in [`outputs/baseline/`](outputs/baseline/) and [`outputs/baseline_4bits/`](outputs/baseline_4bits/).

## What changed after fine-tuning

QLoRA mostly improved task fit: answers became shorter, more direct, and closer to the expected flashcard format.

### Successful example

**Question:** What is the primary long-term intervention for persistent asthma?

**Base model:** "A combination of medications and lifestyle modifications..." followed by a 276-token list.

**QLoRA:** "The primary long-term intervention for persistent asthma is inhaled corticosteroids."

**Reference:** "The primary long-term intervention for persistent asthma is daily inhaled corticosteroids."

Token F1 improved from 0.107 to 0.952.

### What the metrics miss

In a chest-trauma diagnosis example, the fine-tuned model answered **pneumothorax** while the reference was **myocardial contusion**. Token F1 was still 0.800 because most of the sentence structure matched; cosine similarity was 0.577.

This is why lexical and embedding metrics are useful for fast iteration but insufficient for medical QA. A stronger evaluation would add blinded expert review or an LLM judge with a medical rubric for factual errors, missing facts, unsupported claims, and potentially harmful answers.

## Dataset and evaluation

The source is [`flwrlabs/medical-meadow-medical-flashcards`](https://huggingface.co/datasets/flwrlabs/medical-meadow-medical-flashcards). A fixed split is published as [`leandrodevai/medical-meadow-medical-flashcards-splitted`](https://huggingface.co/datasets/leandrodevai/medical-meadow-medical-flashcards-splitted).

| Split | Rows | Usage |
| --- | ---: | --- |
| Train | 30,000 | Fine-tuning |
| Validation | 1,977 | Baselines, sweeps, and checkpoint selection |
| Test | 1,978 | Final evaluation only |

Subsampling uses seed `42`.

The evaluation records:

- token-level F1 after light answer normalization;
- cosine similarity using `sentence-transformers/all-MiniLM-L6-v2`;
- reference-answer NLL and perplexity;
- inference time, throughput, generated tokens, and peak VRAM.

Every evaluated response is saved to CSV so aggregate scores can be traced back to individual generations. Fine-tuning results are in [`outputs/finetuning_evaluation/`](outputs/finetuning_evaluation/).

## Project structure

```text
configs/                  Dataset, model, training, and sweep configuration
notebooks/                Initial exploration and proof of concept
src/medical_flashcards/   Training, evaluation, metrics, and model utilities
scripts/                  W&B agent helpers for PowerShell and Bash
outputs/                  Predictions, telemetry, and leaderboards
tests/                    Unit tests for reusable logic
```

The notebooks preserve the exploration history. The CLI is the reproducible path.

## Running the project

Requirements: Python 3.12, `uv`, and an NVIDIA GPU. The current lockfile uses PyTorch for CUDA 12.6.

Create a local environment file from the provided template:

```bash
# Bash
cp .env.example .env

# PowerShell
Copy-Item .env.example .env
```

Add `HF_TOKEN` only when using gated Hugging Face models. `WANDB_API_KEY` is optional for regular training and required for W&B sweeps. The CLI loads `.env` automatically, and the file is ignored by Git.

```bash
uv sync
uv run pytest
```

Run a BF16 or NF4 baseline:

```bash
uv run medical-flashcards-baseline --eval-size 128

uv run medical-flashcards-baseline \
  --eval-size 128 \
  --load-in-4bit \
  --output-dir outputs/baseline_4bits
```

Train one QLoRA configuration:

```bash
uv run medical-flashcards-train --config configs/train.yaml
```

Evaluate a selected adapter on test:

```bash
uv run medical-flashcards-eval \
  --model-id MODEL_OR_ADAPTER_ID \
  --split test \
  --eval-size 128 \
  --output-dir outputs/test_evaluation
```

Create a W&B sweep:

```bash
uv run medical-flashcards-sweep \
  configs/sweeps/smollm2_lora.yaml \
  --project medical-flashcards-lora
```

Sweep agents can be distributed across GPUs or machines with `scripts/run_wandb_agent.sh` and `scripts/run_wandb_agent.ps1`.

## Scope

This is a proof of concept, not a validated medical QA system or a source of medical advice. It follows a typical ML workflow: start with exploratory experiments, validate the approach, and turn the useful parts into a reproducible training and evaluation pipeline.
