#!/usr/bin/env python3
"""
Stage 1: Compute per-byte cross-entropy scores for all 10 generative LMs
on the tulu-3-wildchat dataset.

For each example (prompt, completion):
  s_m = -NLL_m(completion | prompt) / bytes(completion)

Saves a JSON file with scores per model.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

GENERATIVE_MODELS = [
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-3-12b-it",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-8B",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="allenai/tulu-3-wildchat-reused-on-policy-8b")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-prompts", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--output", required=True, help="Path to save NLL scores JSON.")
    parser.add_argument("--models", nargs="*", default=GENERATIVE_MODELS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def extract_prompt_and_completion(example: dict) -> list[tuple[str, str]]:
    """Returns list of (prompt_text, completion_text) for chosen and rejected."""
    results = []
    for field in ["chosen", "rejected"]:
        conv = example.get(field)
        if conv is None:
            continue
        if isinstance(conv, str):
            # Fallback: treat as single completion with empty prompt
            results.append(("", conv))
        elif isinstance(conv, list):
            # Chat format: list of {"role": ..., "content": ...}
            assistant_turns = [m for m in conv if m.get("role") == "assistant"]
            if not assistant_turns:
                continue
            completion = assistant_turns[-1]["content"]
            # Prompt = everything before the last assistant turn
            prompt_msgs = conv[: conv.index(assistant_turns[-1])]
            # Build prompt string using simple role prefixes (model-agnostic)
            prompt_parts = []
            for m in prompt_msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                prompt_parts.append(f"{role}: {content}")
            prompt = "\n".join(prompt_parts)
            results.append((prompt, completion))
    return results


def compute_nll_scores(
    model,
    tokenizer,
    examples: list[tuple[str, str]],
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> list[float | None]:
    """Compute -NLL(completion|prompt)/bytes(completion) for each (prompt, completion) pair."""
    scores: list[float | None] = []
    model.eval()

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        batch_scores = []

        for prompt, completion in batch:
            completion_bytes = len(completion.encode("utf-8"))
            if completion_bytes == 0:
                batch_scores.append(None)
                continue

            # Tokenize prompt alone to find boundary
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            full_text = prompt + completion
            full_ids = tokenizer.encode(full_text, add_special_tokens=True)

            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]

            n_prompt = len(prompt_ids)
            n_full = len(full_ids)
            n_completion = n_full - n_prompt

            if n_completion <= 0:
                batch_scores.append(None)
                continue

            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            labels = input_ids.clone()
            # Mask prompt tokens from loss
            labels[0, :n_prompt] = -100

            with torch.no_grad():
                out = model(input_ids=input_ids, labels=labels)
                nll_per_token = float(out.loss.item())  # mean NLL over completion tokens

            # Convert to per-byte: nll_per_token * n_completion_tokens / bytes
            nll_total = nll_per_token * n_completion
            score = -nll_total / completion_bytes
            batch_scores.append(score)

        scores.extend(batch_scores)
        if (i // batch_size) % 50 == 0:
            logger.info("  Processed %d/%d examples", i + len(batch), len(examples))

    return scores


def main() -> None:
    args = parse_args()

    logger.info("Loading dataset %s", args.dataset)
    ds = load_dataset(args.dataset, split=args.split)

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    selected = indices[: args.n_prompts]
    subset = ds.select(selected)

    # Build flat list of (prompt, completion) pairs
    examples: list[tuple[str, str]] = []
    example_meta: list[dict] = []  # track original index + field
    for idx, ex in zip(selected, subset):
        pairs = extract_prompt_and_completion(ex)
        for field_idx, (prompt, completion) in enumerate(pairs):
            examples.append((prompt, completion))
            example_meta.append({"dataset_idx": int(idx), "field_idx": field_idx})

    logger.info("Total examples: %d (from %d prompts)", len(examples), args.n_prompts)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = args.device

    all_scores: dict[str, list] = {}

    for model_name in args.models:
        logger.info("Computing NLL for %s", model_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
            )
            model.eval()

            scores = compute_nll_scores(
                model, tokenizer, examples,
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=device,
            )
            all_scores[model_name] = scores
            logger.info("  Done: %d scores, %d None", len(scores), scores.count(None))

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error("Failed for %s: %s", model_name, e)
            all_scores[model_name] = [None] * len(examples)

    output = {
        "dataset": args.dataset,
        "n_prompts": args.n_prompts,
        "seed": args.seed,
        "n_examples": len(examples),
        "example_meta": example_meta,
        "scores": all_scores,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output))
    logger.info("Saved NLL scores to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
