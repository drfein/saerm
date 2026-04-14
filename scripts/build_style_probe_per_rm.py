#!/usr/bin/env python3
"""
Stage 2: For a given reward model, extract activations and build a style probe.

Binary label: y=1 if delta(example) > median(delta)
where delta = s_qwen - s_llama2
  s_m = -NLL_m(completion|prompt) / bytes(completion)  (from Stage 1 output)

Saves activations.pt compatible with analyze_iterative_nulling.py / plot_inlp_tsne.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# Default LM pair for binary split
DEFAULT_LM_A = "Qwen/Qwen3-8B"       # y=1 if this scores higher
DEFAULT_LM_B = "meta-llama/Llama-2-7b-chat-hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nll-scores", required=True, help="JSON output from compute_style_nll_scores.py")
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--output-dir", required=True, help="Dir to save activations.pt")
    parser.add_argument("--lm-a", default=DEFAULT_LM_A, help="LM whose higher score → y=1")
    parser.add_argument("--lm-b", default=DEFAULT_LM_B, help="LM whose higher score → y=0")
    parser.add_argument("--dataset", default="allenai/tulu-3-wildchat-reused-on-policy-8b")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def format_for_rm(prompt: str, completion: str, tokenizer, max_length: int) -> dict:
    """Format a (prompt, completion) pair as a reward model input."""
    # Try chat template if available
    try:
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        text = f"{prompt}\n\n{completion}"
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    return enc


def extract_activations_causal(model, tokenizer, texts_pairs: list[tuple[str, str]], *, batch_size, max_length, device):
    """Extract last-token hidden states from a causal LM reward model."""
    all_acts = []
    for i in range(0, len(texts_pairs), batch_size):
        batch = texts_pairs[i : i + batch_size]
        encodings = []
        for prompt, completion in batch:
            enc = format_for_rm(prompt, completion, tokenizer, max_length)
            encodings.append(enc)

        # Pad manually
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [e["input_ids"].squeeze(0) for e in encodings],
            batch_first=True,
            padding_value=tokenizer.pad_token_id or 0,
        ).to(device)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [e["attention_mask"].squeeze(0) for e in encodings],
            batch_first=True,
            padding_value=0,
        ).to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = out.hidden_states[-1]  # (B, T, D)
            # Last non-pad token per example
            lengths = attention_mask.sum(dim=1) - 1  # (B,)
            acts = hidden[torch.arange(len(batch), device=device), lengths]  # (B, D)

        all_acts.append(acts.cpu().float())
        if (i // batch_size) % 20 == 0:
            logger.info("  Extracted %d/%d", i + len(batch), len(texts_pairs))

    return torch.cat(all_acts, dim=0)


def extract_activations_encoder(model, tokenizer, texts_pairs: list[tuple[str, str]], *, batch_size, max_length, device):
    """Extract [CLS] hidden states from an encoder-based reward model (e.g. DeBERTa)."""
    all_acts = []
    for i in range(0, len(texts_pairs), batch_size):
        batch = texts_pairs[i : i + batch_size]
        texts = [f"{p}\n\n{c}" for p, c in batch]
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            hidden = out.hidden_states[-1]  # (B, T, D)
            acts = hidden[:, 0, :]  # [CLS] token

        all_acts.append(acts.cpu().float())
        if (i // batch_size) % 20 == 0:
            logger.info("  Extracted %d/%d", i + len(batch), len(texts_pairs))

    return torch.cat(all_acts, dim=0)


def main() -> None:
    args = parse_args()

    logger.info("Loading NLL scores from %s", args.nll_scores)
    nll_data = json.loads(Path(args.nll_scores).read_text())
    scores = nll_data["scores"]
    example_meta = nll_data["example_meta"]
    n_examples = nll_data["n_examples"]
    seed = nll_data["seed"]
    n_prompts = nll_data["n_prompts"]

    if args.lm_a not in scores:
        raise ValueError(f"LM-A '{args.lm_a}' not in NLL scores. Available: {list(scores.keys())}")
    if args.lm_b not in scores:
        raise ValueError(f"LM-B '{args.lm_b}' not in NLL scores. Available: {list(scores.keys())}")

    s_a = np.array([x if x is not None else float("nan") for x in scores[args.lm_a]])
    s_b = np.array([x if x is not None else float("nan") for x in scores[args.lm_b]])

    # Compute delta and binary labels
    delta = s_a - s_b
    valid = ~(np.isnan(delta))
    median_delta = float(np.nanmedian(delta))
    labels = (delta > median_delta).astype(int)  # 1 = LM-A style, 0 = LM-B style

    logger.info("Delta: median=%.4f, valid=%d/%d, y=1: %d, y=0: %d",
                median_delta, valid.sum(), len(delta),
                int((labels[valid] == 1).sum()), int((labels[valid] == 0).sum()))

    # Reload dataset to reconstruct (prompt, completion) pairs in same order
    import random as _random
    logger.info("Reloading dataset %s", args.dataset)
    ds = load_dataset(args.dataset, split=args.dataset_split)
    rng = _random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    selected = indices[:n_prompts]
    subset = ds.select(selected)

    # Reconstruct examples in same order as Stage 1
    from scripts.compute_style_nll_scores import extract_prompt_and_completion
    examples: list[tuple[str, str]] = []
    for ex in subset:
        pairs = extract_prompt_and_completion(ex)
        examples.extend(pairs)
    assert len(examples) == n_examples, f"Example count mismatch: {len(examples)} vs {n_examples}"

    # Filter to valid examples only
    valid_indices = [i for i in range(n_examples) if valid[i]]
    examples_valid = [examples[i] for i in valid_indices]
    labels_valid = labels[valid_indices]

    logger.info("Using %d valid examples", len(examples_valid))

    # Load reward model
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = args.device
    is_encoder = "deberta" in args.reward_model.lower()

    logger.info("Loading reward model %s", args.reward_model)
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_encoder:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model, torch_dtype=dtype, device_map=device, trust_remote_code=args.trust_remote_code
        )
        extract_fn = extract_activations_encoder
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.reward_model, torch_dtype=dtype, device_map=device, trust_remote_code=args.trust_remote_code
        )
        extract_fn = extract_activations_causal

    model.eval()

    logger.info("Extracting activations...")
    activations = extract_fn(
        model, tokenizer, examples_valid,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    # Split into positive/negative by style label
    pos_mask = torch.tensor(labels_valid == 1)
    neg_mask = torch.tensor(labels_valid == 0)
    positive_embeddings = activations[pos_mask]
    negative_embeddings = activations[neg_mask]

    logger.info("Positive (LM-A style): %d, Negative (LM-B style): %d",
                positive_embeddings.shape[0], negative_embeddings.shape[0])

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "positive_embeddings": positive_embeddings,
        "negative_embeddings": negative_embeddings,
    }
    torch.save(payload, out_dir / "activations.pt")

    meta = {
        "reward_model": args.reward_model,
        "lm_a": args.lm_a,
        "lm_b": args.lm_b,
        "n_positive": int(positive_embeddings.shape[0]),
        "n_negative": int(negative_embeddings.shape[0]),
        "median_delta": median_delta,
        "hidden_dim": int(activations.shape[1]),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved activations.pt and metadata.json to %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
