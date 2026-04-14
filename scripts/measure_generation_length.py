#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure average generated response length for a causal LM.")
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def load_rows(path: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            rows.append(json.loads(line))
    return rows


def render_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = row.get("messages") if isinstance(row.get("messages"), list) else None
    prompt = str(row.get("prompt", "")).strip()
    if messages and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt


def quantile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[lo])
    frac = idx - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def resolve_local_tokenizer_source(model_name_or_path: str) -> str:
    path = Path(model_name_or_path)
    if not path.exists():
        return model_name_or_path
    search_root = path if path.is_dir() else path.parent
    tokenizer_markers = (
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "tokenizer.model",
    )
    for candidate in (search_root, *search_root.parents):
        if any((candidate / marker).exists() for marker in tokenizer_markers):
            return str(candidate)
    return str(search_root)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.eval_file, args.limit)

    tokenizer_source = resolve_local_tokenizer_source(args.policy_model)
    local_files_only = Path(args.policy_model).exists() or Path(tokenizer_source).exists()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype_from_name(args.dtype),
        local_files_only=Path(args.policy_model).exists(),
    ).to(args.device)
    model.eval()

    lengths: list[int] = []
    for start in tqdm(
        range(0, len(rows), args.batch_size),
        desc="Generating lengths",
        disable=len(rows) <= args.batch_size,
    ):
        batch_rows = rows[start : start + args.batch_size]
        rendered_prompts = [render_prompt(tokenizer, row) for row in batch_rows]
        inputs = tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(args.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        continuation = outputs[:, inputs["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(continuation, skip_special_tokens=True)
        lengths.extend(len(text.strip().split()) for text in texts)

    summary = {
        "num_examples": len(lengths),
        "policy_model": args.policy_model,
        "response_words_mean": sum(lengths) / max(1, len(lengths)),
        "response_words_stdev": statistics.pstdev(lengths) if lengths else 0.0,
        "response_words_variance": statistics.pvariance(lengths) if lengths else 0.0,
        "response_words_min": min(lengths) if lengths else 0,
        "response_words_p25": quantile(lengths, 0.25),
        "response_words_median": quantile(lengths, 0.5),
        "response_words_p75": quantile(lengths, 0.75),
        "response_words_max": max(lengths) if lengths else 0,
    }
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
