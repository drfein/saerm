#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.nb.datasets.base import ContrastivePair, format_conversation
from src.nb.nullbias.probe import build_probe_direction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a diff-mean length probe from saved BoN candidate pools.")
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--force-pair-format", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype name: {name}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def response_words(candidate: dict[str, Any]) -> int:
    if "response_words" in candidate:
        return int(candidate["response_words"])
    return len(str(candidate["response"]).split())


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.candidate_file))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.reward_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model,
        trust_remote_code=True,
        torch_dtype=dtype_from_name(args.torch_dtype),
    ).to(args.device)
    model.eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    pairs: list[ContrastivePair] = []
    for row in rows:
        prompt = str(row["prompt"])
        candidates = [dict(c) for c in row["candidates"]]
        order = sorted(range(len(candidates)), key=lambda i: (response_words(candidates[i]), i))
        shortest = candidates[order[0]]
        longest = candidates[order[-1]]
        pairs.append(
            ContrastivePair(
                positive_text=format_conversation(
                    tokenizer,
                    prompt=prompt,
                    response=str(longest["response"]),
                    force_pair=args.force_pair_format,
                ),
                negative_text=format_conversation(
                    tokenizer,
                    prompt=prompt,
                    response=str(shortest["response"]),
                    force_pair=args.force_pair_format,
                ),
                metadata={
                    "index": row.get("index"),
                    "shortest_words": response_words(shortest),
                    "longest_words": response_words(longest),
                },
            )
        )

    probe, metadata = build_probe_direction(
        model=model,
        tokenizer=tokenizer,
        contrastive_pairs=pairs,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
    )

    torch.save(probe.cpu(), outdir / "probe.pt")
    summary = {
        "candidate_file": str(Path(args.candidate_file).resolve()),
        "reward_model": args.reward_model,
        "method": "per_prompt_longest_vs_shortest",
        "n_pairs": len(pairs),
        **metadata,
    }
    (outdir / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
