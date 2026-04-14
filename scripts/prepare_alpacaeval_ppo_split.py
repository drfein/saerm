#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare prompt-only PPO train/eval splits from alpaca_eval.json.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def to_prompt_record(idx: int, row: dict[str, Any], split: str) -> dict[str, Any]:
    question = str(row.get("instruction", "")).strip()
    if not question:
        raise ValueError(f"Missing instruction for row {idx}")
    return {
        "id": f"alpacaeval_{split}_{idx}",
        "data_source": "alpacaeval",
        "ability": "instruction_following",
        "question": question,
        "prompt": question,
        "extra_info": {
            "dataset": row.get("dataset"),
            "reference_output": row.get("output"),
            "generator": row.get("generator"),
            "split": split,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    outdir = Path(args.output_dir)
    records = load_records(source)
    indices = list(range(len(records)))
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(indices)
        records = [records[i] for i in indices]

    if not 0 < args.train_size < len(records):
        raise ValueError(f"train-size must be between 1 and {len(records) - 1}")
    if not 0 <= args.train_offset < len(records):
        raise ValueError(f"train-offset must be between 0 and {len(records) - 1}")
    if args.train_offset + args.train_size > len(records):
        raise ValueError(
            f"train slice [{args.train_offset}, {args.train_offset + args.train_size}) exceeds dataset size {len(records)}"
        )

    train_slice = records[args.train_offset : args.train_offset + args.train_size]
    eval_slice = records[: args.train_offset] + records[args.train_offset + args.train_size :]

    train_rows = [to_prompt_record(i, row, "train") for i, row in enumerate(train_slice)]
    val_rows = [to_prompt_record(i, row, "val") for i, row in enumerate(eval_slice)]

    ppo_dir = outdir / "ppo"
    eval_dir = outdir / "eval"
    write_jsonl(ppo_dir / "train.jsonl", train_rows)
    write_jsonl(ppo_dir / "val.jsonl", val_rows)
    write_jsonl(eval_dir / "test.jsonl", val_rows)

    manifest = {
        "source": str(source.resolve()),
        "num_total": len(records),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "train_offset": args.train_offset,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "paths": {
            "ppo_train": str((ppo_dir / "train.jsonl").resolve()),
            "ppo_val": str((ppo_dir / "val.jsonl").resolve()),
            "eval_test": str((eval_dir / "test.jsonl").resolve()),
        },
    }
    with (outdir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
