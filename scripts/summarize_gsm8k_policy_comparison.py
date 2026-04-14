#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GSM8K accuracy deltas between two policies.")
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--nulled-summary", required=True)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    baseline = load_json(args.baseline_summary)
    nulled = load_json(args.nulled_summary)

    summary = {
        "baseline_policy": baseline,
        "length_nulled_policy": nulled,
        "deltas": {
            "accuracy": float(nulled["accuracy"]) - float(baseline["accuracy"]),
            "num_examples": int(nulled["num_examples"]) - int(baseline["num_examples"]),
        },
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
