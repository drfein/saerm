#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize behavior deltas between two policy evaluation outputs.")
    parser.add_argument("--baseline-rm-summary", required=True)
    parser.add_argument("--nulled-rm-summary", required=True)
    parser.add_argument("--baseline-length-summary", required=True)
    parser.add_argument("--nulled-length-summary", required=True)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    baseline_rm = load_json(args.baseline_rm_summary)
    nulled_rm = load_json(args.nulled_rm_summary)
    baseline_len = load_json(args.baseline_length_summary)
    nulled_len = load_json(args.nulled_length_summary)

    summary = {
        "baseline_policy": {
            "rm_eval": baseline_rm,
            "length_only": baseline_len,
        },
        "length_nulled_policy": {
            "rm_eval": nulled_rm,
            "length_only": nulled_len,
        },
        "deltas": {
            "rm_eval": {
                "baseline_score_mean": nulled_rm["baseline_score_mean"] - baseline_rm["baseline_score_mean"],
                "nulled_score_mean": nulled_rm["nulled_score_mean"] - baseline_rm["nulled_score_mean"],
                "response_words_mean": nulled_rm["response_words_mean"] - baseline_rm["response_words_mean"],
            },
            "length_only": {
                "response_words_mean": nulled_len["response_words_mean"] - baseline_len["response_words_mean"],
                "response_words_min": nulled_len["response_words_min"] - baseline_len["response_words_min"],
                "response_words_max": nulled_len["response_words_max"] - baseline_len["response_words_max"],
            },
        },
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
