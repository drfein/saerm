#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize behavior evals for three policies.")
    parser.add_argument("--baseline-rm-summary", required=True)
    parser.add_argument("--baseline-length-summary", required=True)
    parser.add_argument("--length-penalty-rm-summary", required=True)
    parser.add_argument("--length-penalty-length-summary", required=True)
    parser.add_argument("--length-nulled-rm-summary", required=True)
    parser.add_argument("--length-nulled-length-summary", required=True)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_policy_summary(rm_summary: dict, length_summary: dict) -> dict:
    return {
        "rm_eval": rm_summary,
        "length_only": length_summary,
    }


def build_delta(lhs: dict, rhs: dict) -> dict:
    return {
        "rm_eval": {
            "baseline_score_mean": rhs["rm_eval"]["baseline_score_mean"] - lhs["rm_eval"]["baseline_score_mean"],
            "nulled_score_mean": rhs["rm_eval"]["nulled_score_mean"] - lhs["rm_eval"]["nulled_score_mean"],
            "response_words_mean": rhs["rm_eval"]["response_words_mean"] - lhs["rm_eval"]["response_words_mean"],
        },
        "length_only": {
            "response_words_mean": rhs["length_only"]["response_words_mean"] - lhs["length_only"]["response_words_mean"],
            "response_words_min": rhs["length_only"]["response_words_min"] - lhs["length_only"]["response_words_min"],
            "response_words_max": rhs["length_only"]["response_words_max"] - lhs["length_only"]["response_words_max"],
        },
    }


def main() -> None:
    args = parse_args()

    baseline = build_policy_summary(
        load_json(args.baseline_rm_summary),
        load_json(args.baseline_length_summary),
    )
    length_penalty = build_policy_summary(
        load_json(args.length_penalty_rm_summary),
        load_json(args.length_penalty_length_summary),
    )
    length_nulled = build_policy_summary(
        load_json(args.length_nulled_rm_summary),
        load_json(args.length_nulled_length_summary),
    )

    summary = {
        "baseline_policy": baseline,
        "length_penalty_policy": length_penalty,
        "length_nulled_policy": length_nulled,
        "deltas": {
            "baseline_to_length_penalty": build_delta(baseline, length_penalty),
            "baseline_to_length_nulled": build_delta(baseline, length_nulled),
            "length_penalty_to_length_nulled": build_delta(length_penalty, length_nulled),
        },
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
