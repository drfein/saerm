#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter saved BoN candidate rows by correctness metadata.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--mode", choices=["any_correct", "all_incorrect"], required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def row_matches(row: dict[str, Any], mode: str) -> bool:
    flags = [bool(candidate.get("is_correct")) for candidate in row.get("candidates", [])]
    if mode == "any_correct":
        return any(flags)
    if mode == "all_incorrect":
        return not any(flags)
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.input_file))
    filtered = [row for row in rows if row_matches(row, args.mode)]

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in filtered:
            f.write(json.dumps(row) + "\n")

    summary = {
        "input_file": str(Path(args.input_file).resolve()),
        "output_file": str(output_path.resolve()),
        "mode": args.mode,
        "num_input_rows": len(rows),
        "num_output_rows": len(filtered),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
