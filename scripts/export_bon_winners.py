#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selected BoN winners to JSONL files for downstream judging.")
    parser.add_argument("--bon-details-file", required=True)
    parser.add_argument("--selector-a", required=True)
    parser.add_argument("--selector-b", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.bon_details_file))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    selector_files = {
        args.selector_a: outdir / f"{args.selector_a}.jsonl",
        args.selector_b: outdir / f"{args.selector_b}.jsonl",
    }

    handles = {name: path.open("w", encoding="utf-8") for name, path in selector_files.items()}
    try:
        for row in rows:
            idx = row.get("index")
            prompt = str(row.get("prompt", ""))
            winners = row["selector_best_response"]
            for selector in (args.selector_a, args.selector_b):
                record = {
                    "id": str(idx),
                    "prompt": prompt,
                    "response": winners[selector],
                }
                handles[selector].write(json.dumps(record) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    summary = {
        "bon_details_file": str(Path(args.bon_details_file).resolve()),
        "num_examples": len(rows),
        "selector_a": args.selector_a,
        "selector_b": args.selector_b,
        "selector_a_file": str(selector_files[args.selector_a].resolve()),
        "selector_b_file": str(selector_files[args.selector_b].resolve()),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
