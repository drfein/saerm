#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selector winners at a fixed N from multi-RM BoN details.")
    parser.add_argument("--details-file", required=True)
    parser.add_argument("--selectors", nargs="+", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.details_file))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = {selector: outdir / f"{selector}.jsonl" for selector in args.selectors}
    handles = {selector: path.open("w", encoding="utf-8") for selector, path in files.items()}
    try:
        for row in rows:
            idx = row.get("index")
            prompt = str(row.get("prompt", ""))
            candidates = row["candidates"]
            best_by_n = row["best_indices_by_n"]
            for selector in args.selectors:
                selector_map = best_by_n[selector]
                winner_idx = selector_map.get(str(args.n), selector_map.get(args.n))
                if winner_idx is None:
                    raise KeyError(f"Missing N={args.n} for selector={selector}")
                cand = candidates[int(winner_idx)]
                record = {
                    "id": str(idx),
                    "prompt": prompt,
                    "response": cand["response"],
                }
                handles[selector].write(json.dumps(record) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    summary = {
        "details_file": str(Path(args.details_file).resolve()),
        "num_examples": len(rows),
        "n": args.n,
        "selectors": args.selectors,
        "selector_files": {selector: str(path.resolve()) for selector, path in files.items()},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
