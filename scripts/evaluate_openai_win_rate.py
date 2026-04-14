#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm


SYSTEM_PROMPT = """You are a careful evaluator of helpful assistant responses.
Given a user prompt and two candidate assistant responses, choose the better response.
Judge on overall helpfulness, correctness, relevance, clarity, and harmlessness.
Do not prefer verbosity by itself.
If the responses are genuinely tied, return tie.
Respond with JSON only: {"winner":"A"|"B"|"tie"}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pairwise win rate with an OpenAI judge model.")
    parser.add_argument("--model-a-file", required=True)
    parser.add_argument("--model-b-file", required=True)
    parser.add_argument("--model-a-name", required=True)
    parser.add_argument("--model-b-name", required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--user-instruction", default="Return only JSON with the better response.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--reasoning-effort")
    return parser.parse_args()


def load_jsonl(path: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records[row["id"]] = row
    return records


def make_user_prompt(prompt: str, response_a: str, response_b: str, instruction: str) -> str:
    return (
        f"User prompt:\n{prompt}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        + instruction
    )


def call_openai_responses(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    current_max_output_tokens = max_output_tokens
    for _ in range(4):
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "pairwise_winrate",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "winner": {"type": "string", "enum": ["A", "B", "tie"]},
                        },
                        "required": ["winner"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
                "verbosity": "low",
            },
            "max_output_tokens": current_max_output_tokens,
        }
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        if temperature != 0.0:
            payload["temperature"] = temperature

        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text_value = data.get("output_text")
        if text_value:
            return json.loads(text_value)

        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    return json.loads(content["text"])

        incomplete = data.get("incomplete_details") or {}
        if data.get("status") == "incomplete" and incomplete.get("reason") == "max_output_tokens":
            current_max_output_tokens *= 2
            continue

        raise ValueError(f"Could not parse response payload: {data}")

    raise ValueError("OpenAI response remained incomplete after increasing max_output_tokens")


def remap_winner(raw_winner: str, a_is_model_a: bool) -> str:
    if raw_winner == "tie":
        return "tie"
    if a_is_model_a:
        return "model_a" if raw_winner == "A" else "model_b"
    return "model_b" if raw_winner == "A" else "model_a"


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    reasoning_effort = args.reasoning_effort
    if reasoning_effort == "none" and args.model.startswith("gpt-5"):
        print("[warn] gpt-5 does not support reasoning effort 'none'; using 'minimal' instead.")
        reasoning_effort = "minimal"

    model_a = load_jsonl(args.model_a_file)
    model_b = load_jsonl(args.model_b_file)
    shared_ids = sorted(set(model_a) & set(model_b))
    if args.max_examples is not None:
        shared_ids = shared_ids[: args.max_examples]
    if not shared_ids:
        raise SystemExit("No shared ids between model outputs")

    rng = random.Random(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    details_path = outdir / "details.jsonl"

    counts = Counter()
    with details_path.open("w", encoding="utf-8") as out_f:
        for ex_id in tqdm(shared_ids, desc="OpenAI win-rate eval"):
            row_a = model_a[ex_id]
            row_b = model_b[ex_id]
            a_first = rng.random() < 0.5
            displayed_a = row_a["response"] if a_first else row_b["response"]
            displayed_b = row_b["response"] if a_first else row_a["response"]
            judge = None

            for attempt in range(6):
                try:
                    judge = call_openai_responses(
                        api_key=api_key,
                        model=args.model,
                        system_prompt=args.system_prompt,
                    user_prompt=make_user_prompt(row_a["prompt"], displayed_a, displayed_b, args.user_instruction),
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    reasoning_effort=reasoning_effort,
                )
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
                    if attempt == 5:
                        raise
                    time.sleep(min(30, 2 ** attempt))

            assert judge is not None
            winner = remap_winner(str(judge["winner"]), a_is_model_a=a_first)
            counts[winner] += 1

            record = {
                "id": ex_id,
                "prompt": row_a["prompt"],
                "model_a_name": args.model_a_name,
                "model_b_name": args.model_b_name,
                "model_a_response": row_a["response"],
                "model_b_response": row_b["response"],
                "displayed_a_is_model_a": a_first,
                "judge_winner_raw": judge["winner"],
                "winner": winner,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)

    total = len(shared_ids)
    summary = {
        "judge_model": args.model,
        "num_examples": total,
        "model_a_name": args.model_a_name,
        "model_b_name": args.model_b_name,
        "model_a_wins": counts["model_a"],
        "model_b_wins": counts["model_b"],
        "ties": counts["tie"],
        "model_a_win_rate": counts["model_a"] / total,
        "model_b_win_rate": counts["model_b"] / total,
        "tie_rate": counts["tie"] / total,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
