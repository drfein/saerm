#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.nb.downstream.gsm8k import extract_final_answer, is_correct, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a policy on GSM8K exact match.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def render_prompt(tokenizer: Any, prompt: str, messages: list[dict[str, str]] | None) -> torch.Tensor:
    if messages and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt")
    return tokenizer(prompt, return_tensors="pt").input_ids


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    rows = load_jsonl(args.eval_file)
    if args.limit is not None:
        rows = rows[:args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype_from_name(args.torch_dtype),
    ).to(args.device)
    model.eval()

    results: list[dict[str, Any]] = []
    correct = 0
    for row in rows:
        prompt = str(row["prompt"])
        messages = row.get("messages")
        inputs = render_prompt(tokenizer, prompt, messages).to(args.device)
        gen = model.generate(
            inputs,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        continuation = gen[0][inputs.shape[-1]:]
        text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        is_ok = is_correct(text, row.get("answer"))
        correct += int(is_ok)
        results.append(
            {
                "id": row.get("id"),
                "question": row.get("question"),
                "gold_answer": row.get("answer"),
                "prediction_text": text,
                "prediction_final_answer": extract_final_answer(text),
                "is_correct": is_ok,
            }
        )

    summary = {
        "num_examples": len(results),
        "accuracy": correct / max(len(results), 1),
        "model_id": args.model_id,
        "eval_file": str(Path(args.eval_file).resolve()),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
