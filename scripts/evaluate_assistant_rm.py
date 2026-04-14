#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.nb.downstream.scoring import RewardModelScorer
from src.nb.downstream.gsm8k import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate assistant responses and score them with baseline and nulled RM.")
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe-file")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--reward-device", default="cuda")
    parser.add_argument("--policy-dtype", default="bfloat16")
    parser.add_argument("--reward-dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def render_prompt(tokenizer: Any, prompt: str, messages: list[dict[str, str]] | None) -> str:
    if messages and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    rows = load_jsonl(args.eval_file)
    if args.limit is not None:
        rows = rows[:args.limit]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_model, trust_remote_code=args.trust_remote_code)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    if getattr(policy_tokenizer, "padding_side", None) != "left":
        policy_tokenizer.padding_side = "left"
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype_from_name(args.policy_dtype),
    ).to(args.policy_device)
    policy_model.eval()

    scorer = RewardModelScorer(
        model_path=args.reward_model,
        probe_path=args.probe_file,
        alpha=args.alpha,
        device=args.reward_device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.reward_dtype,
    )

    details: list[dict[str, Any]] = []
    baseline_scores = []
    nulled_scores = []
    response_lengths = []

    for start in tqdm(
        range(0, len(rows), args.generation_batch_size),
        desc="Generating and scoring",
        disable=len(rows) <= args.generation_batch_size,
    ):
        batch_rows = rows[start : start + args.generation_batch_size]
        prompts = [str(row.get("prompt", "")).strip() for row in batch_rows]
        messages = [row.get("messages") if isinstance(row.get("messages"), list) else None for row in batch_rows]
        rendered_prompts = [
            render_prompt(policy_tokenizer, prompt, message_history)
            for prompt, message_history in zip(prompts, messages)
        ]
        inputs = policy_tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(args.policy_device)
        with torch.no_grad():
            outputs = policy_model.generate(
                **inputs,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=policy_tokenizer.pad_token_id,
                eos_token_id=policy_tokenizer.eos_token_id,
            )
        continuation = outputs[:, inputs["input_ids"].shape[1] :]
        responses = [text.strip() for text in policy_tokenizer.batch_decode(continuation, skip_special_tokens=True)]
        base, nulled = scorer.score_pairs_both(
            prompts,
            responses,
            message_histories=messages,
        )
        for row, prompt, response, baseline_score, nulled_score in zip(
            batch_rows,
            prompts,
            responses,
            base,
            nulled,
        ):
            baseline_scores.append(baseline_score)
            nulled_scores.append(nulled_score)
            response_lengths.append(len(response.split()))
            details.append(
                {
                    "id": row.get("id"),
                    "prompt": prompt,
                    "response": response,
                    "baseline_score": baseline_score,
                    "nulled_score": nulled_score,
                    "response_words": response_lengths[-1],
                    "reference_chosen": row.get("completion"),
                }
            )

    summary = {
        "num_examples": len(details),
        "policy_model": args.policy_model,
        "reward_model": args.reward_model,
        "probe_file": args.probe_file,
        "baseline_score_mean": sum(baseline_scores) / max(len(baseline_scores), 1),
        "nulled_score_mean": sum(nulled_scores) / max(len(nulled_scores), 1),
        "response_words_mean": sum(response_lengths) / max(len(response_lengths), 1),
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (outdir / "details.jsonl").open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
