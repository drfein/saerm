#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nb.downstream.gsm8k import extract_final_answer, is_correct
from src.nb.downstream.scoring import RewardModelScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate best-of-n selection with a reward model.")
    parser.add_argument("--policy-model", required=True, help="Policy model path or HF ID")
    parser.add_argument("--reward-model", required=True, help="Reward model path or HF ID")
    parser.add_argument("--probe-file", help="Optional probe tensor for nulled scoring")
    parser.add_argument("--alpha", type=float, default=1.0, help="Nulling strength")
    parser.add_argument("--prompts", required=True, help="JSON/JSONL file or dataset ID containing prompts")
    parser.add_argument("--split", default="train", help="Dataset split when --prompts is a dataset ID")
    parser.add_argument("--prompt-key", default="prompt", help="Prompt key for dict records")
    parser.add_argument("--messages-key", default="messages", help="Messages key for chat-formatted records")
    parser.add_argument("--answer-key", default="answer", help="Gold answer key for exact-match metrics")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    parser.add_argument("--num-prompts", type=int, default=128, help="Number of prompts to evaluate")
    parser.add_argument("--n", type=int, default=8, help="Candidates per prompt")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--policy-device", default="cuda", help="Device for policy generation")
    parser.add_argument("--reward-device", default="cuda", help="Device for reward model scoring")
    parser.add_argument("--policy-dtype", default="bfloat16", help="Policy dtype")
    parser.add_argument("--reward-dtype", default="bfloat16", help="Reward model dtype")
    parser.add_argument("--batch-size", type=int, default=8, help="Reward model batch size")
    parser.add_argument("--max-length", type=int, default=2048, help="Reward model max length")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--force-pair-format", action="store_true", help="Force RM pair formatting")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_prompt_records(source: str, split: str) -> list[Any]:
    path = Path(source)
    if path.exists():
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if not isinstance(data, list):
            raise ValueError(f"Expected list-like JSON data in {path}")
        return data

    ds = load_dataset(source, split=split)
    return [ds[i] for i in range(len(ds))]


def extract_prompt(record: Any, prompt_key: str, messages_key: str) -> tuple[str, list[dict[str, str]] | None]:
    if isinstance(record, str):
        return record, None

    if isinstance(record, dict):
        if messages_key in record and isinstance(record[messages_key], list):
            messages = record[messages_key]
            prompt_text = ""
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    prompt_text = str(message.get("content", "")).strip()
                    break
            return prompt_text, messages

        if prompt_key in record:
            return str(record[prompt_key]).strip(), None

        for fallback in ("question", "input", "instruction"):
            if fallback in record:
                return str(record[fallback]).strip(), None

    raise ValueError(f"Could not extract prompt from record keys={list(record.keys()) if isinstance(record, dict) else type(record)}")


def dtype_from_name(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def render_prompt(
    tokenizer: Any,
    *,
    prompt_text: str,
    messages: list[dict[str, str]] | None,
) -> torch.Tensor:
    if messages:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            return_tensors="pt",
        )

    return tokenizer(prompt_text, return_tensors="pt").input_ids


def generate_candidates(
    *,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    messages: list[dict[str, str]] | None,
    n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    device: str,
) -> list[str]:
    input_ids = render_prompt(tokenizer, prompt_text=prompt_text, messages=messages).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=n,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = input_ids.shape[-1]
    candidates: list[str] = []
    for seq in outputs:
        continuation = seq[prompt_len:]
        text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        candidates.append(text)
    return candidates


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_records = load_prompt_records(args.prompts, args.split)
    prompt_records = prompt_records[: args.num_prompts]

    policy_tokenizer = AutoTokenizer.from_pretrained(
        args.policy_model,
        trust_remote_code=args.trust_remote_code,
    )
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

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
        force_pair_format=args.force_pair_format,
    )

    records_out: list[dict[str, Any]] = []
    overlap_count = 0
    baseline_selected_scores: list[float] = []
    nulled_selected_scores: list[float] = []
    cross_baseline_scores: list[float] = []
    cross_nulled_scores: list[float] = []
    baseline_correct = 0
    nulled_correct = 0
    oracle_correct = 0

    for idx, record in enumerate(prompt_records):
        prompt_text, messages = extract_prompt(record, args.prompt_key, args.messages_key)
        gold_answer = record.get(args.answer_key) if isinstance(record, dict) else None
        candidates = generate_candidates(
            model=policy_model,
            tokenizer=policy_tokenizer,
            prompt_text=prompt_text,
            messages=messages,
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            device=args.policy_device,
        )
        prompts = [prompt_text] * len(candidates)
        histories = [messages] * len(candidates) if messages else None
        baseline_scores, nulled_scores = scorer.score_pairs_both(
            prompts,
            candidates,
            message_histories=histories,
        )

        baseline_best = max(range(len(candidates)), key=lambda i: baseline_scores[i])
        nulled_best = max(range(len(candidates)), key=lambda i: nulled_scores[i])
        overlap_count += int(baseline_best == nulled_best)
        candidate_correct = [is_correct(c, gold_answer) for c in candidates] if gold_answer is not None else [False] * len(candidates)
        if gold_answer is not None:
            baseline_correct += int(candidate_correct[baseline_best])
            nulled_correct += int(candidate_correct[nulled_best])
            oracle_correct += int(any(candidate_correct))

        baseline_selected_scores.append(baseline_scores[baseline_best])
        nulled_selected_scores.append(nulled_scores[nulled_best])
        cross_baseline_scores.append(nulled_scores[baseline_best])
        cross_nulled_scores.append(baseline_scores[nulled_best])

        records_out.append(
            {
                "index": idx,
                "prompt": prompt_text,
                "baseline_best_index": baseline_best,
                "nulled_best_index": nulled_best,
                "baseline_best_response": candidates[baseline_best],
                "nulled_best_response": candidates[nulled_best],
                "gold_answer": gold_answer,
                "baseline_best_correct": candidate_correct[baseline_best] if gold_answer is not None else None,
                "nulled_best_correct": candidate_correct[nulled_best] if gold_answer is not None else None,
                "candidates": [
                    {
                        "response": response,
                        "final_answer": extract_final_answer(response),
                        "baseline_score": baseline_score,
                        "nulled_score": nulled_score,
                        "is_correct": candidate_is_correct if gold_answer is not None else None,
                    }
                    for response, baseline_score, nulled_score, candidate_is_correct in zip(candidates, baseline_scores, nulled_scores, candidate_correct)
                ],
            }
        )

    summary = {
        "num_prompts": len(records_out),
        "n": args.n,
        "policy_model": args.policy_model,
        "reward_model": args.reward_model,
        "probe_file": args.probe_file,
        "alpha": args.alpha,
        "selection_overlap_rate": overlap_count / max(len(records_out), 1),
        "baseline_selected_baseline_score_mean": sum(baseline_selected_scores) / max(len(baseline_selected_scores), 1),
        "baseline_selected_nulled_score_mean": sum(cross_baseline_scores) / max(len(cross_baseline_scores), 1),
        "nulled_selected_nulled_score_mean": sum(nulled_selected_scores) / max(len(nulled_selected_scores), 1),
        "nulled_selected_baseline_score_mean": sum(cross_nulled_scores) / max(len(cross_nulled_scores), 1),
    }
    if prompt_records and isinstance(prompt_records[0], dict) and args.answer_key in prompt_records[0]:
        summary["baseline_accuracy"] = baseline_correct / max(len(records_out), 1)
        summary["nulled_accuracy"] = nulled_correct / max(len(records_out), 1)
        summary["oracle_accuracy"] = oracle_correct / max(len(records_out), 1)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for record in records_out:
            f.write(json.dumps(record) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
