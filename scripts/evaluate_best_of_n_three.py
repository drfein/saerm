#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nb.downstream.gsm8k import extract_final_answer, is_correct
from src.nb.downstream.scoring import RewardModelScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate best-of-n selection with baseline, probe-nulled, and length-penalized reward scoring.")
    parser.add_argument("--policy-model", required=True, help="Policy model path or HF ID")
    parser.add_argument("--reward-model", required=True, help="Reward model path or HF ID")
    parser.add_argument("--probe-file", help="Optional probe tensor for nulled scoring")
    parser.add_argument("--alpha", type=float, default=1.0, help="Nulling strength")
    parser.add_argument("--length-penalty-max-len", type=float, default=256.0)
    parser.add_argument("--length-penalty-scale", type=float, default=1.0)
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
    parser.add_argument("--prompt-batch-size", type=int, default=16, help="Prompt batch size for vLLM generation")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for batched candidate generation")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
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


def _chat_available(tokenizer: Any) -> bool:
    return hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None


def _render_prompt_from_messages_fallback(messages: list[dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        role = "User" if msg.get("role") == "user" else "Assistant"
        parts.append(f"{role}: {str(msg.get('content', '')).strip()}")
    parts.append("Assistant:")
    return "\n".join(parts)


def render_prompt_text(
    tokenizer: Any,
    *,
    prompt_text: str,
    messages: list[dict[str, str]] | None,
) -> str:
    if messages:
        if _chat_available(tokenizer):
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        return _render_prompt_from_messages_fallback(messages)

    if _chat_available(tokenizer):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        )

    return f"User: {prompt_text}\nAssistant:"


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
    prompt_str = render_prompt_text(tokenizer, prompt_text=prompt_text, messages=messages)
    input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
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


def generate_candidates_vllm(
    *,
    llm: Any,
    prompt_texts: list[str],
    n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[list[str]]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompt_texts, sampling_params)
    candidate_groups: list[list[str]] = []
    for output in outputs:
        candidate_groups.append([candidate.text.strip() for candidate in output.outputs])
    return candidate_groups


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def quantile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[lo])
    frac = idx - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def length_penalize_scores(
    baseline_scores: list[float],
    response_words: list[int],
    *,
    max_len: float,
    scale: float,
) -> tuple[list[float], float]:
    if len(baseline_scores) <= 1:
        sigma = 0.0
    else:
        sigma = statistics.pstdev(baseline_scores)
    sigma *= scale
    adjusted = [
        score + ((1.0 - (words / max_len)) * sigma)
        for score, words in zip(baseline_scores, response_words)
    ]
    return adjusted, sigma


def evaluate_prompt_group(
    *,
    idx: int,
    prompt_text: str,
    gold_answer: Any,
    candidates: list[str],
    baseline_scores: list[float],
    nulled_scores: list[float],
    selector_names: tuple[str, ...],
    selected_baseline_scores: dict[str, list[float]],
    selected_nulled_scores: dict[str, list[float]],
    selected_lengths: dict[str, list[int]],
    selected_correct: dict[str, int],
    selection_overlap: dict[str, int],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    response_words = [len(response.split()) for response in candidates]
    length_penalty_scores, sigma = length_penalize_scores(
        baseline_scores,
        response_words,
        max_len=args.length_penalty_max_len,
        scale=args.length_penalty_scale,
    )

    best_index = {
        "baseline": max(range(len(candidates)), key=lambda i: baseline_scores[i]),
        "length_nulled": max(range(len(candidates)), key=lambda i: nulled_scores[i]),
        "length_penalty": max(range(len(candidates)), key=lambda i: length_penalty_scores[i]),
    }
    selection_overlap["baseline_vs_length_nulled"] += int(best_index["baseline"] == best_index["length_nulled"])
    selection_overlap["baseline_vs_length_penalty"] += int(best_index["baseline"] == best_index["length_penalty"])
    selection_overlap["length_nulled_vs_length_penalty"] += int(best_index["length_nulled"] == best_index["length_penalty"])

    candidate_correct = [is_correct(c, gold_answer) for c in candidates] if gold_answer is not None else [False] * len(candidates)
    oracle_increment = int(any(candidate_correct)) if gold_answer is not None else 0

    for selector in selector_names:
        chosen_idx = best_index[selector]
        selected_baseline_scores[selector].append(baseline_scores[chosen_idx])
        selected_nulled_scores[selector].append(nulled_scores[chosen_idx])
        selected_lengths[selector].append(response_words[chosen_idx])
        if gold_answer is not None:
            selected_correct[selector] += int(candidate_correct[chosen_idx])

    return (
        {
            "index": idx,
            "prompt": prompt_text,
            "gold_answer": gold_answer,
            "selector_best_index": best_index,
            "selector_best_response": {
                name: candidates[best_index[name]] for name in selector_names
            },
            "length_penalty_sigma": sigma,
            "candidates": [
                {
                    "response": response,
                    "final_answer": extract_final_answer(response),
                    "baseline_score": baseline_score,
                    "nulled_score": nulled_score,
                    "length_penalty_score": penalty_score,
                    "response_words": response_word_count,
                    "is_correct": candidate_is_correct if gold_answer is not None else None,
                }
                for response, baseline_score, nulled_score, penalty_score, response_word_count, candidate_is_correct in zip(
                    candidates,
                    baseline_scores,
                    nulled_scores,
                    length_penalty_scores,
                    response_words,
                    candidate_correct,
                )
            ],
        },
        oracle_increment,
    )


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

    policy_model = None
    vllm_model = None
    if args.use_vllm:
        from vllm import LLM

        vllm_model = LLM(
            model=args.policy_model,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            trust_remote_code=args.trust_remote_code,
            dtype=args.policy_dtype,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
        )
    else:
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

    selector_names = ("baseline", "length_nulled", "length_penalty")
    selected_baseline_scores: dict[str, list[float]] = {name: [] for name in selector_names}
    selected_nulled_scores: dict[str, list[float]] = {name: [] for name in selector_names}
    selected_lengths: dict[str, list[int]] = {name: [] for name in selector_names}
    selected_correct: dict[str, int] = {name: 0 for name in selector_names}
    selection_overlap = {
        "baseline_vs_length_nulled": 0,
        "baseline_vs_length_penalty": 0,
        "length_nulled_vs_length_penalty": 0,
    }
    oracle_correct = 0
    records_out: list[dict[str, Any]] = []

    if args.use_vllm:
        for batch_start in tqdm(range(0, len(prompt_records), args.prompt_batch_size), desc="BoN prompt batches"):
            batch_records = prompt_records[batch_start : batch_start + args.prompt_batch_size]
            extracted = [
                (
                    batch_start + offset,
                    *extract_prompt(record, args.prompt_key, args.messages_key),
                    record.get(args.answer_key) if isinstance(record, dict) else None,
                )
                for offset, record in enumerate(batch_records)
            ]
            rendered_prompts = [
                render_prompt_text(policy_tokenizer, prompt_text=prompt_text, messages=messages)
                for _, prompt_text, messages, _ in extracted
            ]
            candidate_groups = generate_candidates_vllm(
                llm=vllm_model,
                prompt_texts=rendered_prompts,
                n=args.n,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )

            flat_prompts: list[str] = []
            flat_candidates: list[str] = []
            flat_histories: list[list[dict[str, str]] | None] = []
            group_sizes: list[int] = []
            for (_, prompt_text, messages, _), candidates in zip(extracted, candidate_groups):
                group_sizes.append(len(candidates))
                flat_prompts.extend([prompt_text] * len(candidates))
                flat_candidates.extend(candidates)
                flat_histories.extend([messages] * len(candidates))
            baseline_all, nulled_all = scorer.score_pairs_both(
                flat_prompts,
                flat_candidates,
                message_histories=flat_histories,
            )

            offset = 0
            for (idx, prompt_text, _messages, gold_answer), candidates, size in zip(extracted, candidate_groups, group_sizes):
                baseline_scores = baseline_all[offset : offset + size]
                nulled_scores = nulled_all[offset : offset + size]
                offset += size
                record_out, oracle_increment = evaluate_prompt_group(
                    idx=idx,
                    prompt_text=prompt_text,
                    gold_answer=gold_answer,
                    candidates=candidates,
                    baseline_scores=baseline_scores,
                    nulled_scores=nulled_scores,
                    selector_names=selector_names,
                    selected_baseline_scores=selected_baseline_scores,
                    selected_nulled_scores=selected_nulled_scores,
                    selected_lengths=selected_lengths,
                    selected_correct=selected_correct,
                    selection_overlap=selection_overlap,
                    args=args,
                )
                oracle_correct += oracle_increment
                records_out.append(record_out)
    else:
        for idx, record in enumerate(tqdm(prompt_records, desc="BoN prompts")):
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
            record_out, oracle_increment = evaluate_prompt_group(
                idx=idx,
                prompt_text=prompt_text,
                gold_answer=gold_answer,
                candidates=candidates,
                baseline_scores=baseline_scores,
                nulled_scores=nulled_scores,
                selector_names=selector_names,
                selected_baseline_scores=selected_baseline_scores,
                selected_nulled_scores=selected_nulled_scores,
                selected_lengths=selected_lengths,
                selected_correct=selected_correct,
                selection_overlap=selection_overlap,
                args=args,
            )
            oracle_correct += oracle_increment
            records_out.append(record_out)

    num_prompts = len(records_out)
    summary: dict[str, Any] = {
        "num_prompts": num_prompts,
        "n": args.n,
        "policy_model": args.policy_model,
        "reward_model": args.reward_model,
        "probe_file": args.probe_file,
        "alpha": args.alpha,
        "length_penalty_max_len": args.length_penalty_max_len,
        "length_penalty_scale": args.length_penalty_scale,
        "selection_overlap_rate": {
            name: count / max(num_prompts, 1)
            for name, count in selection_overlap.items()
        },
    }

    for selector in selector_names:
        summary[f"{selector}_selected_baseline_score_mean"] = mean(selected_baseline_scores[selector])
        summary[f"{selector}_selected_nulled_score_mean"] = mean(selected_nulled_scores[selector])
        summary[f"{selector}_selected_response_words_mean"] = mean([float(x) for x in selected_lengths[selector]])
        summary[f"{selector}_selected_response_words_stdev"] = (
            statistics.pstdev(selected_lengths[selector]) if selected_lengths[selector] else 0.0
        )
        summary[f"{selector}_selected_response_words_variance"] = (
            statistics.pvariance(selected_lengths[selector]) if selected_lengths[selector] else 0.0
        )
        summary[f"{selector}_selected_response_words_min"] = min(selected_lengths[selector]) if selected_lengths[selector] else 0
        summary[f"{selector}_selected_response_words_p25"] = quantile(selected_lengths[selector], 0.25)
        summary[f"{selector}_selected_response_words_median"] = quantile(selected_lengths[selector], 0.5)
        summary[f"{selector}_selected_response_words_p75"] = quantile(selected_lengths[selector], 0.75)
        summary[f"{selector}_selected_response_words_max"] = max(selected_lengths[selector]) if selected_lengths[selector] else 0

    if prompt_records and isinstance(prompt_records[0], dict) and args.answer_key in prompt_records[0]:
        for selector in selector_names:
            summary[f"{selector}_accuracy"] = selected_correct[selector] / max(num_prompts, 1)
        summary["oracle_accuracy"] = oracle_correct / max(num_prompts, 1)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for record in records_out:
            f.write(json.dumps(record) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
