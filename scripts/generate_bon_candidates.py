#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_best_of_n_three import (
    extract_prompt,
    generate_candidates,
    generate_candidates_vllm,
    render_prompt_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and save BoN candidate pools without scoring.")
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--messages-key", default="messages")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--policy-dtype", default="bfloat16")
    parser.add_argument("--prompt-batch-size", type=int, default=16)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
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


def main() -> None:
    args = parse_args()
    all_records = load_prompt_records(args.prompts, args.split)
    records = all_records[args.offset : args.offset + args.num_prompts]
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.policy_model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = None
    llm = None
    if args.use_vllm:
        from vllm import LLM

        llm = LLM(
            model=args.policy_model,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            trust_remote_code=args.trust_remote_code,
            dtype=args.policy_dtype,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
        )
    else:
        from transformers import AutoModelForCausalLM
        import torch

        dtype = getattr(torch, args.policy_dtype) if hasattr(torch, args.policy_dtype) else torch.bfloat16
        policy_model = AutoModelForCausalLM.from_pretrained(
            args.policy_model,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=dtype,
        ).to(args.policy_device)
        policy_model.eval()

    with output_path.open("w", encoding="utf-8") as out_f:
        if args.use_vllm:
            for batch_start in tqdm(range(0, len(records), args.prompt_batch_size), desc="Generate prompt batches"):
                batch_records = records[batch_start : batch_start + args.prompt_batch_size]
                extracted = [
                    (
                        batch_start + offset,
                        *extract_prompt(record, args.prompt_key, args.messages_key),
                        record.get(args.answer_key) if isinstance(record, dict) else None,
                    )
                    for offset, record in enumerate(batch_records)
                ]
                rendered = [
                    render_prompt_text(tokenizer, prompt_text=prompt_text, messages=messages)
                    for _, prompt_text, messages, _ in extracted
                ]
                candidate_groups = generate_candidates_vllm(
                    llm=llm,
                    prompt_texts=rendered,
                    n=args.n,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )
                for (idx, prompt_text, _messages, gold_answer), candidates in zip(extracted, candidate_groups):
                    out_f.write(
                        json.dumps(
                            {
                                "index": idx,
                                "prompt": prompt_text,
                                "gold_answer": gold_answer,
                                "candidates": [{"response": candidate} for candidate in candidates],
                            }
                        )
                        + "\n"
                    )
                    out_f.flush()
        else:
            for idx, record in enumerate(tqdm(records, desc="Generate prompts")):
                prompt_text, messages = extract_prompt(record, args.prompt_key, args.messages_key)
                gold_answer = record.get(args.answer_key) if isinstance(record, dict) else None
                candidates = generate_candidates(
                    model=policy_model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    messages=messages,
                    n=args.n,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    device=args.policy_device,
                )
                out_f.write(
                    json.dumps(
                        {
                            "index": idx,
                            "prompt": prompt_text,
                            "gold_answer": gold_answer,
                            "candidates": [{"response": candidate} for candidate in candidates],
                        }
                    )
                    + "\n"
                )
                out_f.flush()


if __name__ == "__main__":
    main()
