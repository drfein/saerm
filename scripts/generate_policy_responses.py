#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses from a causal LM and save them as JSONL.")
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def load_rows(path: str, limit: int, offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < offset:
                continue
            if len(rows) >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_text_prompt(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt", "")).strip()
    if prompt:
        return prompt
    question = str(row.get("question", "")).strip()
    if question:
        return question
    raise ValueError("Row is missing prompt/question text")


def render_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = row.get("messages") if isinstance(row.get("messages"), list) else None
    prompt = resolve_text_prompt(row)
    if messages and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt


def resolve_local_tokenizer_source(model_name_or_path: str) -> str:
    path = Path(model_name_or_path)
    if not path.exists():
        return model_name_or_path
    search_root = path if path.is_dir() else path.parent
    tokenizer_markers = (
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "tokenizer.model",
    )
    for candidate in (search_root, *search_root.parents):
        if any((candidate / marker).exists() for marker in tokenizer_markers):
            return str(candidate)
    return str(search_root)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    rows = load_rows(args.eval_file, args.limit, args.offset)
    tokenizer_source = resolve_local_tokenizer_source(args.policy_model)
    local_files_only = Path(args.policy_model).exists() or Path(tokenizer_source).exists()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype_from_name(args.dtype),
        local_files_only=Path(args.policy_model).exists(),
    ).to(args.device)
    model.eval()

    generated_rows: list[dict[str, Any]] = []
    response_lengths: list[int] = []

    for start in tqdm(
        range(0, len(rows), args.batch_size),
        desc="Generating responses",
        disable=len(rows) <= args.batch_size,
    ):
        batch_rows = rows[start : start + args.batch_size]
        rendered_prompts = [render_prompt(tokenizer, row) for row in batch_rows]
        inputs = tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(args.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        continuation = outputs[:, inputs["input_ids"].shape[1] :]
        texts = [text.strip() for text in tokenizer.batch_decode(continuation, skip_special_tokens=True)]

        for row, response in zip(batch_rows, texts):
            prompt = resolve_text_prompt(row)
            response_words = len(response.split())
            response_lengths.append(response_words)
            generated_rows.append(
                {
                    "id": row.get("id", len(generated_rows)),
                    "prompt": prompt,
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "response": response,
                    "response_words": response_words,
                }
            )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "details.jsonl").open("w", encoding="utf-8") as f:
        for row in generated_rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "num_examples": len(generated_rows),
        "policy_model": args.policy_model,
        "offset": args.offset,
        "response_words_mean": sum(response_lengths) / max(len(response_lengths), 1),
        "response_words_min": min(response_lengths) if response_lengths else 0,
        "response_words_max": max(response_lengths) if response_lengths else 0,
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
