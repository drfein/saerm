#!/usr/bin/env python3
"""
Evaluate OpenAssistant/reward-model-deberta-v3-large-v2 on
allenai/tulu-3-wildchat-reused-on-policy-8b.
for model-style sensitivity analysis.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple, Any

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ID_DEFAULT = "OpenAssistant/reward-model-deberta-v3-large-v2"
DATASET_ID_DEFAULT = "allenai/tulu-3-wildchat-reused-on-policy-8b"


Message = Dict[str, str]
Pair = Tuple[str, str]


def to_question_answer(messages: List[Message]) -> Pair:
    """Extract (question, answer) from a multi-turn chat."""
    if not messages:
        return ("", "")

    # Find final assistant message as answer (fallback: last message)
    answer_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            answer_idx = i
            break

    if answer_idx is None:
        # No assistant message found
        return (messages[-1].get("content", "") or "", "")

    answer = messages[answer_idx].get("content", "") or ""

    # Find the last user message before that as question
    q_idx = None
    for i in range(answer_idx - 1, -1, -1):
        if messages[i].get("role") == "user":
            q_idx = i
            break

    if q_idx is not None:
        question = messages[q_idx].get("content", "") or ""
        return (question, answer)

    # Fallback: use all prior turns as "question/context"
    question = "\n\n".join(
        f"{m.get('role','')}: {m.get('content','')}"
        for m in messages[:answer_idx]
    )
    return (question, answer)


def infer_device(args_device: str) -> torch.device:
    if args_device != "auto":
        return torch.device(args_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


def clamp_max_length(tokenizer: Any, requested: int) -> int:
    tok_max = getattr(tokenizer, "model_max_length", requested)
    # Some tokenizers report extremely large "model_max_length"; keep requested in that case.
    if tok_max is None or tok_max > 1_000_000:
        tok_max = requested
    return min(requested, tok_max)


@torch.inference_mode()
def compute_rewards(
    model,
    tokenizer,
    pairs: List[Pair],
    batch_size: int,
    device: torch.device,
    max_length: int,
) -> List[float]:
    model.eval()
    max_len = clamp_max_length(tokenizer, max_length)

    rewards: List[float] = []
    for start in tqdm(range(0, len(pairs), batch_size), desc="Computing rewards"):
        batch = pairs[start : start + batch_size]
        questions, answers = zip(*batch)

        inputs = tokenizer(
            list(questions),
            list(answers),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)

        outputs = model(**inputs)
        logits = outputs.logits

        # Expect 1 scalar per example; be robust to shapes [B], [B,1], [B,2]
        if logits.ndim == 1:
            scores = logits
        elif logits.ndim == 2 and logits.shape[1] == 1:
            scores = logits[:, 0]
        elif logits.ndim == 2 and logits.shape[1] == 2:
            # If a 2-class head appears, use "positive" logit conventionally at index 1
            scores = logits[:, 1]
        else:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

        rewards.extend(scores.detach().float().cpu().tolist())

    return rewards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=MODEL_ID_DEFAULT)
    ap.add_argument("--dataset", type=str, default=DATASET_ID_DEFAULT)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", type=str, default="auto", help='auto|cpu|cuda|cuda:0 etc.')
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out", type=str, default=None, help="Output CSV path (default: rewards_<model>.csv)")
    args = ap.parse_args()

    device = infer_device(args.device)
    dtype = infer_dtype(args.dtype)
    if device.type == "cpu":
        dtype = torch.float32  # safer on CPU

    print(f"Loading dataset: {args.dataset} [{args.split}]")
    ds = load_dataset(args.dataset, split=args.split)
    N = min(args.n, len(ds))
    print(f"Using N={N}")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    # Prepare (question, answer) pairs for chosen/rejected
    pairs: List[Pair] = []
    meta: List[Dict[str, Any]] = []

    print("Preparing examples...")
    for i in tqdm(range(N), desc="Preparing data"):
        row = ds[i]
        dataset_id = row["id"]

        chosen_pair = to_question_answer(row["chosen"])
        rejected_pair = to_question_answer(row["rejected"])

        pairs.append(chosen_pair)
        meta.append({"dataset_id": dataset_id, "which": "chosen"})
        pairs.append(rejected_pair)
        meta.append({"dataset_id": dataset_id, "which": "rejected"})

    print("Scoring...")
    rewards = compute_rewards(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        batch_size=args.batch_size,
        device=device,
        max_length=args.max_length,
    )

    # Build output dataframe
    records = []
    for m, r in zip(meta, rewards):
        records.append(
            {
                "model_id": args.model,
                "dataset_id": m["dataset_id"],
                "which": m["which"],
                "reward": float(r),
            }
        )

    df = pd.DataFrame.from_records(records)

    safe_name = args.model.replace("/", "__")
    out_path = args.out or f"rewards_{safe_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")

    print("\nSummary:")
    print(df.groupby("which")["reward"].describe())

    chosen = df[df["which"] == "chosen"].set_index("dataset_id")["reward"]
    rejected = df[df["which"] == "rejected"].set_index("dataset_id")["reward"]
    accuracy = (chosen > rejected).mean()
    print(f"\nAccuracy (chosen > rejected): {accuracy:.2%}")


if __name__ == "__main__":
    main()
