from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


FINAL_ANSWER_PATTERNS = [
    re.compile(r"Final answer:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"####\s*([^\n]+)"),
    re.compile(r"(?:answer|result|solution)\s*(?:is|=|:)\s*(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE),
]

NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def build_gsm8k_prompt(question: str) -> str:
    return (
        "Solve the following grade-school math problem.\n"
        "Show your reasoning clearly.\n"
        "End with a final line exactly in the format: Final answer: <number>\n\n"
        f"Problem: {question.strip()}"
    )


def gsm8k_gold_answer(raw_answer: str) -> str:
    if "####" in raw_answer:
        return raw_answer.split("####")[-1].strip()
    return raw_answer.strip()


def render_gsm8k_completion(raw_answer: str) -> str:
    answer = raw_answer.strip()
    if "####" not in answer:
        return answer
    rationale, final = answer.rsplit("####", 1)
    rationale = rationale.strip()
    final = final.strip()
    if rationale:
        return f"{rationale}\nFinal answer: {final}"
    return f"Final answer: {final}"


def normalize_numeric_string(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.strip().replace(",", "")
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except Exception:
        return cleaned.strip().lower()
    if value.is_integer():
        return str(int(value))
    return f"{value:.12g}"


def extract_final_answer(text: str | None) -> str | None:
    if not text:
        return None
    for pattern in FINAL_ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    numbers = NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].strip()
    return None


def is_correct(prediction: str | None, gold_answer: str | None) -> bool:
    pred = normalize_numeric_string(extract_final_answer(prediction))
    gold = normalize_numeric_string(gold_answer)
    if pred is None or gold is None:
        return False
    return pred == gold


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    in_path = Path(path)
    with in_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def shuffle_rows(rows: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    rng = random.Random(seed)
    rng.shuffle(out)
    return out
