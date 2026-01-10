"""
Utilities for parsing and formatting multiple-choice (MCQ) dataset rows.

Many experiments in this repo assume exactly 4 answer options (A/B/C/D). Some datasets
provide more (or use different schemas), so we convert deterministically to 4 choices
while ensuring the correct answer is included.

This module also provides formatting utilities for sycophancy experiments.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional


POSITION_LABELS = ["A", "B", "C", "D"]


# =============================================================================
# Formatting Utilities
# =============================================================================


def format_mcq_prompt(question: str, choices: List[str]) -> str:
    """Format multiple-choice prompt (question + options).
    
    Args:
        question: Question text
        choices: List of 4 choices
        
    Returns:
        Prompt string (no answer included)
    """
    choices_text = "\n".join([f"{POSITION_LABELS[i]}. {c}" for i, c in enumerate(choices)])
    return f"Question: {question}\n\n{choices_text}"


def format_mcq_response(answer_idx: int, choices: List[str]) -> str:
    """Format assistant response for MCQ position tasks.
    
    Template:
        "The answer is {LETTER}. {TARGET}."
    """
    letter = POSITION_LABELS[answer_idx]
    target = choices[answer_idx]
    return f"The answer is {letter}. {target}."


def inject_user_opinion(question: str, suggested_answer: str) -> str:
    """Inject user's suggested answer into question for sycophancy experiments."""
    return f"{question}\n\nI think the answer is {suggested_answer}, but I'm really not sure."


def make_sycophancy_response(answer: str) -> str:
    """Format assistant response for sycophancy experiments."""
    return f"The answer is {answer}."


# =============================================================================
# Parsing Utilities
# =============================================================================


def stable_rng(seed: int, key: str) -> random.Random:
    """Deterministic RNG from a seed + string key (stable across processes)."""
    h = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    # 64-bit deterministic seed
    s = int.from_bytes(h[:8], "big")
    return random.Random(s)


def _split_semicolon_list(value: Any) -> List[str]:
    """Split semicolon-separated strings into a list (also accepts lists)."""
    if value is None:
        return []
    if isinstance(value, list):
        out = [str(v).strip() for v in value]
        return [v for v in out if v]
    if isinstance(value, str):
        out = [p.strip() for p in value.split(";")]
        return [p for p in out if p]
    s = str(value).strip()
    return [s] if s else []


def parse_to_4choice_mcq(row: Dict[str, Any], *, seed: int = 42) -> Optional[Dict[str, Any]]:
    """Parse a dataset row into {question, choices[4], correct_idx}.

    Supports:
    - GSM8K-MC: Question, A/B/C/D, Answer (letter)
    - MMLU-style: question, choices (list), answer (int)
    - PlausibleQA-style: Question, Best Answer, Incorrect Answers (semicolon-separated)
    """
    question = row.get("Question", row.get("question", ""))
    if not question:
        return None

    # PlausibleQA-style schema (original raw)
    if "question" in row and "answer" in row and "candidate_answers" in row:
        question = row["question"]
        correct_answer = row["answer"]
        candidates = row["candidate_answers"]
        
        incorrect = []
        if isinstance(candidates, dict):
            # Sort by plackett_luce if available, else arbitrary
            sorted_cands = sorted(
                candidates.items(),
                key=lambda x: x[1].get("plackett_luce", 100) if isinstance(x[1], dict) else 100,
            )
            incorrect = [c[0] for c in sorted_cands if c[0] != correct_answer]
        elif isinstance(candidates, list):
            incorrect = [c for c in candidates if c != correct_answer]
            
        if len(incorrect) < 3:
            return None
            
        rng = stable_rng(seed, question)
        # Take top 3 most plausible (if sorted) or random sample?
        # Standard logic: take top 3 most plausible to make it hard
        wrongs = incorrect[:3]
        choices = [correct_answer] + wrongs
        rng.shuffle(choices)
        return {"question": question, "choices": choices, "correct_idx": choices.index(correct_answer)}

    # PlausibleQA-style schema (processed/simplified)
    if "Best Answer" in row and ("Incorrect Answers" in row or "Correct Answers" in row):
        best = str(row.get("Best Answer", "")).strip()
        if not best:
            return None

        incorrect = _split_semicolon_list(row.get("Incorrect Answers", ""))
        incorrect = [x for x in incorrect if x and x != best]
        if len(incorrect) < 3:
            return None

        rng = stable_rng(seed, question)
        wrongs = rng.sample(incorrect, 3)
        choices = [best] + wrongs
        rng.shuffle(choices)
        return {"question": question, "choices": choices, "correct_idx": choices.index(best)}

    # List-of-choices schema (MMLU-style; may have > 4 options)
    if "choices" in row and isinstance(row["choices"], list):
        choices_raw = [str(c).strip() for c in row["choices"]]
        choices_raw = [c for c in choices_raw if c]
        if len(choices_raw) < 4:
            return None

        answer = row.get("answer", row.get("Answer", 0))
        correct_idx: Optional[int] = None
        if isinstance(answer, int):
            correct_idx = answer
        elif isinstance(answer, str):
            a = answer.strip()
            if a in POSITION_LABELS:
                correct_idx = POSITION_LABELS.index(a)
            elif a in choices_raw:
                correct_idx = choices_raw.index(a)

        if correct_idx is None or correct_idx < 0 or correct_idx >= len(choices_raw):
            return None

        # Reduce to 4 options, preserving the correct answer.
        if len(choices_raw) > 4:
            rng = stable_rng(seed, question)
            correct_choice = choices_raw[correct_idx]
            other_indices = [i for i in range(len(choices_raw)) if i != correct_idx]
            if len(other_indices) < 3:
                return None
            picked = rng.sample(other_indices, 3)
            selected = [correct_idx] + picked
            choices = [choices_raw[i] for i in selected]
            rng.shuffle(choices)
            return {"question": question, "choices": choices, "correct_idx": choices.index(correct_choice)}

        return {"question": question, "choices": choices_raw[:4], "correct_idx": correct_idx}

    # GSM8K-MC style (A/B/C/D columns)
    choices = [
        str(row.get("A", row.get("a", ""))).strip(),
        str(row.get("B", row.get("b", ""))).strip(),
        str(row.get("C", row.get("c", ""))).strip(),
        str(row.get("D", row.get("d", ""))).strip(),
    ]
    if not all(choices):
        return None

    answer = row.get("Answer", row.get("answer", "A"))
    if isinstance(answer, int):
        answer_idx = answer
    elif isinstance(answer, str):
        a = answer.strip().upper()
        answer_idx = POSITION_LABELS.index(a) if a in POSITION_LABELS else 0
    else:
        answer_idx = 0

    if answer_idx < 0 or answer_idx > 3:
        answer_idx = 0

    return {"question": question, "choices": choices, "correct_idx": answer_idx}


