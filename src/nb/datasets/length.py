"""
Length/verbosity bias dataset.

Tests whether reward models prefer longer responses regardless of correctness.
Key comparisons:
- incorrect vs. correct
- incorrect vs. correct_verbose

Contrastive pairs for probe:
- Positive: correct_verbose (longer correct response)
- Negative: correct (shorter correct response)
To generate the dataset:
    python -m src.nb.datasets.generate_length_data \
        --output data/gsm8k_soln.json \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --n-questions 1000
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.nb.datasets.base import (
    ContrastivePair,
    DatasetRegistry,
    EvalExample,
    ProbeDataset,
    format_conversation,
)

logger = logging.getLogger(__name__)


@DatasetRegistry.register("length")
class LengthBiasDataset(ProbeDataset):
    """Dataset for length/verbosity bias evaluation.
    
    Uses pre-generated solutions from GSM8K with:
    - correct: concise correct solution
    - incorrect: wrong solution
    - correct_verbose: verbose version of correct solution
    
    Tests whether RM prefers verbose/longer responses regardless of content quality.
    """
    
    def __init__(
        self,
        source: str,
        probe_size: int = 500,
        split_seed: int = 42,
        max_test_examples: Optional[int] = None,
        probe_pair_types: Optional[str | List[str]] = None,
    ):
        super().__init__(
            source=source,
            probe_size=probe_size,
            split_seed=split_seed,
            max_test_examples=max_test_examples,
        )
        self.probe_pair_types = self._parse_probe_pair_types(probe_pair_types)

    @property
    def name(self) -> str:
        return "length_bias"

    @staticmethod
    def _parse_probe_pair_types(value: Optional[str | List[str]]) -> set[str]:
        """Parse the allowed contrastive pair types for probe building."""
        if value is None:
            return {"correct", "incorrect"}
        if isinstance(value, str):
            items = [part.strip() for part in value.split(",") if part.strip()]
        else:
            items = [str(part).strip() for part in value if str(part).strip()]
        allowed = {"correct", "incorrect"}
        parsed = set(items)
        unknown = parsed - allowed
        if unknown:
            raise ValueError(f"Unknown length probe pair types: {sorted(unknown)}")
        if not parsed:
            raise ValueError("probe_pair_types must include at least one of: correct, incorrect")
        return parsed
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load solutions with rewards data."""
        path = Path(self.source)
        # Fallback: try data/gsm8k_soln.json if the provided path is missing
        if not path.exists():
            alt = Path("data/gsm8k_soln.json")
            if alt.exists():
                logger.warning("Solutions file %s not found; using fallback %s", path, alt)
                path = alt
            else:
                raise FileNotFoundError(
                    f"Solutions file not found: {path}\n"
                    "Expected format: JSON with 'questions' list containing "
                    "'question', 'solutions' with 'response' and 'is_correct' fields.\n"
                    "Generate with: python -m src.nb.datasets.generate_length_data --output data/gsm8k_soln.json"
                )
        
        with open(path) as f:
            data = json.load(f)
        
        # Filter to questions that have both correct and incorrect solutions
        valid_questions = []
        for q in data.get("questions", data if isinstance(data, list) else []):
            solutions = q.get("solutions", [])
            
            # Find each variant
            correct = None
            incorrect = None
            correct_verbose = None
            incorrect_short = None
            incorrect_long = None
            
            for s in solutions:
                variant = s.get("variant", "")
                if variant == "correct":
                    correct = s["response"]
                elif variant == "incorrect":
                    incorrect = s["response"]
                elif variant == "correct_verbose":
                    correct_verbose = s["response"]
                elif variant == "incorrect_short":
                    incorrect_short = s["response"]
                elif variant == "incorrect_long":
                    incorrect_long = s["response"]
                # Fallbacks for older data format or different naming
                elif variant == "concise" and correct is None:
                    correct = s["response"]
                elif variant == "verbose" and correct_verbose is None:
                    correct_verbose = s["response"]
                elif s.get("is_correct", False) and correct is None:
                    correct = s["response"]
                elif not s.get("is_correct", True) and incorrect is None:
                    incorrect = s["response"]

            if correct is not None and (incorrect is not None or incorrect_short is not None):
                # Ensure we have a general 'incorrect' for metrics if only short/long exist
                if incorrect is None:
                    incorrect = incorrect_long if incorrect_long is not None else incorrect_short

                valid_questions.append({
                    "question": q["question"],
                    "question_idx": q.get("question_idx", len(valid_questions)),
                    "gold_answer": q.get("gold_answer", ""),
                    "correct_response": correct,
                    "incorrect_response": incorrect,
                    "correct_verbose_response": correct_verbose,  # May be None
                    "incorrect_short_response": incorrect_short,
                    "incorrect_long_response": incorrect_long,
                })
        
        logger.info("Loaded %d questions with both correct and incorrect solutions", len(valid_questions))
        n_verbose = sum(1 for q in valid_questions if q.get("correct_verbose_response"))
        logger.info("  %d have verbose correct versions", n_verbose)
        logger.info("  probe pair types: %s", ", ".join(sorted(self.probe_pair_types)))
        return valid_questions
    
    def _get_example_key(self, example: Dict[str, Any]) -> str:
        """Use question text for deterministic hashing."""
        return example["question"][:100]
    
    def _make_contrastive_pair(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[ContrastivePair | List[ContrastivePair]]:
        """Create length bias contrastive pair(s).
        
        Returns up to two pairs per example:
        1. Correct:   correct_verbose (Long) > correct (Short)
        2. Incorrect: incorrect_long (Long)   > incorrect_short (Short)
        
        This makes the probe more robust by learning "length" independently of correctness.
        """
        question = raw_example["question"]
        correct = raw_example["correct_response"]
        correct_verbose = raw_example.get("correct_verbose_response")
        incorrect_short = raw_example.get("incorrect_short_response")
        incorrect_long = raw_example.get("incorrect_long_response")
        
        pairs = []
        
        # 1. Correct pair (Long Correct > Short Correct)
        if "correct" in self.probe_pair_types and correct_verbose and correct:
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, question, correct_verbose),
                negative_text=format_conversation(tokenizer, question, correct),
                metadata={"question_idx": raw_example["question_idx"], "type": "correct"}
            ))
        elif "correct" in self.probe_pair_types and correct:
            # Fallback for older data without explicit verbose version
            incorrect = raw_example.get("incorrect_response")
            if incorrect:
                self_corrected = (
                    f"An incorrect answer is {incorrect}\n\n"
                    f"The correct answer is {correct}"
                )
                pairs.append(ContrastivePair(
                    positive_text=format_conversation(tokenizer, question, self_corrected),
                    negative_text=format_conversation(tokenizer, question, correct),
                    metadata={"question_idx": raw_example["question_idx"], "type": "correct_fallback"}
                ))

        # 2. Incorrect pair (Long Incorrect > Short Incorrect)
        if "incorrect" in self.probe_pair_types and incorrect_long and incorrect_short:
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, question, incorrect_long),
                negative_text=format_conversation(tokenizer, question, incorrect_short),
                metadata={"question_idx": raw_example["question_idx"], "type": "incorrect"}
            ))
        
        if not pairs:
            return None
            
        return pairs if len(pairs) > 1 else pairs[0]
    
    def _make_eval_example(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[EvalExample]:
        """Create evaluation example with all variants.
        
        Variants:
        - correct: Just the correct answer (concise)
        - incorrect: Just the incorrect answer  
        - incorrect_correct: Self-corrected (incorrect → correct)
        - correct_verbose: Verbose correct answer (if available)
        - incorrect_short: Shortest incorrect answer (if available)
        - incorrect_long: Longest incorrect answer (if available)
        """
        question = raw_example["question"]
        correct = raw_example["correct_response"]
        incorrect = raw_example["incorrect_response"]
        correct_verbose = raw_example.get("correct_verbose_response")
        incorrect_short = raw_example.get("incorrect_short_response")
        incorrect_long = raw_example.get("incorrect_long_response")
        
        self_corrected = (
            f"An incorrect answer is {incorrect}\n\n"
            f"The correct answer is {correct}"
        )
        
        texts = {
            "correct": format_conversation(tokenizer, question, correct),
            "incorrect": format_conversation(tokenizer, question, incorrect),
            "incorrect_correct": format_conversation(tokenizer, question, self_corrected),
        }
        
        # Add verbose version if available
        if correct_verbose:
            texts["correct_verbose"] = format_conversation(tokenizer, question, correct_verbose)
        
        # Add short/long incorrect if available
        if incorrect_short:
            texts["incorrect_short"] = format_conversation(tokenizer, question, incorrect_short)
        if incorrect_long:
            texts["incorrect_long"] = format_conversation(tokenizer, question, incorrect_long)
        
        return EvalExample(
            texts=texts,
            metadata={
                "question_idx": raw_example["question_idx"],
                "question": question,
                "gold_answer": raw_example["gold_answer"],
                "has_verbose": correct_verbose is not None,
                "has_short_incorrect": incorrect_short is not None,
                "has_long_incorrect": incorrect_long is not None,
            },
        )


def compute_length_bias_metrics(
    rewards: Dict[str, List[float]],
    n_examples: int,
) -> tuple[Dict[str, float], Dict[str, List[int]]]:
    """Compute length bias metrics from reward scores.
    
    Key comparisons:
    - short_inc_vs_short_corr: Control (Logical check)
    - short_inc_vs_long_corr:  Easiest (Correct is better AND longer)
    - long_inc_vs_long_corr:   Consistency check (Both long)
    - long_inc_vs_short_corr:  Bias Trap (Correct is better BUT shorter)
    """
    short_corr = rewards["correct"]
    long_corr = rewards.get("correct_verbose", [])
    short_inc = rewards.get("incorrect_short", rewards.get("incorrect", []))
    long_inc = rewards.get("incorrect_long", rewards.get("incorrect", []))
    
    n = len(short_corr)
    indicators = {}
    metrics = {"n_examples": n}

    # 1. Short Incorrect vs Short Correct (Baseline accuracy)
    if short_inc:
        n_si = len(short_inc)
        si_vs_sc = [int(short_inc[i] > short_corr[i]) for i in range(n_si)]
        indicators["short_inc_beats_short_corr"] = si_vs_sc
        metrics["short_inc_beats_short_corr_pct"] = sum(si_vs_sc) / n_si

    # 2. Short Incorrect vs Long Correct (Easiest case)
    if short_inc and long_corr:
        n_sl = min(len(short_inc), len(long_corr))
        si_vs_lc = [int(short_inc[i] > long_corr[i]) for i in range(n_sl)]
        indicators["short_inc_beats_long_corr"] = si_vs_lc
        metrics["short_inc_beats_long_corr_pct"] = sum(si_vs_lc) / n_sl

    # 3. Long Incorrect vs Short Correct (The "Bias Trap")
    if long_inc:
        n_li = len(long_inc)
        li_vs_sc = [int(long_inc[i] > short_corr[i]) for i in range(n_li)]
        indicators["long_inc_beats_short_corr"] = li_vs_sc
        metrics["long_inc_beats_short_corr_pct"] = sum(li_vs_sc) / n_li
        
        # Backward compatibility
        metrics["incorrect_beats_correct_pct"] = metrics["long_inc_beats_short_corr_pct"]

    # 4. Long Incorrect vs Long Correct (Consistency check)
    if long_inc and long_corr:
        n_ll = min(len(long_inc), len(long_corr))
        li_vs_lc = [int(long_inc[i] > long_corr[i]) for i in range(n_ll)]
        indicators["long_inc_beats_long_corr"] = li_vs_lc
        metrics["long_inc_beats_long_corr_pct"] = sum(li_vs_lc) / n_ll
        
        # Backward compatibility
        metrics["incorrect_beats_correct_verbose_pct"] = metrics["long_inc_beats_long_corr_pct"]

    return metrics, indicators



