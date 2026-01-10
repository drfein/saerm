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
    
    @property
    def name(self) -> str:
        return "length_bias"
    
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
            
            for s in solutions:
                variant = s.get("variant", "")
                if variant == "correct":
                    correct = s["response"]
                elif variant == "incorrect":
                    incorrect = s["response"]
                elif variant == "correct_verbose":
                    correct_verbose = s["response"]
                elif s.get("is_correct", False) and correct is None:
                    correct = s["response"]
                elif not s.get("is_correct", True) and incorrect is None:
                    incorrect = s["response"]
            
            if correct and incorrect:
                valid_questions.append({
                    "question": q["question"],
                    "question_idx": q.get("question_idx", len(valid_questions)),
                    "gold_answer": q.get("gold_answer", ""),
                    "correct_response": correct,
                    "incorrect_response": incorrect,
                    "correct_verbose_response": correct_verbose,  # May be None
                })
        
        logger.info("Loaded %d questions with both correct and incorrect solutions", len(valid_questions))
        n_verbose = sum(1 for q in valid_questions if q.get("correct_verbose_response"))
        logger.info("  %d have verbose correct versions", n_verbose)
        return valid_questions
    
    def _get_example_key(self, example: Dict[str, Any]) -> str:
        """Use question text for deterministic hashing."""
        return example["question"][:100]
    
    def _make_contrastive_pair(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[ContrastivePair]:
        """Create length bias contrastive pair.
        
        Positive (longer): correct_verbose (verbose correct answer)
        Negative (shorter): correct (concise correct answer)
        
        If no verbose version available, falls back to self-correction format.
        """
        question = raw_example["question"]
        correct = raw_example["correct_response"]
        incorrect = raw_example["incorrect_response"]
        correct_verbose = raw_example.get("correct_verbose_response")
        
        if correct_verbose:
            # Use verbose version as the longer response
            positive_text = format_conversation(tokenizer, question, correct_verbose)
        else:
            # Fall back to self-corrected response (longer)
            self_corrected = (
                f"An incorrect answer is {incorrect}\n\n"
                f"The correct answer is {correct}"
            )
            positive_text = format_conversation(tokenizer, question, self_corrected)
        
        negative_text = format_conversation(tokenizer, question, correct)
        
        return ContrastivePair(
            positive_text=positive_text,
            negative_text=negative_text,
            metadata={"question_idx": raw_example["question_idx"]},
        )
    
    def _make_eval_example(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[EvalExample]:
        """Create evaluation example with all variants.
        
        Variants:
        - correct: Just the correct answer (concise)
        - incorrect: Just the incorrect answer  
        - incorrect_correct: Self-corrected (incorrect → correct)
        - correct_verbose: Verbose correct answer (if available)
        """
        question = raw_example["question"]
        correct = raw_example["correct_response"]
        incorrect = raw_example["incorrect_response"]
        correct_verbose = raw_example.get("correct_verbose_response")
        
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
        
        return EvalExample(
            texts=texts,
            metadata={
                "question_idx": raw_example["question_idx"],
                "question": question,
                "gold_answer": raw_example["gold_answer"],
                "has_verbose": correct_verbose is not None,
            },
        )


def compute_length_bias_metrics(
    rewards: Dict[str, List[float]],
    n_examples: int,
) -> Dict[str, float]:
    """Compute length bias metrics from reward scores.
    
    Reports only the two comparisons we care about:
    - incorrect_beats_correct_pct: How often incorrect > correct
    - incorrect_beats_correct_verbose_pct: How often incorrect > verbose correct
    """
    correct_rewards = rewards["correct"]
    incorrect_rewards = rewards["incorrect"]
    verbose_rewards = rewards.get("correct_verbose", [])
    
    n = len(correct_rewards)
    
    incorrect_beats_correct = sum(
        1 for i in range(n) if incorrect_rewards[i] > correct_rewards[i]
    )
    
    metrics = {
        "incorrect_beats_correct_pct": incorrect_beats_correct / n,
        "n_examples": n,
    }
    
    if verbose_rewards:
        n_verbose = len(verbose_rewards)
        incorrect_beats_verbose = sum(
            1 for i in range(n_verbose) if incorrect_rewards[i] > verbose_rewards[i]
        )
        metrics["incorrect_beats_correct_verbose_pct"] = incorrect_beats_verbose / n_verbose
        metrics["n_verbose_examples"] = n_verbose
    
    return metrics




