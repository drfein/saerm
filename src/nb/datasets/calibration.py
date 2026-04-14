"""
Calibration bias dataset using Math 500 with manipulated confidence scores.

Tests whether reward models exhibit confidence bias by:
1. Taking the same response text
2. Manually changing only the confidence score (e.g., "Confidence: 1" -> "Confidence: 10")
3. Testing if RM scores change based solely on verbalized confidence

Contrastive pairs for probe:
- Positive: Same response with high confidence (8-10)
- Negative: Same response with low confidence (1-3)

This cleanly isolates the confidence signal for probe learning.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.nb.datasets.base import (
    ContrastivePair,
    DatasetRegistry,
    EvalExample,
    ProbeDataset,
    format_conversation,
)

logger = logging.getLogger(__name__)


def make_prompt(question: str) -> str:
    """Format question as a prompt."""
    return (
        "Solve the following math problem step by step. "
        f"Problem: {question}\n\nSolution:"
    )


def replace_confidence(text: str, new_conf: int) -> str:
    """Replace confidence score in text with new value.
    
    Handles formats:
    - "Confidence: 7"
    - "**Confidence: 7**"
    - "Confidence: 10"
    etc.
    """
    # Pattern to match confidence statements
    pattern = r'(\*\*)?Confidence:\s*(\d+)(\*\*)?'
    
    # Check if confidence exists
    if not re.search(pattern, text):
        # Append confidence if missing
        return f"{text.rstrip()}\n\nConfidence: {new_conf}"
    
    # Replace with new confidence, preserving bold formatting
    def replacement(match):
        prefix = match.group(1) or ""
        suffix = match.group(3) or ""
        return f"{prefix}Confidence: {new_conf}{suffix}"
    
    return re.sub(pattern, replacement, text)


def extract_confidence(text: str) -> Optional[int]:
    """Extract confidence score from text."""
    pattern = r'Confidence:\s*(\d+)'
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    return None


@DatasetRegistry.register("calibration")
class CalibrationBiasDataset(ProbeDataset):
    """Dataset for calibration bias evaluation using Math 500.
    
    Creates probe pairs by taking the same response and only changing
    the confidence score. This cleanly isolates confidence bias.
    
    Probe training: Same response with high vs low confidence
    Evaluation: Multiple confidence levels (1, 3, 5, 7, 10) for both correct and incorrect
    """
    
    def __init__(
        self,
        source: str = "data/math500_uncertainty.json",
        probe_conf_high: List[int] = None,  # e.g., [10] or [8, 9, 10]
        probe_conf_low: List[int] = None,   # e.g., [1] or [1, 2, 3]
        eval_conf_levels: List[int] = None, # e.g., [1, 3, 5, 7, 10]
        min_rollouts_per_question: int = 5,
        **kwargs,
    ):
        """Initialize calibration dataset.
        
        Args:
            source: Path to Math 500 uncertainty JSON file
            probe_conf_high: Confidence levels for "high" probe training (default: [10])
            probe_conf_low: Confidence levels for "low" probe training (default: [1])
            eval_conf_levels: Confidence levels to evaluate (default: [1, 3, 5, 7, 10])
            min_rollouts_per_question: Min rollouts required to use a question
        """
        self.probe_conf_high = probe_conf_high or [10]
        self.probe_conf_low = probe_conf_low or [1]
        self.eval_conf_levels = eval_conf_levels or [1, 3, 5, 7, 10]
        self.min_rollouts = min_rollouts_per_question
        super().__init__(source, **kwargs)
    
    @property
    def name(self) -> str:
        return "calibration_bias"
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load Math 500 uncertainty data and prepare for manipulation."""
        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(
                f"Math 500 uncertainty file not found: {path}\n"
                "Generate with: python -m src.nb.datasets.generate_uncertainty_math"
            )
        
        with open(path) as f:
            data = json.load(f)
        
        valid_examples = []
        for item in data:
            rollouts = item.get("rollouts", [])
            
            # Need sufficient rollouts
            if len(rollouts) < self.min_rollouts:
                continue
            
            # Separate correct and incorrect rollouts
            correct_rollouts = [r for r in rollouts if r.get("is_correct", False)]
            incorrect_rollouts = [r for r in rollouts if not r.get("is_correct", True)]
            
            # Need at least one of each for useful examples
            if not correct_rollouts or not incorrect_rollouts:
                continue
            
            valid_examples.append({
                "question_idx": item["question_idx"],
                "problem": item["problem"],
                "gold_answer": item["gold_answer"],
                "correct_rollouts": correct_rollouts,
                "incorrect_rollouts": incorrect_rollouts,
            })
        
        logger.info("Loaded %d questions with both correct and incorrect rollouts", len(valid_examples))
        logger.info("  Probe high conf: %s, low conf: %s", self.probe_conf_high, self.probe_conf_low)
        logger.info("  Eval conf levels: %s", self.eval_conf_levels)
        return valid_examples
    
    def _get_example_key(self, example: Dict[str, Any]) -> str:
        """Use problem text for deterministic hashing."""
        return example["problem"][:100]
    
    def _make_contrastive_pair(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[ContrastivePair | List[ContrastivePair]]:
        """Create calibration contrastive pairs by manipulating confidence.
        
        Takes the same response and creates high vs low confidence versions.
        Returns up to 2 pairs per example (one from correct, one from incorrect).
        """
        problem = raw_example["problem"]
        prompt = make_prompt(problem)
        
        pairs = []
        
        # Pair 1: Same correct response with high vs low confidence
        if raw_example["correct_rollouts"]:
            response = np.random.choice(raw_example["correct_rollouts"])
            base_text = response["text"]
            
            # Create high confidence version
            high_conf = np.random.choice(self.probe_conf_high)
            high_text = replace_confidence(base_text, high_conf)
            
            # Create low confidence version
            low_conf = np.random.choice(self.probe_conf_low)
            low_text = replace_confidence(base_text, low_conf)
            
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, prompt, high_text),
                negative_text=format_conversation(tokenizer, prompt, low_text),
                metadata={
                    "variant": "correct",
                    "high_conf": high_conf,
                    "low_conf": low_conf,
                    "is_correct": True,
                },
            ))
        
        # Pair 2: Same incorrect response with high vs low confidence
        if raw_example["incorrect_rollouts"]:
            response = np.random.choice(raw_example["incorrect_rollouts"])
            base_text = response["text"]
            
            # Create high confidence version
            high_conf = np.random.choice(self.probe_conf_high)
            high_text = replace_confidence(base_text, high_conf)
            
            # Create low confidence version
            low_conf = np.random.choice(self.probe_conf_low)
            low_text = replace_confidence(base_text, low_conf)
            
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, prompt, high_text),
                negative_text=format_conversation(tokenizer, prompt, low_text),
                metadata={
                    "variant": "incorrect",
                    "high_conf": high_conf,
                    "low_conf": low_conf,
                    "is_correct": False,
                },
            ))
        
        return pairs if pairs else None
    
    def _make_eval_example(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[EvalExample]:
        """Create evaluation example with multiple confidence levels.
        
        For each confidence level in eval_conf_levels, creates:
        - C_X: Correct with confidence X
        - I_X: Incorrect with confidence X
        
        This allows testing calibration across the confidence spectrum.
        """
        problem = raw_example["problem"]
        prompt = make_prompt(problem)
        
        # Sample one correct and one incorrect response
        correct_response = np.random.choice(raw_example["correct_rollouts"])
        incorrect_response = np.random.choice(raw_example["incorrect_rollouts"])
        
        texts = {}
        metadata = {
            "problem": problem,
            "gold_answer": raw_example["gold_answer"],
            "correct_base_text": correct_response["text"],
            "incorrect_base_text": incorrect_response["text"],
        }
        
        # Create versions with different confidence levels
        for conf in self.eval_conf_levels:
            # Correct with this confidence
            correct_text = replace_confidence(correct_response["text"], conf)
            texts[f"C_{conf}"] = format_conversation(tokenizer, prompt, correct_text)
            
            # Incorrect with this confidence
            incorrect_text = replace_confidence(incorrect_response["text"], conf)
            texts[f"I_{conf}"] = format_conversation(tokenizer, prompt, incorrect_text)
        
        return EvalExample(texts=texts, metadata=metadata)
