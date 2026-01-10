"""
BigBench dataset support for bias evaluation experiments.

Loads 18 curated MCQ tasks from tasksource/bigbench (from google/BIG-bench repository:
https://github.com/google/BIG-bench) and adapts them for:
- Sycophancy bias: Tests whether RM caves to user opinions
- Uncertainty bias: Tests whether RM handles uncertainty expressions

Uses HuggingFace dataset splits to prevent probe/eval overlap:
- Probe training: validation split from all tasks
- Evaluation: train split from all tasks
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

import torch

from src.nb.datasets.base import (
    ContrastivePair,
    DatasetRegistry,
    EvalExample,
    ProbeDataset,
    format_conversation,
)
from src.nb.datasets.mcq_parsing import (
    inject_user_opinion,
    make_sycophancy_response,
    format_mcq_prompt,
    format_mcq_response,
    POSITION_LABELS,
)

logger = logging.getLogger(__name__)

# Curated MCQ tasks from tasksource/bigbench (18 tasks)
# These are well-structured multiple-choice tasks with multiple_choice_targets
BIGBENCH_TASKS = [
    'bbq_lite_json',
    'code_line_description',
    'conceptual_combinations',
    'emoji_movie',
    'formal_fallacies_syllogisms_negation',
    'hindu_knowledge',
    'known_unknowns',
    'language_identification',
    'logic_grid_puzzle',
    'logical_deduction',
    'misconceptions_russian',
    'novel_concepts',
    'play_dialog_same_or_different',
    'strange_stories',
    'strategyqa',
    'symbol_interpretation',
    'vitaminc_fact_verification',
    'winowhy',
]


def parse_bigbench_mcq(
    input_text: str,
    targets: List[str],
    multiple_choice_targets: List[str],
    multiple_choice_scores: List[int],
) -> Optional[Dict[str, Any]]:
    """Parse BigBench MCQ format from tasksource/bigbench into standard structure.
    
    Args:
        input_text: Question/prompt text
        targets: List of correct answer(s)
        multiple_choice_targets: All possible answer choices
        multiple_choice_scores: Scores for each choice (1=correct, 0=incorrect)
    
    Returns:
        Dict with question, choices, correct_idx, or None if parsing fails
    """
    if not multiple_choice_targets or not multiple_choice_scores:
        return None
    
    if len(multiple_choice_targets) != len(multiple_choice_scores):
        return None
    
    # Find correct answer index
    correct_indices = [i for i, score in enumerate(multiple_choice_scores) if score == 1]
    if not correct_indices:
        return None
    
    # Use first correct answer
    correct_idx = correct_indices[0]
    
    return {
        "question": input_text.strip(),
        "choices": multiple_choice_targets,
        "correct_idx": correct_idx,
        "num_choices": len(multiple_choice_targets),
    }


def parse_bigbench_mcq_legacy(input_text: str, target: str) -> Optional[Dict[str, Any]]:
    """Legacy parser for BigBench MCQ format (for backwards compatibility).
    
    Handles multiple formats:
    1. Letter format: "Options:\n(A) ...\n(B) ..." with target "(A)"
    2. Dash format: "Options:\n- Yes\n- No" with target "Yes"
    3. No options: Just True/False or Yes/No answers
    
    Returns:
        Dict with question, choices, correct_idx, or None if parsing fails
    """
    target_clean = target.strip()
    
    # Check if has "Options:" section
    if "Options:" in input_text:
        parts = input_text.split("Options:")
        question = parts[0].strip()
        options_text = parts[1].strip()
        
        # Try letter format: (A), (B), (C), (D)
        letter_pattern = r'\(([A-D])\)\s*([^\n]+?)(?=\n\([A-D]\)|$)'
        letter_matches = re.findall(letter_pattern, options_text, re.DOTALL)
        
        if letter_matches and len(letter_matches) >= 2:
            # Letter format
            choices = {}
            for label, text in letter_matches:
                choices[label] = text.strip()
            
            # Clean target
            if target_clean.startswith("(") and target_clean.endswith(")"):
                target_clean = target_clean[1:-1].strip().upper()
            
            if target_clean not in choices:
                return None
            
            choice_labels = sorted(choices.keys())
            choice_list = [choices[label] for label in choice_labels]
            correct_idx = choice_labels.index(target_clean)
            
            return {
                "question": question,
                "choices": choice_list,
                "correct_idx": correct_idx,
                "num_choices": len(choice_list),
            }
        
        # Try dash format: - Yes\n- No
        dash_pattern = r'^-\s*(.+?)$'
        lines = options_text.split('\n')
        dash_choices = []
        for line in lines:
            match = re.match(dash_pattern, line.strip())
            if match:
                dash_choices.append(match.group(1).strip())
        
        if dash_choices and len(dash_choices) >= 2:
            # Find which choice matches target
            target_lower = target_clean.lower()
            for i, choice in enumerate(dash_choices):
                if choice.lower() == target_lower:
                    return {
                        "question": question,
                        "choices": dash_choices,
                        "correct_idx": i,
                        "num_choices": len(dash_choices),
                    }
            return None
    
    # No options section - infer from target (Yes/No, True/False, etc.)
    question = input_text.strip()
    target_lower = target_clean.lower()
    
    # Common binary choices
    if target_lower in ['yes', 'no']:
        choices = ['Yes', 'No']
    elif target_lower in ['true', 'false']:
        choices = ['True', 'False']
    elif target_lower in ['valid', 'invalid']:
        choices = ['valid', 'invalid']
    else:
        return None
    
    try:
        correct_idx = [c.lower() for c in choices].index(target_lower)
        return {
            "question": question,
            "choices": choices,
            "correct_idx": correct_idx,
            "num_choices": len(choices),
        }
    except ValueError:
        return None


def load_bigbench_data(
    tasks: Optional[List[str]] = None,
    max_per_task: Optional[int] = None,
    seed: int = 42,
    dataset_split: str = "train",  # HuggingFace split: "train" or "validation"
) -> List[Dict[str, Any]]:
    """Load data from BIG-bench tasks via tasksource/bigbench.
    
    Uses actual HuggingFace dataset splits to prevent probe/eval overlap:
    - validation split for probe training
    - train split for evaluation
    
    Args:
        tasks: List of task names to load (None = all 112 MCQ tasks)
        max_per_task: Maximum examples per task (None = all)
        seed: Random seed for sampling
        dataset_split: HuggingFace split to load ("train" or "validation")
        
    Returns:
        List of parsed examples with task metadata
    """
    if tasks is None:
        tasks = BIGBENCH_TASKS
    
    all_examples = []
    rng = random.Random(seed)
    
    for task_name in tasks:
        try:
            logger.info("Loading BigBench task: %s (split=%s)", task_name, dataset_split)
            ds = load_dataset('tasksource/bigbench', task_name, split=dataset_split)
            task_data = list(ds)
            
            # Sample if needed
            if max_per_task and len(task_data) > max_per_task:
                task_data = rng.sample(task_data, max_per_task)
            
            # Parse each example
            for idx, item in enumerate(task_data):
                parsed = parse_bigbench_mcq(
                    input_text=item['inputs'],
                    targets=item['targets'],
                    multiple_choice_targets=item['multiple_choice_targets'],
                    multiple_choice_scores=item['multiple_choice_scores'],
                )
                if parsed:
                    parsed['task_name'] = task_name
                    parsed['task_idx'] = idx
                    all_examples.append(parsed)
            
            logger.info("  Loaded %d examples from %s", 
                       len([e for e in all_examples if e['task_name'] == task_name]), 
                       task_name)
        except Exception as e:
            logger.warning("Failed to load task %s: %s", task_name, e)
    
    logger.info("Total BigBench examples loaded: %d (split=%s)", len(all_examples), dataset_split)
    return all_examples


# =============================================================================
# Sycophancy Bias
# =============================================================================


@DatasetRegistry.register("sycophancy_bigbench")
class SycophancyBigBenchDataset(ProbeDataset):
    """Sycophancy bias dataset from BigBench Hard.
    
    For probe training: samples from ALL tasks
    For evaluation: evaluates against ALL tasks
    
    Tests whether RM caves to user opinions by comparing accuracy under:
    - no_opinion: baseline (no user suggestion)
    - correct_opinion: user suggests correct answer
    - incorrect_opinion: user suggests wrong answer
    
    Probe contrasts agreeing vs disagreeing with user opinion:
    - Positive (sycophantic): agrees with user (says what user suggested)
    - Negative (non-sycophantic): disagrees with user (says something else)
    """
    
    def __init__(
        self,
        source: str = "tasksource/bigbench",
        probe_tasks: Optional[List[str]] = None,
        eval_tasks: Optional[List[str]] = None,
        max_per_task_probe: int = 100,  # Sample 100 from each task for probe
        max_per_task_eval: int = 500,  # Sample 500 from each task for eval
        **kwargs,
    ):
        """Initialize BigBench sycophancy dataset.
        
        Args:
            source: Dataset ID (default: tasksource/bigbench)
            probe_tasks: Tasks to sample for probe (None = all)
            eval_tasks: Tasks to evaluate on (None = all)
            max_per_task_probe: Max examples per task for probe
            max_per_task_eval: Max examples per task for eval
        """
        self.dataset_id = source
        self.probe_tasks = probe_tasks
        self.eval_tasks = eval_tasks
        self.max_per_task_probe = max_per_task_probe
        self.max_per_task_eval = max_per_task_eval
        super().__init__(source=source, **kwargs)
    
    @property
    def name(self) -> str:
        return "sycophancy_bigbench"
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load BigBench data for probe training (from validation split)."""
        return load_bigbench_data(
            tasks=self.probe_tasks,
            max_per_task=self.max_per_task_probe,
            seed=self.split_seed,
            dataset_split="validation",
        )
    
    def _load_eval_data(self) -> List[Dict[str, Any]]:
        """Load BigBench data for evaluation (from train split, no overlap with probe)."""
        return load_bigbench_data(
            tasks=self.eval_tasks,
            max_per_task=self.max_per_task_eval,
            seed=self.split_seed,
            dataset_split="train",
        )
    
    def _get_example_key(self, example: Dict[str, Any]) -> str:
        return f"{example['task_name']}_{example['task_idx']}"
    
    def _compute_splits(self) -> None:
        """Override to use ALL loaded data for probe (no secondary split).
        
        BigBench uses HuggingFace's validation split for probe and train split
        for evaluation, so we don't need an additional split here.
        """
        from collections import Counter
        n_total = len(self._raw_data)
        
        # Use ALL examples for probe - evaluation uses separate _load_eval_data()
        self._probe_indices = list(range(n_total))
        self._test_indices = []  # Not used - evaluation uses train split
        
        # Log per-task contributions
        task_counts = Counter(e.get("task_name", "unknown") for e in self._raw_data)
        logger.info("BigBench probe data: %d total examples from %d tasks", n_total, len(task_counts))
        for task, count in sorted(task_counts.items()):
            logger.info("  %s: %d examples", task, count)
    
    def _make_contrastive_pair(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[ContrastivePair]:
        return None  # Override get_probe_pairs instead
    
    def compute_baseline_correctness_mask(self, model: Any, tokenizer: Any, device: str = "cuda", batch_size: int = 8) -> List[bool]:
        """Compute which probe examples the baseline model gets correct.
        
        Args:
            model: The reward model to evaluate
            tokenizer: Tokenizer for the model
            device: Device to run evaluation on
            batch_size: Batch size for evaluation
            
        Returns:
            List of bools indicating which probe examples are correct at baseline
        """
        self._ensure_loaded()
        
        # Build texts for no-opinion evaluation of probe examples
        no_opinion_texts = []
        probe_examples_data = []
        
        for idx in self._probe_indices:
            example = self._raw_data[idx]
            question = example["question"]
            choices = example["choices"]
            correct_idx = example["correct_idx"]
            
            # Create texts for all choices with no opinion
            example_texts = []
            for choice in choices:
                response = make_sycophancy_response(choice)
                text = format_conversation(tokenizer, question, response)
                example_texts.append(text)
            
            no_opinion_texts.extend(example_texts)
            probe_examples_data.append({
                "idx": idx,
                "correct_idx": correct_idx,
                "n_choices": len(choices),
            })
        
        # Get rewards from baseline model
        from src.nb.nullbias.probe import get_rewards_with_nulling
        
        logger.info("Evaluating baseline correctness for %d probe examples", len(probe_examples_data))
        rewards = get_rewards_with_nulling(
            model, tokenizer, no_opinion_texts,
            probe=None,
            batch_size=batch_size,
            device=device,
            show_progress=True,
        )
        
        # Determine which examples are correct
        correctness_mask = []
        reward_idx = 0
        for example_data in probe_examples_data:
            n_choices = example_data["n_choices"]
            correct_idx = example_data["correct_idx"]
            
            example_rewards = rewards[reward_idx:reward_idx + n_choices]
            predicted_idx = example_rewards.argmax().item()
            
            is_correct = (predicted_idx == correct_idx)
            correctness_mask.append(is_correct)
            
            reward_idx += n_choices
        
        n_correct = sum(correctness_mask)
        logger.info(
            "Baseline gets %d/%d (%.1f%%) of probe examples correct",
            n_correct, len(correctness_mask), 100 * n_correct / len(correctness_mask)
        )
        
        return correctness_mask
    
    def get_probe_pairs(self, tokenizer: Any, baseline_correct_mask: Optional[List[bool]] = None) -> List[ContrastivePair]:
        """Create contrastive pairs: caving vs resisting wrong suggestions.
        
        Only uses questions where the RM gets correct at baseline (if mask provided).
        Only creates pairs for WRONG user suggestions to eliminate correctness confounding.
        
        Args:
            tokenizer: Tokenizer for formatting conversations
            baseline_correct_mask: Optional list of bools indicating which probe examples
                                   the baseline model gets correct.
        """
        self._ensure_loaded()
        
        pairs = []
        seen_questions = set()
        mask_idx = 0
        
        for idx in self._probe_indices:
            example = self._raw_data[idx]
            
            # Check if we should skip based on baseline correctness
            if baseline_correct_mask is not None:
                if mask_idx >= len(baseline_correct_mask) or not baseline_correct_mask[mask_idx]:
                    mask_idx += 1
                    continue
                mask_idx += 1
            
            question = example["question"]
            choices = example["choices"]
            correct_idx = example["correct_idx"]
            
            # Only use each question once
            q_key = self._get_example_key(example)
            if q_key in seen_questions:
                continue
            seen_questions.add(q_key)
            
            correct_answer = choices[correct_idx]
            
            # Pick a random incorrect answer
            incorrect_indices = [i for i in range(len(choices)) if i != correct_idx]
            if not incorrect_indices:
                continue
            incorrect_idx = random.Random(idx + self.split_seed).choice(incorrect_indices)
            incorrect_answer = choices[incorrect_idx]
            
            # Only create pairs for WRONG user suggestions
            prompt_wrong = inject_user_opinion(question, incorrect_answer)
            
            # Positive: caves to wrong suggestion
            response_cave = make_sycophancy_response(incorrect_answer)
            positive = format_conversation(tokenizer, prompt_wrong, response_cave)
            
            # Negative: resists wrong suggestion
            response_resist = make_sycophancy_response(correct_answer)
            negative = format_conversation(tokenizer, prompt_wrong, response_resist)
            
            pairs.append(ContrastivePair(
                positive_text=positive,
                negative_text=negative,
                metadata={"task": example["task_name"], "pair_type": "resist_wrong"},
            ))
            
        logger.info("Created %d contrastive pairs for BigBench sycophancy probe (RM-correct questions, wrong suggestions only)", len(pairs))
        return pairs

    def _make_eval_example(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[EvalExample]:
        """Create evaluation example for sycophancy.
        
        Each question is evaluated under THREE conditions:
        - no_opinion_choice_X: no user suggestion (baseline)
        - correct_opinion_choice_X: user suggests correct
        - incorrect_opinion_choice_X: user suggests wrong
        """
        question = raw_example["question"]
        choices = raw_example["choices"]
        correct_idx = raw_example["correct_idx"]
        
        # Pick a random incorrect answer for user suggestion
        incorrect_indices = [i for i in range(len(choices)) if i != correct_idx]
        if not incorrect_indices:
            return None
        
        rng = random.Random(raw_example["task_idx"] + self.split_seed)
        incorrect_idx = rng.choice(incorrect_indices)
        
        correct_answer = choices[correct_idx]
        incorrect_answer = choices[incorrect_idx]
        
        texts = {}
        
        # Condition 0: No user opinion (baseline)
        for i, choice in enumerate(choices):
            response = make_sycophancy_response(choice)
            texts[f"no_opinion_choice_{i}"] = format_conversation(tokenizer, question, response)
        
        # Condition 1: User suggests CORRECT answer
        prompt_correct_opinion = inject_user_opinion(question, correct_answer)
        for i, choice in enumerate(choices):
            response = make_sycophancy_response(choice)
            texts[f"correct_opinion_choice_{i}"] = format_conversation(
                tokenizer, prompt_correct_opinion, response
            )
        
        # Condition 2: User suggests INCORRECT answer
        prompt_incorrect_opinion = inject_user_opinion(question, incorrect_answer)
        for i, choice in enumerate(choices):
            response = make_sycophancy_response(choice)
            texts[f"incorrect_opinion_choice_{i}"] = format_conversation(
                tokenizer, prompt_incorrect_opinion, response
            )
        
        return EvalExample(
            texts=texts,
            metadata={
                "question": question,
                "correct_idx": correct_idx,
                "task_name": raw_example["task_name"],
                "num_choices": len(choices),
            },
        )
    
    def get_eval_examples(self, tokenizer: Any) -> List[EvalExample]:
        """Get evaluation examples from all BigBench tasks."""
        eval_data = self._load_eval_data()
        
        examples = []
        for raw_example in eval_data:
            example = self._make_eval_example(raw_example, tokenizer)
            if example is not None:
                examples.append(example)
        
        if self.max_test_examples is not None:
            examples = examples[:self.max_test_examples]
        
        logger.info("Created %d eval examples for BigBench sycophancy", len(examples))
        return examples


@DatasetRegistry.register("correctness_bigbench")
class CorrectnessBigBenchDataset(SycophancyBigBenchDataset):
    """Correctness probe dataset for BigBench: correct vs incorrect response.
    
    Uses plain questions (no user opinion) so the probe
    isolates response correctness without opinion confounds.
    
    Positive: correct response
    Negative: incorrect response
    """
    
    @property
    def name(self) -> str:
        return "correctness_bigbench"
    
    def get_probe_pairs(self, tokenizer: Any) -> List[ContrastivePair]:
        """Create contrastive pairs: correct vs incorrect response.
        
        Each question yields 1 pair with no user opinion:
        - Positive: correct response
        - Negative: incorrect response
        """
        self._ensure_loaded()
        
        pairs = []
        seen_questions = set()
        
        for idx in self._probe_indices:
            example = self._raw_data[idx]
            question = example["question"]
            choices = example["choices"]
            correct_idx = example["correct_idx"]
            
            q_key = self._get_example_key(example)
            if q_key in seen_questions:
                continue
            seen_questions.add(q_key)
            
            correct_answer = choices[correct_idx]
            
            # Pick a random incorrect answer
            incorrect_indices = [i for i in range(len(choices)) if i != correct_idx]
            if not incorrect_indices:
                continue
            incorrect_idx = random.Random(idx + self.split_seed).choice(incorrect_indices)
            incorrect_answer = choices[incorrect_idx]
            
            response_correct = make_sycophancy_response(correct_answer)
            response_incorrect = make_sycophancy_response(incorrect_answer)
            
            # Plain question, no user opinion
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, question, response_correct),
                negative_text=format_conversation(tokenizer, question, response_incorrect),
                metadata={"task": example["task_name"]},
            ))
        
        logger.info("Created %d correctness probe pairs for BigBench (no opinion)", len(pairs))
        return pairs


# =============================================================================
# Uncertainty Bias
# =============================================================================

UNCERTAINTY_PHRASES = [
    "I'm not entirely sure, but I think",
    "I believe, though I'm not certain,",
    "If I had to guess, I'd say",
    "I'm not 100% confident, but",
]


def add_uncertainty(response: str, seed: int = 42) -> str:
    """Add uncertainty phrase to response."""
    phrase = random.Random(seed).choice(UNCERTAINTY_PHRASES)
    # Extract the answer part
    if "The answer is" in response:
        answer = response.split("The answer is")[1].strip()
        return f"{phrase} the answer is {answer}"
    return f"{phrase} {response}"


@DatasetRegistry.register("uncertainty_bigbench")
class UncertaintyBigBenchDataset(ProbeDataset):
    """Uncertainty bias dataset from BigBench Hard.
    
    Tests whether RM handles uncertainty expressions appropriately.
    
    For each question, creates 4 response variants:
    - C: Direct correct
    - C+U: Hedged correct
    - I: Direct incorrect
    - I+U: Hedged incorrect
    
    Ideal ordering: C > C+U > I+U > I
    """
    
    def __init__(
        self,
        source: str = "tasksource/bigbench",
        probe_tasks: Optional[List[str]] = None,
        eval_tasks: Optional[List[str]] = None,
        max_per_task_probe: int = 100,
        max_per_task_eval: int = 500,
        **kwargs,
    ):
        self.dataset_id = source
        self.probe_tasks = probe_tasks
        self.eval_tasks = eval_tasks
        self.max_per_task_probe = max_per_task_probe
        self.max_per_task_eval = max_per_task_eval
        super().__init__(source=source, **kwargs)
    
    @property
    def name(self) -> str:
        return "uncertainty_bigbench"
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load BigBench data for probe training (from validation split)."""
        return load_bigbench_data(
            tasks=self.probe_tasks,
            max_per_task=self.max_per_task_probe,
            seed=self.split_seed,
            dataset_split="validation",
        )
    
    def _load_eval_data(self) -> List[Dict[str, Any]]:
        """Load BigBench data for evaluation (from train split, no overlap with probe)."""
        return load_bigbench_data(
            tasks=self.eval_tasks,
            max_per_task=self.max_per_task_eval,
            seed=self.split_seed,
            dataset_split="train",
        )
    
    def _get_example_key(self, example: Dict[str, Any]) -> str:
        return f"{example['task_name']}_{example['task_idx']}"
    
    def _compute_splits(self) -> None:
        """Override to use ALL loaded data for probe (no secondary split).
        
        BigBench uses HuggingFace's validation split for probe and train split
        for evaluation, so we don't need an additional split here.
        """
        from collections import Counter
        n_total = len(self._raw_data)
        
        # Use ALL examples for probe - evaluation uses separate _load_eval_data()
        self._probe_indices = list(range(n_total))
        self._test_indices = []  # Not used - evaluation uses train split
        
        # Log per-task contributions
        task_counts = Counter(e.get("task_name", "unknown") for e in self._raw_data)
        logger.info("BigBench probe data: %d total examples from %d tasks", n_total, len(task_counts))
        for task, count in sorted(task_counts.items()):
            logger.info("  %s: %d examples", task, count)
    
    def _make_contrastive_pair(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[ContrastivePair]:
        return None  # Override get_probe_pairs instead
    
    def get_probe_pairs(self, tokenizer: Any) -> List[ContrastivePair]:
        """Create contrastive pairs for uncertainty probe.
        
        Positive: Direct/confident response
        Negative: Hedged/uncertain response (same answer)
        """
        self._ensure_loaded()
        
        pairs = []
        
        for idx in self._probe_indices:
            example = self._raw_data[idx]
            question = example["question"]
            choices = example["choices"]
            correct_idx = example["correct_idx"]
            
            # Use correct answer
            correct_answer = choices[correct_idx]
            response_direct = make_sycophancy_response(correct_answer)
            response_hedged = add_uncertainty(response_direct, seed=idx + self.split_seed)
            
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, question, response_direct),
                negative_text=format_conversation(tokenizer, question, response_hedged),
                metadata={"task": example["task_name"], "response_type": "correct"},
            ))
            
            # Also create pair with incorrect answer
            incorrect_indices = [i for i in range(len(choices)) if i != correct_idx]
            if incorrect_indices:
                rng = random.Random(idx + self.split_seed)
                incorrect_idx = rng.choice(incorrect_indices)
                incorrect_answer = choices[incorrect_idx]
                response_direct_wrong = make_sycophancy_response(incorrect_answer)
                response_hedged_wrong = add_uncertainty(response_direct_wrong, seed=idx + self.split_seed + 1)
                
                pairs.append(ContrastivePair(
                    positive_text=format_conversation(tokenizer, question, response_direct_wrong),
                    negative_text=format_conversation(tokenizer, question, response_hedged_wrong),
                    metadata={"task": example["task_name"], "response_type": "incorrect"},
                ))
        
        logger.info("Created %d contrastive pairs for BigBench uncertainty probe", len(pairs))
        return pairs
    
    def _make_eval_example(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[EvalExample]:
        """Create evaluation example for uncertainty.
        
        Tests 4 response variants:
        - C: Direct correct
        - C_U: Hedged correct
        - I_low: Direct incorrect
        - I_U_low: Hedged incorrect
        """
        question = raw_example["question"]
        choices = raw_example["choices"]
        correct_idx = raw_example["correct_idx"]
        
        # Pick a random incorrect answer
        incorrect_indices = [i for i in range(len(choices)) if i != correct_idx]
        if not incorrect_indices:
            return None
        
        rng = random.Random(raw_example["task_idx"] + self.split_seed)
        incorrect_idx = rng.choice(incorrect_indices)
        
        correct_answer = choices[correct_idx]
        incorrect_answer = choices[incorrect_idx]
        
        # Create 4 response variants
        response_c = make_sycophancy_response(correct_answer)
        response_c_u = add_uncertainty(response_c, seed=raw_example["task_idx"])
        response_i = make_sycophancy_response(incorrect_answer)
        response_i_u = add_uncertainty(response_i, seed=raw_example["task_idx"] + 1)
        
        texts = {
            "C": format_conversation(tokenizer, question, response_c),
            "C_U": format_conversation(tokenizer, question, response_c_u),
            "I_low": format_conversation(tokenizer, question, response_i),
            "I_U_low": format_conversation(tokenizer, question, response_i_u),
        }
        
        return EvalExample(
            texts=texts,
            metadata={
                "question": question,
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
                "task_name": raw_example["task_name"],
            },
        )
    
    def get_eval_examples(self, tokenizer: Any) -> List[EvalExample]:
        """Get evaluation examples from all BigBench tasks."""
        eval_data = self._load_eval_data()
        
        examples = []
        for raw_example in eval_data:
            example = self._make_eval_example(raw_example, tokenizer)
            if example is not None:
                examples.append(example)
        
        if self.max_test_examples is not None:
            examples = examples[:self.max_test_examples]
        
        logger.info("Created %d eval examples for BigBench uncertainty", len(examples))
        return examples


# =============================================================================
# Position Bias
# =============================================================================


def format_mcq_conversation(tokenizer: Any, question: str, choices: List[str], answer_idx: int):
    """Format MCQ as a user/assistant conversation for reward models."""
    prompt = format_mcq_prompt(question, choices)
    response = format_mcq_response(answer_idx, choices)
    return format_conversation(tokenizer, prompt, response)


@DatasetRegistry.register("position_bigbench")
class PositionBigBenchDataset(ProbeDataset):
    """Position bias dataset from BigBench Hard.
    
    Tests whether RM prefers certain answer positions (A/B/C/D)
    regardless of content.
    
    For probe training: samples from ALL tasks
    For evaluation: evaluates against ALL tasks
    """
    
    def __init__(
        self,
        source: str = "tasksource/bigbench",
        probe_tasks: Optional[List[str]] = None,
        eval_tasks: Optional[List[str]] = None,
        max_per_task_probe: int = 100,
        max_per_task_eval: int = 500,
        **kwargs,
    ):
        self.dataset_id = source
        self.probe_tasks = probe_tasks
        self.eval_tasks = eval_tasks
        self.max_per_task_probe = max_per_task_probe
        self.max_per_task_eval = max_per_task_eval
        super().__init__(source=source, **kwargs)
    
    @property
    def name(self) -> str:
        return "position_bigbench"
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load BigBench data for probe training (from validation split)."""
        return load_bigbench_data(
            tasks=self.probe_tasks,
            max_per_task=self.max_per_task_probe,
            seed=self.split_seed,
            dataset_split="validation",
        )
    
    def _load_eval_data(self) -> List[Dict[str, Any]]:
        """Load BigBench data for evaluation (from train split, no overlap with probe)."""
        return load_bigbench_data(
            tasks=self.eval_tasks,
            max_per_task=self.max_per_task_eval,
            seed=self.split_seed,
            dataset_split="train",
        )
    
    def _get_example_key(self, example: Dict[str, Any]) -> str:
        return f"{example['task_name']}_{example['task_idx']}"
    
    def _compute_splits(self) -> None:
        """Override to use ALL loaded data for probe (no secondary split).
        
        BigBench uses HuggingFace's validation split for probe and train split
        for evaluation, so we don't need an additional split here.
        """
        from collections import Counter
        n_total = len(self._raw_data)
        
        # Use ALL examples for probe - evaluation uses separate _load_eval_data()
        self._probe_indices = list(range(n_total))
        self._test_indices = []  # Not used - evaluation uses train split
        
        # Log per-task contributions
        task_counts = Counter(e.get("task_name", "unknown") for e in self._raw_data)
        logger.info("BigBench probe data: %d total examples from %d tasks", n_total, len(task_counts))
        for task, count in sorted(task_counts.items()):
            logger.info("  %s: %d examples", task, count)
    
    def _make_contrastive_pair(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[ContrastivePair]:
        return None  # Override get_probe_pairs instead
    
    def get_probe_pairs(self, tokenizer: Any) -> List[ContrastivePair]:
        """Create position bias probe pairs.
        
        For each question and each choice, creates:
        - Positive: that choice at position A
        - Negative: that choice at positions B, C, D (averaged later)
        """
        self._ensure_loaded()
        
        pairs = []
        
        for idx in self._probe_indices:
            example = self._raw_data[idx]
            question = example["question"]
            choices = example["choices"]
            
            # For each choice content, create text at position A vs B/C/D
            for choice_idx, choice_content in enumerate(choices):
                # Get other choices
                other_choices = [c for i, c in enumerate(choices) if i != choice_idx]
                
                # Create shuffled choice lists with target at each position
                for target_pos in range(len(choices)):
                    shuffled = []
                    other_idx = 0
                    for pos in range(len(choices)):
                        if pos == target_pos:
                            shuffled.append(choice_content)
                        else:
                            if other_idx < len(other_choices):
                                shuffled.append(other_choices[other_idx])
                                other_idx += 1
                    
                    if len(shuffled) != len(choices):
                        continue
                    
                    text = format_mcq_conversation(tokenizer, question, shuffled, target_pos)
                    
                    # Position A is positive, others are negative
                    if target_pos == 0:
                        pairs.append(ContrastivePair(
                            positive_text=text,
                            negative_text="",  # Placeholder
                            metadata={"position": "A", "choice_idx": choice_idx, "task": example["task_name"]},
                        ))
                    else:
                        pairs.append(ContrastivePair(
                            positive_text="",  # Placeholder
                            negative_text=text,
                            metadata={"position": POSITION_LABELS[target_pos], "choice_idx": choice_idx, "task": example["task_name"]},
                        ))
        
        logger.info("Created %d position bias probe texts for BigBench", len(pairs))
        return pairs
    
    def get_position_embeddings_texts(self, tokenizer: Any) -> Dict[int, List[str]]:
        """Get texts organized by position for probe building.
        
        Returns:
            Dictionary mapping position index (0-3) to list of formatted texts
        """
        self._ensure_loaded()
        
        texts_by_pos = {0: [], 1: [], 2: [], 3: []}
        
        for idx in self._probe_indices:
            example = self._raw_data[idx]
            question = example["question"]
            choices = example["choices"]
            
            # Only use 4-choice questions for position bias
            if len(choices) != 4:
                continue
            
            for choice_idx, choice_content in enumerate(choices):
                other_choices = [c for i, c in enumerate(choices) if i != choice_idx]
                
                for target_pos in range(4):
                    shuffled = []
                    other_idx = 0
                    for pos in range(4):
                        if pos == target_pos:
                            shuffled.append(choice_content)
                        else:
                            shuffled.append(other_choices[other_idx])
                            other_idx += 1
                    
                    text = format_mcq_conversation(tokenizer, question, shuffled, target_pos)
                    texts_by_pos[target_pos].append(text)
        
        return texts_by_pos
    
    def _make_eval_example(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[EvalExample]:
        """Create evaluation example.
        
        For position bias, we shuffle choices so correct answer is at
        a balanced position across examples.
        """
        return None  # Override get_eval_examples instead
    
    def get_eval_examples(self, tokenizer: Any) -> List[EvalExample]:
        """Get evaluation examples with balanced position distribution.
        
        Shuffles each question so the correct answer rotates through
        positions A/B/C/D evenly.
        """
        eval_data = self._load_eval_data()
        
        # Filter to only 4-choice questions
        eval_data = [e for e in eval_data if len(e["choices"]) == 4]
        
        examples = []
        for idx, example in enumerate(eval_data):
            question = example["question"]
            choices = example["choices"]
            original_correct_idx = example["correct_idx"]
            
            # Balance positions: assign correct answer to position (idx % 4)
            target_correct_pos = idx % 4
            
            # Shuffle choices so correct ends up at target position
            correct_answer = choices[original_correct_idx]
            wrong_answers = [c for i, c in enumerate(choices) if i != original_correct_idx]
            
            # Use deterministic shuffle
            rng = random.Random(idx + self.split_seed)
            rng.shuffle(wrong_answers)
            
            shuffled_choices = []
            wrong_idx = 0
            for pos in range(4):
                if pos == target_correct_pos:
                    shuffled_choices.append(correct_answer)
                else:
                    shuffled_choices.append(wrong_answers[wrong_idx])
                    wrong_idx += 1
            
            # Create texts for each answer position
            texts = {}
            for pos in range(4):
                texts[POSITION_LABELS[pos]] = format_mcq_conversation(tokenizer, question, shuffled_choices, pos)
            
            examples.append(EvalExample(
                texts=texts,
                metadata={
                    "question_idx": idx,
                    "question": question,
                    "correct_position": POSITION_LABELS[target_correct_pos],
                    "correct_idx": target_correct_pos,
                    "shuffled_choices": shuffled_choices,
                    "task_name": example["task_name"],
                },
            ))
        
        if self.max_test_examples is not None:
            examples = examples[:self.max_test_examples]
        
        logger.info("Created %d eval examples for BigBench position bias", len(examples))
        return examples


@DatasetRegistry.register("correctness_position_bigbench")
class CorrectnessPositionBigBenchDataset(PositionBigBenchDataset):
    """Correctness probe dataset for BigBench position experiments.
    
    Builds contrastive pairs (correct vs incorrect) while balancing the *answer position*.
    This ensures the learned correctness direction is not confounded with position A/B/C/D.
    
    Positive: correct answer selected at a given position
    Negative: incorrect answer selected at the same position
    """
    
    @property
    def name(self) -> str:
        return "correctness_position_bigbench"
    
    def get_probe_pairs(self, tokenizer: Any) -> List[ContrastivePair]:
        self._ensure_loaded()
        
        pairs: List[ContrastivePair] = []
        for idx in self._probe_indices:
            example = self._raw_data[idx]
            question = example["question"]
            choices = example["choices"]
            correct_idx = example["correct_idx"]
            
            # Only use 4-choice questions
            if len(choices) != 4:
                continue
            
            # Balance across positions deterministically
            target_pos = idx % 4
            
            correct_answer = choices[correct_idx]
            wrong_answers = [c for i, c in enumerate(choices) if i != correct_idx]
            if not wrong_answers:
                continue
            
            # Deterministic shuffle for remaining slots
            rng = random.Random(idx + self.split_seed)
            rng.shuffle(wrong_answers)
            
            # Choose a primary wrong to act as the "incorrect selected answer"
            wrong_selected = wrong_answers[0]
            
            # Build "correct selected" choice list: correct at target_pos
            correct_choices: List[str] = []
            w_i = 0
            for pos in range(4):
                if pos == target_pos:
                    correct_choices.append(correct_answer)
                else:
                    correct_choices.append(wrong_answers[w_i])
                    w_i += 1
            
            # Build "incorrect selected" choice list
            incorrect_choices: List[str] = [None, None, None, None]  # type: ignore[list-item]
            incorrect_choices[target_pos] = wrong_selected
            correct_pos_other = (target_pos + 1) % 4
            incorrect_choices[correct_pos_other] = correct_answer
            
            # Fill remaining slots
            fill = list(wrong_answers[1:])
            while len(fill) < 2:
                for w in wrong_answers:
                    if w != wrong_selected:
                        fill.append(w)
                        if len(fill) >= 2:
                            break
            
            fill_i = 0
            for pos in range(4):
                if incorrect_choices[pos] is None:
                    incorrect_choices[pos] = fill[fill_i]
                    fill_i += 1
            
            # Format texts
            positive_text = format_mcq_conversation(tokenizer, question, correct_choices, target_pos)
            negative_text = format_mcq_conversation(tokenizer, question, incorrect_choices, target_pos)
            
            pairs.append(
                ContrastivePair(
                    positive_text=positive_text,
                    negative_text=negative_text,
                    metadata={"position": POSITION_LABELS[target_pos], "target_pos": target_pos, "task": example["task_name"]},
                )
            )
        
        logger.info("Created %d correctness probe pairs for BigBench position (balanced positions)", len(pairs))
        return pairs


# =============================================================================
# Position Freeform (BigBench)
# =============================================================================

def format_freeform_choices_prompt(question: str, choices: List[str]) -> str:
    """Format prompt with choices listed in natural language."""
    if len(choices) == 4:
        choices_text = f"The answer is either {choices[0]}, {choices[1]}, {choices[2]}, or {choices[3]}."
    else:
        choices_text = "The answer is either " + ", ".join(choices[:-1]) + f", or {choices[-1]}."
    return f"{question}\n\n{choices_text}"


def format_freeform_response(answer: str) -> str:
    """Format response for freeform position task."""
    return f"The answer is {answer}."


@DatasetRegistry.register("position_freeform_bigbench")
class PositionFreeformBigBenchDataset(ProbeDataset):
    """Position bias dataset from BigBench with freeform listing of choices.
    
    Tests whether RM accuracy changes based on where correct answer 
    appears in the list of options (first vs last).
    
    Prompt: "Question\n\nThe answer is either {a}, {b}, {c}, or {d}."
    Response: "The answer is {X}."
    
    Evaluates two orderings per question:
    - correct_first: correct answer listed first
    - correct_last: correct answer listed last
    """
    
    def __init__(
        self,
        source: str = "tasksource/bigbench",
        probe_tasks: Optional[List[str]] = None,
        eval_tasks: Optional[List[str]] = None,
        max_per_task_probe: int = 100,
        max_per_task_eval: int = 500,
        **kwargs,
    ):
        self.dataset_id = source
        self.probe_tasks = probe_tasks
        self.eval_tasks = eval_tasks
        self.max_per_task_probe = max_per_task_probe
        self.max_per_task_eval = max_per_task_eval
        super().__init__(source=source, **kwargs)
    
    @property
    def name(self) -> str:
        return "position_freeform_bigbench"
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load BigBench data for probe training (from validation split).
        
        Freeform position tests first-vs-last ordering, so any choice count works.
        """
        return load_bigbench_data(
            tasks=self.probe_tasks,
            max_per_task=self.max_per_task_probe,
            seed=self.split_seed,
            dataset_split="validation",
        )
    
    def _load_eval_data(self) -> List[Dict[str, Any]]:
        """Load BigBench data for evaluation (from train split, no overlap with probe).
        
        Freeform position tests first-vs-last ordering, so any choice count works.
        """
        return load_bigbench_data(
            tasks=self.eval_tasks,
            max_per_task=self.max_per_task_eval,
            seed=self.split_seed,
            dataset_split="train",
        )
    
    def _get_example_key(self, example: Dict[str, Any]) -> str:
        return f"{example['task_name']}_{example['task_idx']}"
    
    def _compute_splits(self) -> None:
        """Override to use ALL loaded data for probe (no secondary split).
        
        BigBench uses HuggingFace's validation split for probe and train split
        for evaluation, so we don't need an additional split here.
        """
        from collections import Counter
        n_total = len(self._raw_data)
        
        # Use ALL examples for probe - evaluation uses separate _load_eval_data()
        self._probe_indices = list(range(n_total))
        self._test_indices = []  # Not used - evaluation uses train split
        
        # Log per-task contributions
        task_counts = Counter(e.get("task_name", "unknown") for e in self._raw_data)
        logger.info("BigBench probe data: %d total examples from %d tasks", n_total, len(task_counts))
        for task, count in sorted(task_counts.items()):
            logger.info("  %s: %d examples", task, count)
    
    def _make_contrastive_pair(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[ContrastivePair]:
        return None  # Override get_probe_pairs instead
    
    def get_probe_pairs(self, tokenizer: Any) -> List[ContrastivePair]:
        """Create contrastive pairs based on where correct answer is listed.
        
        For each question, creates 2 pairs:
        1. Correct response: (correct-first) vs (correct-last)
        2. Incorrect response: (correct-first) vs (correct-last)
        
        This captures position bias on BOTH correct and incorrect responses.
        
        Positive: correct answer listed FIRST
        Negative: correct answer listed LAST
        """
        from src.nb.datasets.base import format_conversation
        
        self._ensure_loaded()
        
        pairs = []
        for idx in self._probe_indices:
            example = self._raw_data[idx]
            question = example["question"]
            choices = example["choices"]
            correct_idx = example["correct_idx"]
            correct_answer = choices[correct_idx]
            incorrect_choices = [c for i, c in enumerate(choices) if i != correct_idx]
            
            # Build prompts for both orderings
            choices_correct_first = [correct_answer] + incorrect_choices
            choices_correct_last = incorrect_choices + [correct_answer]
            prompt_first = format_freeform_choices_prompt(question, choices_correct_first)
            prompt_last = format_freeform_choices_prompt(question, choices_correct_last)
            
            # Pair 1: Correct response
            response_correct = format_freeform_response(correct_answer)
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, prompt_first, response_correct),
                negative_text=format_conversation(tokenizer, prompt_last, response_correct),
                metadata={"question_idx": idx, "response_type": "correct", "task": example["task_name"]},
            ))
            
            # Pair 2: Incorrect response (use first incorrect choice)
            if incorrect_choices:
                response_incorrect = format_freeform_response(incorrect_choices[0])
                pairs.append(ContrastivePair(
                    positive_text=format_conversation(tokenizer, prompt_first, response_incorrect),
                    negative_text=format_conversation(tokenizer, prompt_last, response_incorrect),
                    metadata={"question_idx": idx, "response_type": "incorrect", "task": example["task_name"]},
                ))
        
        logger.info("Created %d contrastive pairs for BigBench position freeform probe", len(pairs))
        return pairs
    
    def _make_eval_example(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[EvalExample]:
        """Create evaluation example with both orderings.
        
        For each question, tests two orderings:
        - correct_first: correct answer listed first
        - correct_last: correct answer listed last
        
        For each ordering, scores all 4 possible responses.
        """
        from src.nb.datasets.base import format_conversation
        
        question = raw_example["question"]
        choices = raw_example["choices"]
        correct_idx = raw_example["correct_idx"]
        correct_answer = choices[correct_idx]
        incorrect_choices = [c for i, c in enumerate(choices) if i != correct_idx]
        
        # Ordering 1: correct answer first
        choices_correct_first = [correct_answer] + incorrect_choices
        prompt_first = format_freeform_choices_prompt(question, choices_correct_first)
        
        # Ordering 2: correct answer last
        choices_correct_last = incorrect_choices + [correct_answer]
        prompt_last = format_freeform_choices_prompt(question, choices_correct_last)
        
        texts = {}
        
        # Score all 4 responses for correct-first ordering
        for i, choice in enumerate(choices_correct_first):
            response = format_freeform_response(choice)
            texts[f"correct_first_choice_{i}"] = format_conversation(tokenizer, prompt_first, response)
        
        # Score all 4 responses for correct-last ordering
        for i, choice in enumerate(choices_correct_last):
            response = format_freeform_response(choice)
            texts[f"correct_last_choice_{i}"] = format_conversation(tokenizer, prompt_last, response)
        
        return EvalExample(
            texts=texts,
            metadata={
                "question_idx": raw_example["task_idx"],
                "question": question,
                "choices_correct_first": choices_correct_first,
                "choices_correct_last": choices_correct_last,
                "correct_idx_first": 0,  # Correct is always at index 0 in correct_first
                "correct_idx_last": 3,   # Correct is always at index 3 in correct_last
                "task_name": raw_example["task_name"],
            },
        )
    
    def get_eval_examples(self, tokenizer: Any) -> List[EvalExample]:
        """Get evaluation examples."""
        eval_data = self._load_eval_data()
        
        examples = []
        for example in eval_data:
            eval_ex = self._make_eval_example(example, tokenizer)
            if eval_ex is not None:
                examples.append(eval_ex)
        
        if self.max_test_examples is not None:
            examples = examples[:self.max_test_examples]
        
        logger.info("Created %d eval examples for BigBench position freeform", len(examples))
        return examples


@DatasetRegistry.register("correctness_position_freeform_bigbench")
class CorrectnessPositionFreeformBigBenchDataset(PositionFreeformBigBenchDataset):
    """Correctness probe dataset for BigBench position freeform experiments.
    
    Builds contrastive pairs (correct vs incorrect response) while keeping
    position (correct-first vs correct-last) balanced.
    
    Positive: correct answer selected
    Negative: incorrect answer selected
    """
    
    @property
    def name(self) -> str:
        return "correctness_position_freeform_bigbench"
    
    def get_probe_pairs(self, tokenizer: Any) -> List[ContrastivePair]:
        """Create correctness probe pairs balanced across positions."""
        from src.nb.datasets.base import format_conversation
        
        self._ensure_loaded()
        
        pairs = []
        for idx in self._probe_indices:
            example = self._raw_data[idx]
            question = example["question"]
            choices = example["choices"]
            correct_idx = example["correct_idx"]
            correct_answer = choices[correct_idx]
            incorrect_choices = [c for i, c in enumerate(choices) if i != correct_idx]
            
            if not incorrect_choices:
                continue
            
            # Balance positions: alternate between correct-first and correct-last
            use_correct_first = (idx % 2 == 0)
            
            if use_correct_first:
                ordered_choices = [correct_answer] + incorrect_choices
            else:
                ordered_choices = incorrect_choices + [correct_answer]
            
            prompt = format_freeform_choices_prompt(question, ordered_choices)
            
            # Positive: correct response, Negative: incorrect response
            positive_text = format_conversation(tokenizer, prompt, format_freeform_response(correct_answer))
            negative_text = format_conversation(tokenizer, prompt, format_freeform_response(incorrect_choices[0]))
            
            pairs.append(ContrastivePair(
                positive_text=positive_text,
                negative_text=negative_text,
                metadata={
                    "question_idx": idx,
                    "position": "first" if use_correct_first else "last",
                    "task": example["task_name"],
                },
            ))
        
        logger.info("Created %d correctness probe pairs for BigBench position freeform (balanced positions)", len(pairs))
        return pairs

