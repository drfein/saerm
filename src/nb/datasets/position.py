"""
Position bias dataset for multiple-choice questions.

Tests whether reward models prefer certain answer positions (A/B/C/D)
regardless of content. Supports GSM8K-MC and MMLU datasets.

Contrastive pairs for probe:
- Positive: same content placed at position A
- Negative: same content placed at position B/C/D (average)

The probe direction encodes "answer is at position A".
Nulling this direction removes position bias.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from src.nb.datasets.base import (
    ContrastivePair,
    DatasetRegistry,
    EvalExample,
    ProbeDataset,
    format_conversation,
)
from src.nb.datasets.mcq_parsing import (
    parse_to_4choice_mcq,
    format_mcq_prompt,
    format_mcq_response,
    POSITION_LABELS,
)

logger = logging.getLogger(__name__)


def format_mcq_conversation(tokenizer: Any, question: str, choices: List[str], answer_idx: int):
    """Format MCQ as a user/assistant conversation for reward models."""
    prompt = format_mcq_prompt(question, choices)
    response = format_mcq_response(answer_idx, choices)
    return format_conversation(tokenizer, prompt, response)


def parse_mcq_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Backward-compatible wrapper around shared MCQ parsing.
    
    Note: Position experiments require exactly 4 options; we deterministically
    reduce datasets with >4 options (e.g., PlausibleQA) while preserving the correct answer.
    """
    return parse_to_4choice_mcq(row, seed=42)


@DatasetRegistry.register("position")
class PositionBiasDataset(ProbeDataset):
    """Dataset for position bias evaluation on MCQ.
    
    Supports GSM8K-MC and MMLU datasets with A/B/C/D multiple choice format.
    Tests whether RM prefers certain positions regardless of content.
    """
    
    def __init__(
        self,
        source: str = "guipenedo/gsm8k-mc",
        split: str = "train",
        eval_split: str = "test",
        probe_size: int = 500,
        split_seed: int = 42,
        max_test_examples: Optional[int] = None,
        subset: Optional[str] = None,
    ):
        """Initialize position bias dataset.
        
        Args:
            source: HuggingFace dataset ID (e.g., "guipenedo/gsm8k-mc", "cais/mmlu")
            split: Split to use for probe training (usually train)
            eval_split: Split to use for evaluation (usually test)
            probe_size: Number of examples for probe training
            split_seed: Seed for deterministic operations
            max_test_examples: Cap on test examples
            subset: Dataset subset/config name (e.g., "all" for MMLU)
        """
        super().__init__(source, probe_size, split_seed, max_test_examples)
        self.train_split = split
        self.eval_split = eval_split
        self.subset = subset
        self._eval_data: Optional[List[Dict[str, Any]]] = None
    
    @property
    def name(self) -> str:
        return "position_bias"
    
    def _load_dataset_split(self, split: str) -> Any:
        """Load a dataset split, handling MMLU's special format."""
        if self.subset:
            logger.info("Loading %s/%s (split=%s)", self.source, self.subset, split)
            return load_dataset(self.source, self.subset, split=split)
        else:
            logger.info("Loading %s (split=%s)", self.source, split)
            return load_dataset(self.source, split=split)
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load MCQ dataset for probe training.
        
        Only loads enough examples for probe building (probe_size * 2 for safety margin).
        """
        dataset = self._load_dataset_split(self.train_split)
        
        # Only need probe_size examples; load 2x for safety margin after filtering
        max_to_load = self.probe_size * 2
        
        examples = []
        for idx, row in enumerate(dataset):
            parsed = parse_mcq_row(row)
            if parsed:
                parsed["idx"] = idx
                examples.append(parsed)
                if len(examples) >= max_to_load:
                    break
        
        logger.info("Loaded %d MCQ examples for probe training", len(examples))
        return examples
    
    def _load_eval_data(self) -> List[Dict[str, Any]]:
        """Load MCQ dataset for evaluation."""
        if self._eval_data is not None:
            return self._eval_data
        
        dataset = self._load_dataset_split(self.eval_split)
        
        examples = []
        for idx, row in enumerate(dataset):
            parsed = parse_mcq_row(row)
            if parsed:
                parsed["idx"] = idx
                examples.append(parsed)
        
        if self.max_test_examples is not None:
            examples = examples[:self.max_test_examples]
        
        self._eval_data = examples
        logger.info("Loaded %d MCQ examples for evaluation", len(examples))
        return examples
    
    def _compute_splits(self) -> None:
        """Override to handle different splitting strategies based on eval_split.
        
        Two strategies:
        1. If eval_split == train_split: use hash-based split to prevent contamination
        2. If eval_split != train_split: use loaded data for probe (capped), eval is separate
        
        This prevents ANY data contamination between probe building and evaluation.
        """
        if self.train_split == self.eval_split:
            # Same split: must hash-split to avoid contamination
            super()._compute_splits()
            logger.info(
                "Position dataset: hash-split %d probe / %d test from same split (%s)",
                len(self._probe_indices), len(self._test_indices), self.train_split
            )
        else:
            # Different splits: use loaded data for probe (capped), eval is separate
            n_total = len(self._raw_data)
            actual_probe_size = min(self.probe_size, n_total)
            self._probe_indices = list(range(actual_probe_size))
            self._test_indices = []  # Not used - evaluation uses eval_split
            logger.info(
                "Position dataset: %d probe examples (capped at %d) from %s split, eval will use %s split (separate)",
                actual_probe_size, self.probe_size, self.train_split, self.eval_split
            )
    
    def _get_example_key(self, example: Dict[str, Any]) -> str:
        """Use question text for deterministic hashing."""
        return example["question"][:100]
    
    def _make_contrastive_pair(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[ContrastivePair]:
        """Create position bias contrastive pair.
        
        For each choice, creates formatted text with that choice at position A
        vs at other positions. The probe learns position A vs not-A.
        """
        # This is handled by get_probe_pairs for efficiency
        return None
    
    def get_probe_pairs(self, tokenizer: Any) -> List[ContrastivePair]:
        """Create position bias probe pairs.
        
        For each question and each choice, creates:
        - Positive: that choice at position A
        - Negative: that choice at positions B, C, D (averaged later)
        
        This is done by collecting embeddings, not text pairs.
        """
        self._ensure_loaded()
        
        # For position bias, we need to create formatted texts for each
        # choice at each position. The probe is built differently.
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
                    
                    # Position A is positive, others are negative
                    if target_pos == 0:
                        pairs.append(ContrastivePair(
                            positive_text=text,
                            negative_text="",  # Placeholder
                            metadata={"position": "A", "choice_idx": choice_idx},
                        ))
                    else:
                        pairs.append(ContrastivePair(
                            positive_text="",  # Placeholder
                            negative_text=text,
                            metadata={"position": POSITION_LABELS[target_pos], "choice_idx": choice_idx},
                        ))
        
        # Reorganize into proper pairs (A vs average of B,C,D)
        # For simplicity, we'll just return the texts and handle pairing in probe building
        logger.info("Created %d position bias probe texts", len(pairs))
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
        # This is handled by get_eval_examples
        return None
    
    def get_eval_examples(self, tokenizer: Any) -> List[EvalExample]:
        """Get evaluation examples with balanced position distribution.
        
        Shuffles each question so the correct answer rotates through
        positions A/B/C/D evenly.
        """
        eval_data = self._load_eval_data()
        
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
                },
            ))
        
        return examples


@DatasetRegistry.register("correctness_position_mcq")
class CorrectnessPositionMCQDataset(PositionBiasDataset):
    """Correctness probe dataset for MCQ position experiments.
    
    Builds contrastive pairs (correct vs incorrect) while balancing the *answer position*.
    This ensures the learned correctness direction is not confounded with position A/B/C/D.
    
    Positive: correct answer selected at a given position
    Negative: incorrect answer selected at the same position
    """
    
    @property
    def name(self) -> str:
        return "correctness_position_mcq"
    
    def get_probe_pairs(self, tokenizer: Any) -> List[ContrastivePair]:
        self._ensure_loaded()
        
        pairs: List[ContrastivePair] = []
        for idx in self._probe_indices:
            example = self._raw_data[idx]
            question = example["question"]
            choices = example["choices"]
            correct_idx = example["correct_idx"]
            
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
            remaining_wrongs = wrong_answers[1:]
            
            # Build "correct selected" choice list: correct at target_pos
            correct_choices: List[str] = []
            w_i = 0
            for pos in range(4):
                if pos == target_pos:
                    correct_choices.append(correct_answer)
                else:
                    correct_choices.append(wrong_answers[w_i])
                    w_i += 1
            
            # Build "incorrect selected" choice list:
            # - wrong_selected at target_pos (so answer position is identical)
            # - place the correct answer at a different position (next position) so prompt still contains it
            incorrect_choices: List[str] = [None, None, None, None]  # type: ignore[list-item]
            incorrect_choices[target_pos] = wrong_selected
            correct_pos_other = (target_pos + 1) % 4
            incorrect_choices[correct_pos_other] = correct_answer
            
            # Fill remaining slots with remaining wrongs (fall back to any wrongs if needed)
            fill = list(remaining_wrongs)
            # If we need more fillers, reuse other wrongs (excluding wrong_selected)
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
            
            # Format texts (MCQ as user/assistant conversation)
            pos_text = POSITION_LABELS[target_pos]
            positive_text = format_mcq_conversation(tokenizer, question, correct_choices, target_pos)
            negative_text = format_mcq_conversation(tokenizer, question, incorrect_choices, target_pos)
            
            pairs.append(
                ContrastivePair(
                    positive_text=positive_text,
                    negative_text=negative_text,
                    metadata={"position": pos_text, "target_pos": target_pos},
                )
            )
        
        logger.info("Created %d correctness probe pairs for MCQ position (balanced positions)", len(pairs))
        return pairs


def compute_position_metrics(
    rewards: Dict[str, List[float]],
    correct_positions: List[int],
) -> Dict[str, float]:
    """Compute position bias metrics.
    
    Args:
        rewards: Dictionary mapping position labels to reward lists
        correct_positions: List of correct position indices (0-3)
        
    Returns:
        Dictionary with metrics:
        - accuracy: Overall accuracy (highest reward = correct position)
        - accuracy_when_A/B/C/D: Accuracy when correct answer is at that position
        - position_selection: Distribution of selected positions
        - position_bias: Max deviation from uniform (25%)
    """
    n = len(correct_positions)
    
    # Count correct predictions and position selections
    correct_count = 0
    position_counts = [0, 0, 0, 0]
    
    # Track accuracy per correct position
    correct_at_pos = [0, 0, 0, 0]  # How many times correct is at each position
    correct_when_at_pos = [0, 0, 0, 0]  # How many times we got it right when correct is at that position
    
    for i in range(n):
        # Get rewards for each position
        pos_rewards = [rewards[POSITION_LABELS[p]][i] for p in range(4)]
        predicted = pos_rewards.index(max(pos_rewards))
        correct_pos = correct_positions[i]
        
        position_counts[predicted] += 1
        correct_at_pos[correct_pos] += 1
        
        if predicted == correct_pos:
            correct_count += 1
            correct_when_at_pos[correct_pos] += 1
    
    accuracy = correct_count / n
    position_dist = [c / n * 100 for c in position_counts]
    max_bias = max(abs(p - 25) for p in position_dist)
    
    # Compute accuracy per position (avoid division by zero)
    accuracy_per_pos = {}
    for pos in range(4):
        if correct_at_pos[pos] > 0:
            accuracy_per_pos[f"accuracy_when_{POSITION_LABELS[pos]}"] = correct_when_at_pos[pos] / correct_at_pos[pos]
            accuracy_per_pos[f"n_correct_at_{POSITION_LABELS[pos]}"] = correct_at_pos[pos]
        else:
            accuracy_per_pos[f"accuracy_when_{POSITION_LABELS[pos]}"] = 0.0
            accuracy_per_pos[f"n_correct_at_{POSITION_LABELS[pos]}"] = 0
    
    result = {
        "accuracy": accuracy,
        "position_distribution": position_dist,
        "position_A_pct": position_dist[0],
        "position_B_pct": position_dist[1],
        "position_C_pct": position_dist[2],
        "position_D_pct": position_dist[3],
        "max_position_bias": max_bias,
        "n_examples": n,
    }
    result.update(accuracy_per_pos)
    
    return result


def format_numbered_choices_prompt(question: str, choices: List[str]) -> str:
    """Format prompt with question and numbered answer options.
    
    Args:
        question: Question text
        choices: List of answer choices in order
        
    Returns:
        Prompt with question and numbered options
    """
    choices_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
    return f"{question}\n\n{choices_text}"


def format_position_response(answer: str) -> str:
    """Format response for position task.
    
    Args:
        answer: The answer to give
        
    Returns:
        Simple response with the answer
    """
    return f"The answer is {answer}."


@DatasetRegistry.register("position_plausibleqa")
class PositionPlausibleQADataset(ProbeDataset):
    """Position bias dataset adapted from PlausibleQA.
    
    Tests whether reward models prefer answers based on list position.
    Uses 5 choices: 1 correct + 4 most plausible incorrect.
    
    Prompt: Question + numbered list (1-5)
    Response: "The answer is {X}."
    
    Evaluates two orderings:
    - correct_first: correct answer at position 1
    - correct_last: correct answer at position 5
    
    Probe: correct-first vs correct-last (same response, different prompt ordering)
    """
    
    NUM_CHOICES = 5  # 1 correct + 4 incorrect
    
    def __init__(
        self,
        source: str,
        probe_size: int = 500,
        split_seed: int = 42,
        max_test_examples: Optional[int] = None,
    ):
        super().__init__(source, probe_size, split_seed, max_test_examples)
    
    @property
    def name(self) -> str:
        return "position_plausibleqa"
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load PlausibleQA and prepare for position bias testing."""
        import json
        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(f"PlausibleQA file not found: {path}")
        
        with open(path) as f:
            data = json.load(f)
        
        examples = []
        random.seed(self.split_seed)
        num_incorrect = self.NUM_CHOICES - 1  # 4 incorrect answers
        
        for idx, item in enumerate(data):
            question = item.get("question", "")
            correct_answer = item.get("answer", "")
            candidates = item.get("candidate_answers", {})
            
            if not question or not correct_answer or not candidates:
                continue
            
            # Get top N most plausible incorrect answers
            if isinstance(candidates, dict):
                sorted_candidates = sorted(
                    candidates.items(),
                    key=lambda x: x[1].get("plackett_luce", 100),
                )
                if len(sorted_candidates) < num_incorrect:
                    continue
                incorrect_answers = [c[0] for c in sorted_candidates[:num_incorrect]]
            elif isinstance(candidates, list):
                if len(candidates) < num_incorrect:
                    continue
                incorrect_answers = candidates[:num_incorrect]
            else:
                continue
            
            examples.append({
                "idx": idx,
                "question": question,
                "correct_answer": correct_answer,
                "incorrect_answers": incorrect_answers,
                "all_choices": [correct_answer] + incorrect_answers,
            })
        
        logger.info("Loaded %d PlausibleQA examples for position bias", len(examples))
        return examples
    
    def _get_example_key(self, example: Dict[str, Any]) -> str:
        return example["question"][:100]
    
    def _make_contrastive_pair(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[ContrastivePair]:
        return None  # Override get_probe_pairs instead
    
    def get_probe_pairs(self, tokenizer: Any) -> List[ContrastivePair]:
        """Create contrastive pairs based on where correct answer is listed.
        
        For each question, creates 2 pairs:
        1. Correct response: (correct-first prompt) vs (correct-last prompt)
        2. Incorrect response: (correct-first prompt) vs (correct-last prompt)
        
        Response stays constant, prompt position varies.
        This isolates position bias from correctness.
        
        Positive: correct answer listed FIRST (position 1)
        Negative: correct answer listed LAST (position 5)
        """
        from src.nb.datasets.base import format_conversation
        
        self._ensure_loaded()
        
        pairs = []
        for idx in self._probe_indices:
            example = self._raw_data[idx]
            question = example["question"]
            correct = example["correct_answer"]
            incorrect_answers = example["incorrect_answers"]
            
            # Two prompt orderings
            choices_correct_first = [correct] + incorrect_answers
            choices_correct_last = incorrect_answers + [correct]
            prompt_first = format_numbered_choices_prompt(question, choices_correct_first)
            prompt_last = format_numbered_choices_prompt(question, choices_correct_last)
            
            # Pair 1: Correct response (held constant)
            response_correct = format_position_response(correct)
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, prompt_first, response_correct),
                negative_text=format_conversation(tokenizer, prompt_last, response_correct),
                metadata={"question_idx": example["idx"], "response_type": "correct"},
            ))
            
            # Pair 2: Incorrect response (use first incorrect, held constant)
            response_incorrect = format_position_response(incorrect_answers[0])
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, prompt_first, response_incorrect),
                negative_text=format_conversation(tokenizer, prompt_last, response_incorrect),
                metadata={"question_idx": example["idx"], "response_type": "incorrect"},
            ))
        
        logger.info("Created %d contrastive pairs for position PlausibleQA probe", len(pairs))
        return pairs
    
    def _make_eval_example(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[EvalExample]:
        """Create evaluation example with both orderings.
        
        For each question, tests two orderings:
        - correct_first: correct answer at position 1
        - correct_last: correct answer at position 5
        
        For each ordering, scores all 5 responses.
        """
        from src.nb.datasets.base import format_conversation
        
        question = raw_example["question"]
        correct = raw_example["correct_answer"]
        incorrect_answers = raw_example["incorrect_answers"]
        
        # Two orderings
        choices_correct_first = [correct] + incorrect_answers
        choices_correct_last = incorrect_answers + [correct]
        prompt_first = format_numbered_choices_prompt(question, choices_correct_first)
        prompt_last = format_numbered_choices_prompt(question, choices_correct_last)
        
        texts = {}
        
        # Score all 5 responses for correct-first ordering
        for i, choice in enumerate(choices_correct_first):
            response = format_position_response(choice)
            texts[f"correct_first_choice_{i}"] = format_conversation(tokenizer, prompt_first, response)
        
        # Score all 5 responses for correct-last ordering
        for i, choice in enumerate(choices_correct_last):
            response = format_position_response(choice)
            texts[f"correct_last_choice_{i}"] = format_conversation(tokenizer, prompt_last, response)
        
        return EvalExample(
            texts=texts,
            metadata={
                "question_idx": raw_example["idx"],
                "question": question,
                "correct_answer": correct,
                # In correct_first: correct at index 0
                # In correct_last: correct at index 4
                "correct_first_idx": 0,
                "correct_last_idx": self.NUM_CHOICES - 1,
            },
        )


@DatasetRegistry.register("position_plausibleqa_mcq")
class PositionPlausibleQAMCQDataset(PositionBiasDataset):
    """PlausibleQA formatted as standard A/B/C/D multiple choice for position bias.
    
    Uses 1 correct answer + 3 most plausible incorrect answers (4 total).
    Tests position bias using the same A/B/C/D multiprobe approach as GSM8K/MMLU.
    """
    
    NUM_INCORRECT = 3  # 1 correct + 3 incorrect = 4 total choices
    
    def __init__(
        self,
        source: str,
        probe_size: int = 500,
        split_seed: int = 42,
        max_test_examples: Optional[int] = None,
    ):
        # Initialize with dummy values, we'll override loading
        self.plausible_source = source
        super().__init__(
            source="dummy",  # We'll override _load_dataset_split
            split="train",
            eval_split="test",
            probe_size=probe_size,
            split_seed=split_seed,
            max_test_examples=max_test_examples,
        )
    
    @property
    def name(self) -> str:
        return "position_plausibleqa_mcq"
    
    def _load_plausibleqa_as_mcq(self, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Helper to load PlausibleQA and format as 4-choice MCQ."""
        import json
        path = Path(self.plausible_source)
        if not path.exists():
            raise FileNotFoundError(f"PlausibleQA file not found: {path}")
        
        with open(path) as f:
            data = json.load(f)
        
        examples = []
        
        for idx, item in enumerate(data):
            question = item.get("question", "")
            correct_answer = item.get("answer", "")
            candidates = item.get("candidate_answers", {})
            
            if not question or not correct_answer or not candidates:
                continue
            
            # Get top NUM_INCORRECT most plausible incorrect answers
            if isinstance(candidates, dict):
                sorted_candidates = sorted(
                    candidates.items(),
                    key=lambda x: x[1].get("plackett_luce", 100),
                )
                if len(sorted_candidates) < self.NUM_INCORRECT:
                    continue
                incorrect_answers = [c[0] for c in sorted_candidates[:self.NUM_INCORRECT]]
            elif isinstance(candidates, list):
                if len(candidates) < self.NUM_INCORRECT:
                    continue
                incorrect_answers = candidates[:self.NUM_INCORRECT]
            else:
                continue
            
            # Create 4-choice MCQ (A/B/C/D)
            all_choices = [correct_answer] + incorrect_answers
            examples.append({
                "idx": idx,
                "question": question,
                "choices": all_choices,
                "correct_idx": 0,  # Correct is first in list, will be shuffled during eval
            })
            
            if max_examples and len(examples) >= max_examples:
                break
        
        return examples
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load PlausibleQA and format as 4-choice MCQ."""
        max_to_load = self.probe_size * 2
        examples = self._load_plausibleqa_as_mcq(max_examples=max_to_load)
        logger.info("Loaded %d PlausibleQA examples as 4-choice MCQ for position bias", len(examples))
        return examples
    
    def _load_eval_data(self) -> List[Dict[str, Any]]:
        """Load PlausibleQA eval data as 4-choice MCQ."""
        examples = self._load_plausibleqa_as_mcq(max_examples=self.max_test_examples)
        logger.info("Loaded %d PlausibleQA examples for MCQ evaluation", len(examples))
        return examples


@DatasetRegistry.register("correctness_position_plausibleqa_mcq")
class CorrectnessPositionPlausibleQAMCQDataset(PositionPlausibleQAMCQDataset):
    """Correctness probe dataset for PlausibleQA MCQ position experiments.
    
    Builds contrastive pairs (correct vs incorrect) while balancing the answer position.
    This ensures the learned correctness direction is not confounded with position A/B/C/D.
    """
    
    @property
    def name(self) -> str:
        return "correctness_position_plausibleqa_mcq"
    
    def get_probe_pairs(self, tokenizer: Any) -> List[ContrastivePair]:
        """Generate correctness probe pairs balanced across positions."""
        self._ensure_loaded()
        
        pairs: List[ContrastivePair] = []
        for idx in self._probe_indices:
            example = self._raw_data[idx]
            question = example["question"]
            choices = example["choices"]
            correct_idx = example["correct_idx"]
            
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
            remaining_wrongs = wrong_answers[1:]
            
            # Build "correct selected" choice list: correct at target_pos
            correct_choices: List[str] = []
            w_i = 0
            for pos in range(4):
                if pos == target_pos:
                    correct_choices.append(correct_answer)
                else:
                    correct_choices.append(wrong_answers[w_i])
                    w_i += 1
            
            # Build "incorrect selected" choice list:
            # - wrong_selected at target_pos (so answer position is identical)
            # - place the correct answer at a different position (next position) so prompt still contains it
            incorrect_choices: List[str] = [None, None, None, None]  # type: ignore[list-item]
            incorrect_choices[target_pos] = wrong_selected
            correct_pos_other = (target_pos + 1) % 4
            incorrect_choices[correct_pos_other] = correct_answer
            
            # Fill remaining slots with remaining wrongs
            fill = list(remaining_wrongs)
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
                    metadata={"position": POSITION_LABELS[target_pos], "target_pos": target_pos},
                )
            )
        
        logger.info("Created %d correctness probe pairs for PlausibleQA MCQ (balanced positions)", len(pairs))
        return pairs


def compute_binary_position_metrics(
    rewards: Dict[str, List[float]],
    n_examples: int,
    num_choices: int = 5,
) -> Dict[str, float]:
    """Compute position bias metrics for PlausibleQA (5 choices).
    
    For each question in each orientation, the RM picks the highest-reward response.
    We measure accuracy (did it pick the correct answer?).
    
    Args:
        rewards: Dictionary with reward lists for each ordering and choice
        n_examples: Number of examples
        num_choices: Number of answer choices (default 5)
        
    Returns:
        Dictionary with accuracy metrics for each orientation
    """
    n = n_examples
    
    # Accuracy when correct answer is listed FIRST (position 1)
    correct_first_accuracy = 0
    for i in range(n):
        # In correct_first ordering, correct answer is at index 0
        choice_rewards = [rewards[f"correct_first_choice_{j}"][i] for j in range(num_choices)]
        predicted = choice_rewards.index(max(choice_rewards))
        if predicted == 0:  # Correct is at index 0
            correct_first_accuracy += 1
    correct_first_accuracy /= n
    
    # Accuracy when correct answer is listed LAST (position 5)
    correct_last_accuracy = 0
    for i in range(n):
        # In correct_last ordering, correct answer is at last index
        choice_rewards = [rewards[f"correct_last_choice_{j}"][i] for j in range(num_choices)]
        predicted = choice_rewards.index(max(choice_rewards))
        if predicted == num_choices - 1:  # Correct is at last index
            correct_last_accuracy += 1
    correct_last_accuracy /= n
    
    overall_accuracy = (correct_first_accuracy + correct_last_accuracy) / 2
    accuracy_gap = correct_first_accuracy - correct_last_accuracy
    
    return {
        "accuracy": overall_accuracy,
        "accuracy_correct_first": correct_first_accuracy,
        "accuracy_correct_last": correct_last_accuracy,
        "accuracy_gap": accuracy_gap,
        "n_examples": n,
        "num_choices": num_choices,
    }


def format_freeform_choices_prompt(question: str, choices: List[str]) -> str:
    """Format prompt with choices listed in natural language.
    
    Args:
        question: Question text
        choices: List of answer choices in desired order
        
    Returns:
        Formatted prompt with choices listed
    """
    if len(choices) == 4:
        choices_text = f"The answer is either {choices[0]}, {choices[1]}, {choices[2]}, or {choices[3]}."
    else:
        choices_text = "The answer is either " + ", ".join(choices[:-1]) + f", or {choices[-1]}."
    return f"{question}\n\n{choices_text}"


def format_freeform_response(answer: str) -> str:
    """Format response for freeform position task."""
    return f"The answer is {answer}."


@DatasetRegistry.register("position_freeform")
class PositionFreeformDataset(ProbeDataset):
    """Position bias dataset with freeform listing of choices.
    
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
        source: str = "guipenedo/gsm8k-mc",
        split: str = "train",
        eval_split: str = "test",
        num_choices: int = 4,
        probe_size: int = 500,
        split_seed: int = 42,
        max_test_examples: Optional[int] = None,
        subset: Optional[str] = None,
    ):
        self._split = split
        self._eval_split = eval_split
        self._subset = subset
        self.num_choices = num_choices
        super().__init__(source, probe_size, split_seed, max_test_examples)
    
    @property
    def name(self) -> str:
        return "position_freeform"
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load GSM8K-MC dataset."""
        logger.info("Loading %s (split=%s)...", self.source, self._split)
        
        if self._subset:
            ds = load_dataset(self.source, self._subset, split=self._split)
        else:
            ds = load_dataset(self.source, split=self._split)
        
        examples = []
        rng = random.Random(self.split_seed)
        for idx, item in enumerate(ds):
            # Handle different dataset formats
            # Format 1: separate A, B, C, D columns (guipenedo/gsm8k-mc)
            if "A" in item and "B" in item and "C" in item and "D" in item:
                question = item.get("Question", item.get("question", ""))
                choices = [str(item["A"]), str(item["B"]), str(item["C"]), str(item["D"])]
                answer = item.get("Answer", item.get("answer", "A"))
                answer_idx = POSITION_LABELS.index(answer.upper()) if isinstance(answer, str) else answer
            # Format 2: choices list
            elif "choices" in item:
                question = item.get("question", "")
                choices = item.get("choices", [])
                answer_idx = item.get("answer_idx", item.get("answer", 0))
                if isinstance(answer_idx, str):
                    answer_idx = POSITION_LABELS.index(answer_idx.upper())
            else:
                continue
            
            if not question or len(choices) != self.num_choices:
                continue
            
            examples.append({
                "idx": idx,
                "question": question,
                "choices": choices,
                "correct_idx": answer_idx,
                "correct_answer": choices[answer_idx],
            })
        
        logger.info("Loaded %d examples from %s", len(examples), self.source)
        return examples
    
    def _get_eval_data(self) -> List[Dict[str, Any]]:
        """Load evaluation split."""
        if self._eval_split == self._split:
            return self._raw_data
        
        logger.info("Loading eval split: %s", self._eval_split)
        if self._subset:
            ds = load_dataset(self.source, self._subset, split=self._eval_split)
        else:
            ds = load_dataset(self.source, split=self._eval_split)
        
        examples = []
        rng = random.Random(self.split_seed + 1)
        for idx, item in enumerate(ds):
            # Handle different dataset formats
            if "A" in item and "B" in item and "C" in item and "D" in item:
                question = item.get("Question", item.get("question", ""))
                choices = [str(item["A"]), str(item["B"]), str(item["C"]), str(item["D"])]
                answer = item.get("Answer", item.get("answer", "A"))
                answer_idx = POSITION_LABELS.index(answer.upper()) if isinstance(answer, str) else answer
            elif "choices" in item:
                question = item.get("question", "")
                choices = item.get("choices", [])
                answer_idx = item.get("answer_idx", item.get("answer", 0))
                if isinstance(answer_idx, str):
                    answer_idx = POSITION_LABELS.index(answer_idx.upper())
            else:
                continue
            
            if not question or len(choices) != self.num_choices:
                continue
            
            examples.append({
                "idx": idx,
                "question": question,
                "choices": choices,
                "correct_idx": answer_idx,
                "correct_answer": choices[answer_idx],
            })
        
        return examples
    
    def _get_example_key(self, example: Dict[str, Any]) -> str:
        return example["question"][:100]
    
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
            correct_answer = example["correct_answer"]
            incorrect_choices = [c for c in choices if c != correct_answer]
            
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
                metadata={"question_idx": example["idx"], "response_type": "correct"},
            ))
            
            # Pair 2: Incorrect response (use first incorrect choice)
            if incorrect_choices:
                response_incorrect = format_freeform_response(incorrect_choices[0])
                pairs.append(ContrastivePair(
                    positive_text=format_conversation(tokenizer, prompt_first, response_incorrect),
                    negative_text=format_conversation(tokenizer, prompt_last, response_incorrect),
                    metadata={"question_idx": example["idx"], "response_type": "incorrect"},
                ))
        
        logger.info("Created %d contrastive pairs for position freeform probe", len(pairs))
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
        correct_answer = raw_example["correct_answer"]
        incorrect_choices = [c for c in choices if c != correct_answer]
        
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
                "question_idx": raw_example["idx"],
                "question": question,
                "correct_answer": correct_answer,
                # In correct_first ordering, correct is at index 0
                # In correct_last ordering, correct is at last index
                "correct_first_idx": 0,
                "correct_last_idx": len(choices_correct_last) - 1,
                "num_choices": len(choices_correct_last),
            },
        )
    
    def _compute_splits(self) -> None:
        """Override to handle different splitting strategies based on eval_split.
        
        Two strategies:
        1. If eval_split == train_split: use hash-based split of loaded data (no contamination)
        2. If eval_split != train_split: use loaded data for probe (eval is separate), capped at probe_size
        """
        if self._eval_split == self._split:
            # Same split: must hash-split to avoid contamination
            super()._compute_splits()
            logger.info(
                "Position freeform: hash-split %d probe / %d test from same split (%s)",
                len(self._probe_indices), len(self._test_indices), self._split
            )
        else:
            # Different splits: use loaded data for probe (capped), eval is separate
            n_total = len(self._raw_data)
            actual_probe_size = min(self.probe_size, n_total)
            self._probe_indices = list(range(actual_probe_size))
            self._test_indices = []  # Not used - evaluation uses eval_split
            logger.info(
                "Position freeform: %d probe examples (capped at %d) from %s split, eval will use %s split (separate)",
                actual_probe_size, self.probe_size, self._split, self._eval_split
            )
    
    def get_eval_examples(self, tokenizer: Any) -> List[EvalExample]:
        """Get evaluation examples from eval split."""
        self._ensure_loaded()

        # IMPORTANT: if eval_split == train_split, avoid probe/test leakage by
        # evaluating ONLY on held-out indices computed by _compute_splits().
        if self._eval_split == self._split:
            indices = list(self._test_indices)
            if self.max_test_examples is not None:
                indices = indices[: self.max_test_examples]

            examples: List[EvalExample] = []
            for i in indices:
                raw_example = self._raw_data[i]
                example = self._make_eval_example(raw_example, tokenizer)
                if example is not None:
                    examples.append(example)
        else:
            eval_data = self._get_eval_data()

            # Apply max_test_examples limit
            if self.max_test_examples is not None:
                eval_data = eval_data[: self.max_test_examples]

            examples: List[EvalExample] = []
            for raw_example in eval_data:
                example = self._make_eval_example(raw_example, tokenizer)
                if example is not None:
                    examples.append(example)
        
        logger.info("Created %d eval examples for position freeform", len(examples))
        return examples


def compute_freeform_position_metrics(
    rewards: Dict[str, List[float]],
    n_examples: int,
    num_choices: int = 4,
) -> Dict[str, float]:
    """Compute position bias metrics for freeform choice listing.
    
    Robustly handles varying numbers of choices per example by filtering None values.
    
    Args:
        rewards: Dictionary with reward lists for each ordering and choice
        n_examples: Number of examples
        num_choices: Maximum number of choices to check
        
    Returns:
        Dictionary with accuracy metrics
    """
    n = n_examples
    if n == 0:
        return {"accuracy": 0, "n_examples": 0}
    
    # Accuracy when correct answer is listed FIRST
    correct_first_accuracy = 0
    actual_n_first = 0
    for i in range(n):
        # Determine actual rewards for this example (filter out None)
        example_rewards = []
        for j in range(num_choices):
            key = f"correct_first_choice_{j}"
            if key in rewards and rewards[key][i] is not None:
                example_rewards.append(rewards[key][i])
        
        if not example_rewards:
            continue
            
        actual_n_first += 1
        predicted = example_rewards.index(max(example_rewards))
        if predicted == 0:  # Correct is at index 0
            correct_first_accuracy += 1
            
    if actual_n_first > 0:
        correct_first_accuracy /= actual_n_first
    
    # Accuracy when correct answer is listed LAST
    correct_last_accuracy = 0
    actual_n_last = 0
    for i in range(n):
        # Determine actual rewards for this example (filter out None)
        example_rewards = []
        for j in range(num_choices):
            key = f"correct_last_choice_{j}"
            if key in rewards and rewards[key][i] is not None:
                example_rewards.append(rewards[key][i])
        
        if not example_rewards:
            continue
            
        actual_n_last += 1
        predicted = example_rewards.index(max(example_rewards))
        if predicted == len(example_rewards) - 1:  # Correct is at last index
            correct_last_accuracy += 1
            
    if actual_n_last > 0:
        correct_last_accuracy /= actual_n_last
    
    overall_accuracy = (correct_first_accuracy + correct_last_accuracy) / 2
    accuracy_gap = correct_first_accuracy - correct_last_accuracy
    
    return {
        "accuracy": overall_accuracy,
        "accuracy_correct_first": correct_first_accuracy,
        "accuracy_correct_last": correct_last_accuracy,
        "accuracy_gap": accuracy_gap,
        "n_examples": n,
        "actual_n": (actual_n_first + actual_n_last) / 2,
    }


# =============================================================================
# Correctness datasets for probe cleaning
# =============================================================================

@DatasetRegistry.register("correctness_position_freeform")
class CorrectnessPositionFreeformDataset(PositionFreeformDataset):
    """Correctness probe for position freeform: correct vs incorrect response.
    
    Uses both position orderings (first/last) so the probe isolates 
    response correctness, not position.
    
    Positive: correct response (under both orderings)
    Negative: incorrect response (under both orderings)
    """
    
    @property
    def name(self) -> str:
        return "correctness_position_freeform"
    
    def get_probe_pairs(self, tokenizer: Any) -> List[ContrastivePair]:
        """Create contrastive pairs: correct vs incorrect response.
        
        Each question yields 2 pairs (one per ordering):
        - Positive: correct response
        - Negative: incorrect response
        """
        from src.nb.datasets.base import format_conversation
        
        self._ensure_loaded()
        
        pairs = []
        for idx in self._probe_indices:
            example = self._raw_data[idx]
            question = example["question"]
            choices = example["choices"]
            correct_answer = example["correct_answer"]
            incorrect_choices = [c for c in choices if c != correct_answer]
            
            if not incorrect_choices:
                continue
            
            response_correct = format_freeform_response(correct_answer)
            response_incorrect = format_freeform_response(incorrect_choices[0])
            
            # Under correct-first ordering
            choices_first = [correct_answer] + incorrect_choices
            prompt_first = format_freeform_choices_prompt(question, choices_first)
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, prompt_first, response_correct),
                negative_text=format_conversation(tokenizer, prompt_first, response_incorrect),
                metadata={"ordering": "correct_first"},
            ))
            
            # Under correct-last ordering
            choices_last = incorrect_choices + [correct_answer]
            prompt_last = format_freeform_choices_prompt(question, choices_last)
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, prompt_last, response_correct),
                negative_text=format_conversation(tokenizer, prompt_last, response_incorrect),
                metadata={"ordering": "correct_last"},
            ))
        
        logger.info("Created %d correctness probe pairs for position freeform", len(pairs))
        return pairs


@DatasetRegistry.register("correctness_position_plausibleqa")
class CorrectnessPositionPlausibleQADataset(PositionPlausibleQADataset):
    """Correctness probe for PlausibleQA position: correct vs incorrect response.
    
    Uses both position orderings (first/last) so the probe isolates
    response correctness, not position.
    
    Positive: correct response (under both orderings)
    Negative: incorrect response (under both orderings)
    """
    
    @property
    def name(self) -> str:
        return "correctness_position_plausibleqa"
    
    def get_probe_pairs(self, tokenizer: Any) -> List[ContrastivePair]:
        """Create contrastive pairs: correct vs incorrect response."""
        from src.nb.datasets.base import format_conversation
        
        self._ensure_loaded()
        
        pairs = []
        for idx in self._probe_indices:
            example = self._raw_data[idx]
            question = example["question"]
            correct = example["correct_answer"]
            incorrect_answers = example["incorrect_answers"]
            
            response_correct = format_position_response(correct)
            response_incorrect = format_position_response(incorrect_answers[0])
            
            # Under correct-first ordering
            choices_first = [correct] + incorrect_answers
            prompt_first = format_numbered_choices_prompt(question, choices_first)
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, prompt_first, response_correct),
                negative_text=format_conversation(tokenizer, prompt_first, response_incorrect),
                metadata={"ordering": "correct_first"},
            ))
            
            # Under correct-last ordering
            choices_last = incorrect_answers + [correct]
            prompt_last = format_numbered_choices_prompt(question, choices_last)
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, prompt_last, response_correct),
                negative_text=format_conversation(tokenizer, prompt_last, response_incorrect),
                metadata={"ordering": "correct_last"},
            ))
        
        logger.info("Created %d correctness probe pairs for PlausibleQA position", len(pairs))
        return pairs


# =============================================================================
# PlausibleQA Freeform (5 choices, comma-separated list)
# =============================================================================

@DatasetRegistry.register("position_freeform_plausibleqa")
class PositionFreeformPlausibleQADataset(ProbeDataset):
    """Position bias dataset for PlausibleQA with freeform listing.
    
    Tests whether RM accuracy changes based on where correct answer 
    appears in the list of options (first vs last).
    
    Prompt: "{question}\n\nThe answer is either {a}, {b}, {c}, {d}, or {e}."
    Response: "The answer is {X}."
    
    Uses 5 choices: 1 correct + 4 most plausible incorrect.
    
    Evaluates two orderings per question:
    - correct_first: correct answer listed first
    - correct_last: correct answer listed last
    """
    
    NUM_CHOICES = 5
    
    def __init__(
        self,
        source: str,
        probe_size: int = 500,
        split_seed: int = 42,
        max_test_examples: Optional[int] = None,
    ):
        super().__init__(source, probe_size, split_seed, max_test_examples)
    
    @property
    def name(self) -> str:
        return "position_freeform_plausibleqa"
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load PlausibleQA and prepare for freeform position bias testing."""
        import json
        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(f"PlausibleQA file not found: {path}")
        
        with open(path) as f:
            data = json.load(f)
        
        examples = []
        num_incorrect = self.NUM_CHOICES - 1
        
        for idx, item in enumerate(data):
            question = item.get("question", "")
            correct_answer = item.get("answer", "")
            candidates = item.get("candidate_answers", {})
            
            if not question or not correct_answer or not candidates:
                continue
            
            # Get top N most plausible incorrect answers
            if isinstance(candidates, dict):
                sorted_candidates = sorted(
                    candidates.items(),
                    key=lambda x: x[1].get("plackett_luce", 100),
                )
                if len(sorted_candidates) < num_incorrect:
                    continue
                incorrect_answers = [c[0] for c in sorted_candidates[:num_incorrect]]
            elif isinstance(candidates, list):
                if len(candidates) < num_incorrect:
                    continue
                incorrect_answers = candidates[:num_incorrect]
            else:
                continue
            
            examples.append({
                "idx": idx,
                "question": question,
                "correct_answer": correct_answer,
                "incorrect_answers": incorrect_answers,
            })
        
        logger.info("Loaded %d PlausibleQA examples for freeform position bias", len(examples))
        return examples
    
    def _get_example_key(self, example: Dict[str, Any]) -> str:
        return example["question"][:100]
    
    def _make_contrastive_pair(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[ContrastivePair]:
        return None
    
    def get_probe_pairs(self, tokenizer: Any) -> List[ContrastivePair]:
        """Create contrastive pairs: correct-first vs correct-last ordering.
        
        Positive: correct answer listed FIRST
        Negative: correct answer listed LAST
        """
        from src.nb.datasets.base import format_conversation
        
        self._ensure_loaded()
        
        pairs = []
        for idx in self._probe_indices:
            example = self._raw_data[idx]
            question = example["question"]
            correct = example["correct_answer"]
            incorrect_answers = example["incorrect_answers"]
            
            # Two orderings
            choices_first = [correct] + incorrect_answers
            choices_last = incorrect_answers + [correct]
            prompt_first = format_freeform_choices_prompt(question, choices_first)
            prompt_last = format_freeform_choices_prompt(question, choices_last)
            
            # Pair with correct response
            response_correct = format_freeform_response(correct)
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, prompt_first, response_correct),
                negative_text=format_conversation(tokenizer, prompt_last, response_correct),
                metadata={"question_idx": example["idx"], "response_type": "correct"},
            ))
            
            # Pair with incorrect response
            response_incorrect = format_freeform_response(incorrect_answers[0])
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, prompt_first, response_incorrect),
                negative_text=format_conversation(tokenizer, prompt_last, response_incorrect),
                metadata={"question_idx": example["idx"], "response_type": "incorrect"},
            ))
        
        logger.info("Created %d contrastive pairs for freeform PlausibleQA probe", len(pairs))
        return pairs
    
    def _make_eval_example(
        self, raw_example: Dict[str, Any], tokenizer: Any
    ) -> Optional[EvalExample]:
        """Create evaluation example with both orderings."""
        from src.nb.datasets.base import format_conversation
        
        question = raw_example["question"]
        correct = raw_example["correct_answer"]
        incorrect_answers = raw_example["incorrect_answers"]
        
        choices_first = [correct] + incorrect_answers
        choices_last = incorrect_answers + [correct]
        prompt_first = format_freeform_choices_prompt(question, choices_first)
        prompt_last = format_freeform_choices_prompt(question, choices_last)
        
        texts = {}
        
        # Score all 5 responses for correct-first ordering
        for i, choice in enumerate(choices_first):
            response = format_freeform_response(choice)
            texts[f"correct_first_choice_{i}"] = format_conversation(tokenizer, prompt_first, response)
        
        # Score all 5 responses for correct-last ordering
        for i, choice in enumerate(choices_last):
            response = format_freeform_response(choice)
            texts[f"correct_last_choice_{i}"] = format_conversation(tokenizer, prompt_last, response)
        
        return EvalExample(
            texts=texts,
            metadata={
                "question_idx": raw_example["idx"],
                "question": question,
                "correct_answer": correct,
                "correct_first_idx": 0,
                "correct_last_idx": self.NUM_CHOICES - 1,
                "num_choices": self.NUM_CHOICES,
            },
        )


@DatasetRegistry.register("correctness_position_freeform_plausibleqa")
class CorrectnessPositionFreeformPlausibleQADataset(PositionFreeformPlausibleQADataset):
    """Correctness probe for freeform PlausibleQA: correct vs incorrect response.
    
    Uses both orderings so the probe isolates correctness, not position.
    """
    
    @property
    def name(self) -> str:
        return "correctness_position_freeform_plausibleqa"
    
    def get_probe_pairs(self, tokenizer: Any) -> List[ContrastivePair]:
        """Create contrastive pairs: correct vs incorrect response."""
        from src.nb.datasets.base import format_conversation
        
        self._ensure_loaded()
        
        pairs = []
        for idx in self._probe_indices:
            example = self._raw_data[idx]
            question = example["question"]
            correct = example["correct_answer"]
            incorrect_answers = example["incorrect_answers"]
            
            response_correct = format_freeform_response(correct)
            response_incorrect = format_freeform_response(incorrect_answers[0])
            
            # Under correct-first ordering
            choices_first = [correct] + incorrect_answers
            prompt_first = format_freeform_choices_prompt(question, choices_first)
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, prompt_first, response_correct),
                negative_text=format_conversation(tokenizer, prompt_first, response_incorrect),
                metadata={"ordering": "correct_first"},
            ))
            
            # Under correct-last ordering
            choices_last = incorrect_answers + [correct]
            prompt_last = format_freeform_choices_prompt(question, choices_last)
            pairs.append(ContrastivePair(
                positive_text=format_conversation(tokenizer, prompt_last, response_correct),
                negative_text=format_conversation(tokenizer, prompt_last, response_incorrect),
                metadata={"ordering": "correct_last"},
            ))
        
        logger.info("Created %d correctness probe pairs for freeform PlausibleQA", len(pairs))
        return pairs

