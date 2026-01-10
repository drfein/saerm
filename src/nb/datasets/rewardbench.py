"""
RewardBench dataset wrapper.

Dataset: allenai/reward-bench-2

Each example has one correct answer and multiple incorrect answers.
For evaluation an example is counted correct only if the reward for the
correct answer exceeds ALL incorrect answers.

Probe construction:
- Positive: correct answer
- Negative: a randomly chosen incorrect answer
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from src.nb.datasets.base import (
    ContrastivePair,
    DatasetRegistry,
    EvalExample,
    ProbeDataset,
    format_conversation,
)

logger = logging.getLogger(__name__)


@DatasetRegistry.register("rewardbench")
class RewardBenchDataset(ProbeDataset):
    """allenai/reward-bench-2 dataset for RM evaluation."""

    def __init__(
        self,
        source: str = "allenai/reward-bench-2",
        split: str = "test",
        probe_size: int = 500,
        split_seed: int = 42,
        max_test_examples: Optional[int] = None,
        **_: Any,
    ):
        self.split = split
        super().__init__(
            source=source,
            probe_size=probe_size,
            split_seed=split_seed,
            max_test_examples=max_test_examples,
        )

    @property
    def name(self) -> str:
        return "rewardbench"

    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load reward-bench-2 split and normalize fields."""
        logger.info("Loading %s (split=%s)", self.source, self.split)
        dataset = load_dataset(self.source, split=self.split)
        examples: List[Dict[str, Any]] = []

        for idx, row in enumerate(dataset):
            prompt = (
                row.get("prompt")
                or row.get("question")
                or row.get("instruction")
                or row.get("input")
            )

            # Responses may be provided in different forms:
            responses = []
            if "responses" in row and isinstance(row["responses"], list):
                for r in row["responses"]:
                    text = r.get("text") or r.get("response") or r.get("completion")
                    is_correct = bool(r.get("is_correct", r.get("label", r.get("score", 0) > 0)))
                    if text:
                        responses.append({"text": text, "is_correct": is_correct})
            elif "chosen" in row:
                responses.append({"text": row["chosen"], "is_correct": True})
                rejected = row.get("rejected_responses") or row.get("rejected") or row.get("rejected_list")
                if isinstance(rejected, list):
                    for t in rejected:
                        responses.append({"text": t, "is_correct": False})
            elif "answer" in row:
                responses.append({"text": row["answer"], "is_correct": True})
                wrongs = row.get("distractors") or row.get("incorrect_answers") or []
                for t in wrongs:
                    responses.append({"text": t, "is_correct": False})

            correct = [r for r in responses if r.get("is_correct")]
            incorrect = [r for r in responses if not r.get("is_correct")]

            if not prompt or not correct or not incorrect:
                continue

            examples.append(
                {
                    "idx": idx,
                    "prompt": prompt,
                    "correct": [c["text"] for c in correct],
                    "incorrect": [c["text"] for c in incorrect],
                }
            )

        logger.info("Loaded %d reward-bench examples with correct+incorrect answers", len(examples))
        return examples

    def _get_example_key(self, example: Dict[str, Any]) -> str:
        return example["prompt"][:100]

    def _make_contrastive_pair(
        self,
        raw_example: Dict[str, Any],
        tokenizer: Any,
    ) -> Optional[ContrastivePair]:
        """Construct a contrastive pair: correct vs a random incorrect."""
        prompt = raw_example["prompt"]
        correct = random.choice(raw_example["correct"])
        incorrect = random.choice(raw_example["incorrect"])

        positive = format_conversation(tokenizer, prompt, correct)
        negative = format_conversation(tokenizer, prompt, incorrect)

        return ContrastivePair(
            positive_text=positive,
            negative_text=negative,
            metadata={"idx": raw_example["idx"]},
        )

    def _make_eval_example(
        self,
        raw_example: Dict[str, Any],
        tokenizer: Any,
    ) -> Optional[EvalExample]:
        """Evaluation: accuracy requires correct > all incorrect."""
        prompt = raw_example["prompt"]
        correct = raw_example["correct"][0]
        incorrect_list = raw_example["incorrect"]

        texts = {"correct": format_conversation(tokenizer, prompt, correct)}
        for j, inc in enumerate(incorrect_list):
            texts[f"incorrect_{j}"] = format_conversation(tokenizer, prompt, inc)

        return EvalExample(
            texts=texts,
            metadata={"idx": raw_example["idx"], "n_incorrect": len(incorrect_list)},
        )


