from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
from tqdm.auto import tqdm

from ..process_data.base import BasePreferenceDataset
from ..rm.base import BaseRewardModel


@dataclass(frozen=True)
class PreferenceEvalResult:
    accuracy: float
    total: int
    correct: int


class PreferenceEvaluator:
    """Compute pairwise accuracy of a reward model on a preference dataset.

    For each pair (chosen, rejected), we score both with the reward model and
    count a correct prediction when reward(chosen) > reward(rejected).
    """

    def __init__(self, model: BaseRewardModel):
        self._model = model

    def evaluate(
        self,
        dataset: BasePreferenceDataset,
        split: str = "test",
        show_progress: bool = True,
        batch_size: Optional[int] = None,
    ) -> PreferenceEvalResult:
        pairs = dataset.as_text_tuples(split=split)
        # Filter out pairs with null/empty entries
        def _is_bad(s: object) -> bool:
            if s is None:
                return True
            if not isinstance(s, str):
                s = str(s)
            t = s.strip()
            return t == "" or t.lower() == "null"

        pairs = [(c, r) for (c, r) in pairs if not _is_bad(c) and not _is_bad(r)]
        total = len(pairs)
        correct = 0

        # Fast batched path for SAERM-like models exposing _codes_from_texts and head weights
        can_batch = (
            batch_size is not None
            and hasattr(self._model, "_codes_from_texts")
            and hasattr(self._model, "_w")
        )

        if can_batch:
            # type: ignore[attr-defined]
            w = getattr(self._model, "_w")
            mu = getattr(self._model, "_mu", None)
            sig = getattr(self._model, "_sig", None)
            bs = int(batch_size or 0)
            rng = range(0, total, bs)
            iterator = tqdm(rng, desc=f"Evaluate ({split})") if show_progress else rng
            for i in iterator:
                chunk = pairs[i : i + bs]
                chosen_texts = [c for c, _ in chunk]
                rejected_texts = [r for _, r in chunk]
                texts: Sequence[str] = chosen_texts + rejected_texts
                # Compute codes in one go; model handles internal encode batching
                Z = getattr(self._model, "_codes_from_texts")(texts)
                if mu is not None and sig is not None:
                    Z = (Z - mu) / sig
                scores = Z @ w  # (2B,)
                b = len(chunk)
                s_ch = scores[:b]
                s_rj = scores[b:]
                correct += int((s_ch > s_rj).float().sum().item())
        else:
            iterator = tqdm(pairs, desc=f"Evaluate ({split})") if show_progress else pairs
            for chosen, rejected in iterator:
                s_ch = self._model.reward(chosen)
                s_rj = self._model.reward(rejected)
                correct += int(s_ch > s_rj)

        acc = float(correct) / total if total > 0 else 0.0
        return PreferenceEvalResult(accuracy=acc, total=total, correct=correct)


__all__ = [
    "PreferenceEvaluator",
    "PreferenceEvalResult",
]


