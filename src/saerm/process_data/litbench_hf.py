from __future__ import annotations

from typing import List

from .base import BasePreferenceDataset, PreferencePair


class LitBenchHF(BasePreferenceDataset):
    """Hugging Face-based loader for LitBench Train/Test-Enhanced.

    Loads pairs from the HF Hub and exposes them via the BasePreferenceDataset
    interface. This mirrors the notebook extraction logic that tolerates
    multiple possible field names for chosen/rejected text.
    """

    def __init__(
        self,
        *,
        hf_token: str | None = None,
        train_repo_id: str = "SAA-Lab/LitBench-Train",
        test_repo_id: str = "SAA-Lab/LitBench-Test-Enhanced",
    ) -> None:
        from datasets import load_dataset

        self._hf_token = hf_token
        self._ds_train = load_dataset(train_repo_id, split="train", token=hf_token)
        # Test-Enhanced only exposes a "train" split; use it as test
        self._ds_test = load_dataset(test_repo_id, split="train", token=hf_token)

    @staticmethod
    def _get_first_nonempty(ex: dict, keys: List[str]) -> str:
        for k in keys:
            if k in ex and ex[k] is not None:
                val = str(ex[k]).strip()
                if val:
                    return val
        return ""

    def _pairs_from_split(self, split: str) -> List[PreferencePair]:
        src = self._ds_train if split == "train" else self._ds_test
        out: List[PreferencePair] = []
        for ex in src:
            chosen = self._get_first_nonempty(ex, ["chosen_story", "chosen", "chosen_text"])
            rejected = self._get_first_nonempty(ex, ["rejected_story", "rejected", "rejected_text"])
            if chosen and rejected and chosen != rejected:
                out.append(PreferencePair(chosen=chosen, rejected=rejected))
        return out

    def train(self):
        return self._pairs_from_split("train")

    def test(self):
        return self._pairs_from_split("test")


__all__ = ["LitBenchHF"]


