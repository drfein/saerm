import sys
import types

import pytest


def _make_record(**kwargs):
    ex = {
        "chosen_story": None,
        "chosen": None,
        "chosen_text": None,
        "rejected_story": None,
        "rejected": None,
        "rejected_text": None,
    }
    ex.update(kwargs)
    return ex


class _FakeDataset(list):
    pass


def test_litbench_hf_pairs():
    import os

    # Prefer real HF datasets in this test; require dependencies to be installed
    import datasets  # noqa: F401
    token = os.environ.get("HF_TOKEN", "").strip() or None

    from saerm.process_data.litbench_hf import LitBenchHF
    from saerm.process_data.base import PreferencePair

    loader = LitBenchHF(hf_token=token)
    train_pairs = loader.train()
    test_pairs = loader.test()

    assert len(train_pairs) > 0 and len(test_pairs) > 0
    assert all(
        isinstance(p, PreferencePair)
        and p.chosen
        and p.rejected
        and p.chosen != p.rejected
        for p in train_pairs[:8]
    )


def test_get_first_nonempty():
    from saerm.process_data.litbench_hf import LitBenchHF

    ex = {"a": None, "b": "  ", "c": "x"}
    assert LitBenchHF._get_first_nonempty(ex, ["a", "b", "c"]) == "x"
    assert LitBenchHF._get_first_nonempty(ex, ["a"]) == ""


