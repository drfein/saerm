"""Optional Hugging Face integration tests.

Run with: pytest -m hf --hf --max-examples=4

Requires:
  - Environment variable HF_TOKEN (or pass via env)
  - Network access
  - Optional: BTRM_REPO_ID for BTRM.load (else uses a tiny public model)
"""

import os
import sys

import numpy as np
import pytest


def _require_hf_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    return token


@pytest.mark.hf
def test_litbench_hf_live_small(monkeypatch):
    token = _require_hf_token()

    # Use the live loader; just check it yields some pairs
    from saerm.process_data.litbench_hf import LitBenchHF

    ds = LitBenchHF(hf_token=token)
    train_pairs = ds.train()
    test_pairs = ds.test()

    assert len(train_pairs) > 0
    assert len(test_pairs) > 0
    assert all(p.chosen and p.rejected and p.chosen != p.rejected for p in train_pairs[:8])


@pytest.mark.hf
def test_btrm_live_tokenize_and_score(monkeypatch):
    token = _require_hf_token()

    # Use provided repo id by default; can be overridden via BTRM_REPO_ID
    repo = os.environ.get(
        "BTRM_REPO_ID",
        "danielfein/bt-rewmodel-meta-llama_Llama-3.2-3B-final-20250513_082322",
    )

    from saerm.rm.btrm import BTRM

    inst = BTRM.load(
        repo,
        {"hf_token": token, "max_length": 64, "device": "cpu", "trust_remote_code": False},
    )
    s = inst.reward("This is a fantastic day!", prompt=None)
    assert isinstance(s, float)


