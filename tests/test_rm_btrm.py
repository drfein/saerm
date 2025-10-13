import sys
import types

import pytest
import torch


def test_compose_input_and_scalar_score():
    import saerm.rm.btrm as btrm_mod  # import the package module

    BTRM = btrm_mod.BTRM

    # _compose_input
    assert BTRM._compose_input("body", None) == "body"
    assert BTRM._compose_input("body", "") == "body"
    joined = BTRM._compose_input("body", "prompt")
    assert "Prompt:" in joined and "Response:" in joined and "body" in joined

    # _scalar_score on various shapes
    assert BTRM._scalar_score(torch.tensor(3.0)).item() == 3.0
    v1 = BTRM._scalar_score(torch.tensor([[1.0], [2.0]])).tolist()
    assert v1 == [1.0, 2.0]
    v2 = BTRM._scalar_score(torch.tensor([[0.5, -1.0]])).item()
    assert abs(v2 - (0.5 - -1.0)) < 1e-6
    v3 = BTRM._scalar_score(torch.tensor([[1.0, 2.0, 3.0]])).item()
    assert abs(v3 - 2.0) < 1e-6


def test_reward_calls_tokenizer_and_model(monkeypatch):
    import saerm.rm.btrm as btrm_mod

    class FakeTok:
        pad_token = "<pad>"
        pad_token_id = 0

        def __call__(self, text, padding=True, truncation=True, max_length=None, return_tensors=None):
            return {"input_ids": torch.ones(1, 4, dtype=torch.long)}

    class FakeOut:
        def __init__(self):
            self.logits = torch.tensor([[0.2, -0.1]])

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, **enc):
            return FakeOut()

        def to(self, device):
            return self

    inst = btrm_mod.BTRM(model=FakeModel(), tokenizer=FakeTok(), max_length=8, device="cpu")
    s = inst.reward("hello", prompt=None)
    assert isinstance(s, float)


def test_load_uses_fake_transformers(monkeypatch):
    import saerm.rm.btrm as btrm_mod

    class FakeTok:
        def __init__(self):
            self.pad_token = None
            self.pad_token_id = None
            self.eos_token = "</s>"
            self.eos_token_id = 2

    class FakeTokCls:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return FakeTok()

    class FakeModel:
        def __init__(self):
            self.config = types.SimpleNamespace()

        @staticmethod
        def from_pretrained(path, **kwargs):
            return FakeModel()

        def to(self, device):
            return self

    fake_tf = types.SimpleNamespace(
        AutoTokenizer=FakeTokCls,
        AutoModelForSequenceClassification=FakeModel,
    )

    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    inst = btrm_mod.BTRM.load("fake/repo", {"device": "cpu"})
    assert isinstance(inst, btrm_mod.BTRM)

    # pad token should be set to eos if missing
    assert inst._tokenizer.pad_token == inst._tokenizer.eos_token
    assert inst._tokenizer.pad_token_id == inst._tokenizer.eos_token_id


def test_train_not_implemented():
    import saerm.rm.btrm as btrm_mod

    with pytest.raises(NotImplementedError):
        btrm_mod.BTRM(None, None).train(None)  # type: ignore[arg-type]


