from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from saerm.config import EmbeddingJobConfig, StorageConfig, load_experiment_config
from saerm.embeddings.cache import EmbeddingCacheManager
from saerm.storage import StorageManager


def _patch_transformer_loaders(monkeypatch):
    pytest.importorskip("transformers")

    class _DummyModel(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
            batch, seq_len = input_ids.shape
            base = torch.arange(batch * seq_len, dtype=torch.float32).view(batch, seq_len, 1)
            hidden_states = tuple(base + layer for layer in range(3))
            return type("DummyOutput", (), {"hidden_states": hidden_states})

        def to(self, device):
            return self

        def eval(self):
            return self

    class _DummyTokenizer:
        def __call__(self, text, return_tensors="pt", truncation=True, **kwargs):
            length = max(1, min(8, len(str(text))))
            input_ids = torch.ones(1, length, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    monkeypatch.setattr(
        "saerm.embeddings.cache.AutoModelForSequenceClassification.from_pretrained",
        lambda *args, **kwargs: _DummyModel(),
    )
    monkeypatch.setattr(
        "saerm.embeddings.cache.AutoModel.from_pretrained",
        lambda *args, **kwargs: _DummyModel(),
    )
    monkeypatch.setattr(
        "saerm.embeddings.cache.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _DummyTokenizer(),
    )


def test_embedding_cache_records_choices(monkeypatch, tmp_path):
    _patch_transformer_loaders(monkeypatch)

    storage = StorageManager(StorageConfig(root_url=str(tmp_path)))
    manager = EmbeddingCacheManager(storage, device="cpu")

    record = {
        "prompt": "Why?",
        "chosen": [
            {"role": "user", "content": "Why?"},
            {"role": "assistant", "content": "Because I said so."},
        ],
        "rejected": [
            {"role": "user", "content": "Why?"},
            {"role": "assistant", "content": "No idea."},
        ],
    }

    job = EmbeddingJobConfig(
        job_id="dummy",
        model="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        layer="last",
        dataset="dummy_ds",
        prompt_field="prompt",
        chosen_field="chosen",
        rejected_field="rejected",
        batch_size=2,
        max_examples=1,
    )

    manager.run_job(job, [record])

    payload = manager.load_embeddings("dummy")
    embeddings = payload["embeddings"]
    assert embeddings.shape[0] == 2
    records = payload["records"]
    assert len(records) == 2
    assert {entry["choice"] for entry in records} == {"chosen", "rejected"}
    assert {entry["dataset"] for entry in records} == {"dummy_ds"}
    assert {entry["example_index"] for entry in records} == {0}


@pytest.mark.integration
def test_embedding_cache_real_dataset(monkeypatch, tmp_path):
    datasets = pytest.importorskip("datasets")
    _patch_transformer_loaders(monkeypatch)
    torch = pytest.importorskip("torch")

    storage = StorageManager(StorageConfig(root_url=str(tmp_path)))
    manager = EmbeddingCacheManager(storage, device="cpu")

    ds = datasets.load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.2", split="train").select(range(20))
    job = EmbeddingJobConfig(
        job_id="integration",
        model="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        layer="last",
        dataset="skywork_train",
        prompt_field="prompt",
        chosen_field="chosen",
        rejected_field="rejected",
        max_examples=20,
    )

    manager.run_job(job, ds)

    payload = manager.load_embeddings("integration")
    embeddings = payload["embeddings"]
    assert embeddings.shape[0] == 40

    records = payload["records"]
    assert len(records) == 40

    example_choices: dict[int, set[str]] = {}
    for entry in records:
        example_choices.setdefault(int(entry["example_index"]), set()).add(entry["choice"])
        assert entry["dataset"] == "skywork_train"
    assert all(choices == {"chosen", "rejected"} for choices in example_choices.values())

    unique_rows = torch.unique(embeddings, dim=0).shape[0]
    assert unique_rows >= int(embeddings.shape[0] * 0.9)


def test_embedding_cache_config_five_examples(monkeypatch, tmp_path):
    _patch_transformer_loaders(monkeypatch)

    storage = StorageManager(StorageConfig(root_url=str(tmp_path)))
    manager = EmbeddingCacheManager(storage, device="cpu")

    config = load_experiment_config()
    assert config.embedding_jobs, "expected at least one embedding job in config"
    base_job = config.embedding_jobs[0]
    job = replace(base_job, job_id="config-5", max_examples=5)

    chosen_field = job.chosen_field or "chosen"
    rejected_field = job.rejected_field or "rejected"
    dataset = []
    for idx in range(5):
        prompt = f"prompt {idx}"
        dataset.append({
            chosen_field: [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"chosen {idx}"},
            ],
            rejected_field: [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"rejected {idx}"},
            ],
        })

    manager.run_job(job, dataset)

    payload = manager.load_embeddings("config-5")
    embeddings = payload["embeddings"]
    assert embeddings.shape[0] == 10
    records = payload["records"]
    assert len(records) == 10
    assert sum(1 for record in records if record["choice"] == "chosen") == 5
    assert sum(1 for record in records if record["choice"] == "rejected") == 5
