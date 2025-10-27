import json
import threading

import numpy as np

from saerm.autointerp import AutoInterpreter
from saerm.config import (
    AutoInterpJobConfig,
    DatasetConfig,
    EmbeddingJobConfig,
    ExperimentConfig,
    HeadTrainingConfig,
    SAETrainingConfig,
    StorageConfig,
)
from saerm.heads.bt_linear import LinearBTHead
from saerm.storage import StorageManager


class FakeLLMClient:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []
        self.model_name = "fake-gemini"
        self._lock = threading.Lock()

    def generate(self, prompt, assistant_prompt=None):
        with self._lock:
            index = len(self.calls)
            if index >= len(self._responses):
                raise RuntimeError("Not enough fake responses configured")
            self.calls.append((prompt, assistant_prompt))
        return self._responses[index]


def test_autointerp_runs_and_writes_results(tmp_path):
    storage = StorageManager(StorageConfig(root_url=str(tmp_path)))
    storage.prepare()

    feature_examples_path = storage.sae_feature_examples_path("demo-sae")
    feature_examples_path.write_text(
        json.dumps(
            {
                "0": [
                    {"activation": 1.23, "text": "Alpha example", "dataset_index": 0},
                    {"activation": 0.9, "text": "Beta example"},
                ],
                "1": [
                    {"activation": 0.42, "text": "Gamma example"},
                ],
            }
        )
    )

    job = AutoInterpJobConfig(
        job_id="demo-autointerp",
        sae_job="demo-sae",
        max_examples_per_feature=1,
    )
    fake_llm = FakeLLMClient(
        [
            "Explanation: Strong theme around tokens\nScore: 5",
            "Explanation: Hard to tell pattern",
        ]
    )
    interpreter = AutoInterpreter(storage, job, llm_client=fake_llm)
    result = interpreter.run()

    assert fake_llm.calls, "Expected LLM to be invoked"
    assert "feature_results" in result
    output_path = storage.sae_autointerp_path("demo-sae", "demo-autointerp")
    saved = json.loads(output_path.read_text())

    assert saved["feature_results"]["0"]["score"] == 5
    assert saved["feature_results"]["1"]["score"] is None
    assert saved["feature_results"]["1"]["parse_errors"] == ["missing_score"]
    assert "Alpha example" in fake_llm.calls[0][0]
    assert "Source:" not in fake_llm.calls[0][0]


def test_autointerp_respects_max_concepts(tmp_path):
    storage = StorageManager(StorageConfig(root_url=str(tmp_path)))
    storage.prepare()
    feature_examples_path = storage.sae_feature_examples_path("demo-sae")
    feature_examples_path.write_text(
        json.dumps(
            {
                "0": [{"activation": 0.5, "text": "First"}],
                "1": [{"activation": 0.4, "text": "Second"}],
            }
        )
    )

    job = AutoInterpJobConfig(
        job_id="demo-limit",
        sae_job="demo-sae",
        max_concepts=1,
    )
    fake_llm = FakeLLMClient(["Explanation: Pattern\nScore: 4"])
    interpreter = AutoInterpreter(storage, job, llm_client=fake_llm)
    result = interpreter.run()

    assert len(fake_llm.calls) == 1
    feature_results = result["feature_results"]
    assert list(feature_results.keys()) == ["0"]


def test_autointerp_hydrates_missing_context(tmp_path, monkeypatch):
    storage = StorageManager(StorageConfig(root_url=str(tmp_path)))
    storage.prepare()
    feature_examples_path = storage.sae_feature_examples_path("demo-sae")
    feature_examples_path.write_text(
        json.dumps(
            {
                "0": [
                    {
                        "activation": 0.8,
                        "dataset_index": 0,
                        "example_id": 0,
                        "source": "chosen",
                    }
                ],
                "1": [
                    {
                        "activation": 0.7,
                        "dataset_index": 0,
                        "example_id": 0,
                        "source": "chosen",
                    }
                ],
            }
        )
    )

    config = ExperimentConfig(
        storage=StorageConfig(root_url=str(tmp_path)),
        datasets={"dummy": DatasetConfig(name="dummy", split="train")},
        embedding_jobs=[EmbeddingJobConfig(job_id="embed-job", model="test-model", dataset="dummy")],
        sae_jobs=[
            SAETrainingConfig(
                job_id="demo-sae",
                embedding_job="embed-job",
                hidden_size=1,
                k_active=1,
                learning_rate=1e-3,
                batch_size=1,
            )
        ],
    )

    class FakeDataset:
        def __init__(self, rows):
            self._rows = rows
            self.calls = 0

        def __getitem__(self, idx):
            self.calls += 1
            return self._rows[idx]

        def __len__(self):
            return len(self._rows)

    class FakeDatasetManager:
        def __init__(self, mapping):
            self._mapping = mapping

        def get(self, key, split=None):
            return self._mapping[key]

    dataset = FakeDataset(
        [
            {
                "prompt": "What is 2+2?",
                "chosen": [{"role": "assistant", "content": "4"}],
                "rejected": [{"role": "assistant", "content": "5"}],
            }
        ]
    )
    dataset_manager = FakeDatasetManager({"dummy": dataset})

    class DummyTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
            parts = [f"{msg['role']}: {msg['content']}" for msg in messages]
            return " | ".join(parts)

    monkeypatch.setattr("saerm.autointerp.AutoTokenizer", type("Tok", (), {"from_pretrained": staticmethod(lambda _: DummyTokenizer())}))

    job = AutoInterpJobConfig(job_id="demo-autointerp", sae_job="demo-sae")
    fake_llm = FakeLLMClient(
        ["Explanation: math\nScore: 5", "Explanation: repeat\nScore: 4"]
    )
    interpreter = AutoInterpreter(
        storage,
        job,
        config=config,
        dataset_manager=dataset_manager,
        llm_client=fake_llm,
    )
    interpreter.run()

    prompt = fake_llm.calls[0][0]
    assert "assistant: 4" in prompt.lower()
    assert "[no context available]" not in prompt
    assert "assistant: 4" in fake_llm.calls[1][0].lower()
    assert dataset.calls == 1


def test_autointerp_batches_persist(tmp_path, monkeypatch):
    storage = StorageManager(StorageConfig(root_url=str(tmp_path)))
    storage.prepare()
    feature_examples_path = storage.sae_feature_examples_path("demo-sae")
    feature_examples_path.write_text(
        json.dumps(
            {
                "0": [{"activation": 0.5, "text": "ctx0"}],
                "1": [{"activation": 0.4, "text": "ctx1"}],
            }
        )
    )

    job = AutoInterpJobConfig(
        job_id="demo-batch",
        sae_job="demo-sae",
        batch_size=1,
        max_examples_per_feature=1,
    )
    fake_llm = FakeLLMClient(
        [
            "Explanation: first\nScore: 5",
            "Explanation: second\nScore: 4",
        ]
    )

    original_persist = AutoInterpreter._persist_results
    persist_calls = []

    def tracker(self, output_path, results, assistant_prompt, max_examples, model_name):
        persist_calls.append(len(results))
        return original_persist(self, output_path, results, assistant_prompt, max_examples, model_name)

    monkeypatch.setattr(AutoInterpreter, "_persist_results", tracker)
    interpreter = AutoInterpreter(storage, job, llm_client=fake_llm)
    interpreter.run()

    assert persist_calls[0] == 1
    assert persist_calls[-1] == 2


def test_autointerp_selects_features_from_head(tmp_path):
    storage = StorageManager(StorageConfig(root_url=str(tmp_path)))
    storage.prepare()
    feature_examples_path = storage.sae_feature_examples_path("demo-sae")
    feature_examples_path.write_text(
        json.dumps(
            {
                "0": [{"activation": 0.5, "text": "ctx0"}],
                "1": [{"activation": 0.4, "text": "ctx1"}],
                "2": [{"activation": 0.3, "text": "ctx2"}],
            }
        )
    )

    head = LinearBTHead()
    head._weights = np.asarray([0.8, -0.2, 0.1])
    head_path = storage.head_model_path("demo-head")
    head_path.parent.mkdir(parents=True, exist_ok=True)
    head.save(str(head_path))

    config = ExperimentConfig(
        storage=StorageConfig(root_url=str(tmp_path)),
        datasets={"dummy": DatasetConfig(name="dummy")},
        embedding_jobs=[EmbeddingJobConfig(job_id="embed-job", model="test-model", dataset="dummy")],
        sae_jobs=[
            SAETrainingConfig(
                job_id="demo-sae",
                embedding_job="embed-job",
                hidden_size=3,
                k_active=1,
                learning_rate=1e-3,
                batch_size=1,
            )
        ],
        head_jobs=[
            HeadTrainingConfig(
                job_id="demo-head",
                embedding_job="embed-job",
                sae_job="demo-sae",
                dataset="dummy",
                head_type="linear_bt",
            )
        ],
    )

    job = AutoInterpJobConfig(
        job_id="demo-autointerp",
        sae_job="demo-sae",
        head_job="demo-head",
        head_top_n=1,
    )
    fake_llm = FakeLLMClient([
        "Explanation: pos\nScore: 5",
        "Explanation: neg\nScore: 2",
    ])
    interpreter = AutoInterpreter(storage, job, config=config, llm_client=fake_llm)
    result = interpreter.run()

    assert set(result["feature_results"].keys()) == {"0", "1"}
