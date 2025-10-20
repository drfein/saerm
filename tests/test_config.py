from pathlib import Path

from saerm.config import load_experiment_config


def test_load_experiment_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
storage:
  root_url: ./outputs

embedding_jobs: []
sae_jobs: []
head_jobs: []
"""
    )
    config = load_experiment_config(cfg_path)
    assert config.storage.root_url == "./outputs"
    assert config.embedding_jobs == []
