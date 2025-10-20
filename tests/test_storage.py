from saerm.config import StorageConfig
from saerm.storage import StorageManager


def test_storage_manager_paths(tmp_path):
    cfg = StorageConfig(root_url=str(tmp_path))
    storage = StorageManager(cfg)
    storage.prepare()
    emb_path = storage.embedding_tensor_path("demo")
    sae_path = storage.sae_checkpoint_path("demo")
    head_path = storage.head_model_path("demo")
    assert emb_path.parent.exists()
    assert sae_path.parent.exists()
    assert head_path.parent.exists()
