import numpy as np
import torch


def test_forward_and_training_small():
    from saerm.sae.batchtopk_sae import BatchTopKSAE

    torch.manual_seed(0)
    X = torch.randn(256, 16)
    model = BatchTopKSAE(input_dim=16, num_neurons=32, k_active=3, device="cpu")
    hist = model.fit(X, n_epochs=3, patience=2, show_progress=False, batch_size=64)
    assert len(hist["train_loss"]) >= 1

    model.eval()
    x_hat, info = model(X[:8])
    assert x_hat.shape == X[:8].shape
    codes = info["codes"]
    assert codes.shape == (8, 32)
    # Per-sample K activity at eval
    nnz = (codes > 0).sum(dim=1).tolist()
    assert all(n == 3 for n in nnz)


def test_get_activations_numpy_input():
    from saerm.sae.batchtopk_sae import BatchTopKSAE

    X = np.random.RandomState(0).randn(100, 8).astype(np.float32)
    model = BatchTopKSAE(input_dim=8, num_neurons=16, k_active=2, device="cpu")
    model.fit(torch.from_numpy(X), n_epochs=1, patience=1, show_progress=False, batch_size=32)
    acts = model.get_activations(X, batch_size=20, show_progress=False)
    assert acts.shape == (100, 16)




