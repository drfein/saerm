import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_dim=10, hid=7, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, out_dim))

    def forward(self, x):
        return self.net(x)


def test_mounted_sae_forward_and_fit():
    from saerm.sae.mounted import MountedSAE

    base = ToyModel(in_dim=10, hid=6, out_dim=3)
    # Mount on first linear layer of the Sequential
    mounted = MountedSAE(base_model=base, layer="net.0", num_neurons=12, k_active=2, flatten=True)

    X = torch.randn(128, 10)
    # First forward initializes SAE lazily
    _ = base(X[:4])
    x_hat, info = mounted(X[:4], return_base_output=True)
    assert "codes" in info and info["codes"].shape[1] == 12
    assert "base_output" in info

    # Train the mounted SAE on raw base inputs
    hist = mounted.fit(X, n_epochs=2, patience=2, batch_size=32, show_progress=False)
    assert len(hist["train_loss"]) >= 1

    # Get activations path
    with torch.no_grad():
        Z = mounted.get_activations(X[:32], batch_size=16, show_progress=False)
    assert Z.shape[0] == 32 and Z.shape[1] == 12




