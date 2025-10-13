## SAERM quick guide

This repository provides:

- `saerm/process_data/litbench_hf.py` — Loader for the LitBench Train/Test-Enhanced datasets on Hugging Face, emitting `PreferencePair`s with robust field handling.
- `saerm/rm/btrm/` — BTRM: a simple reward model wrapper over a Hugging Face sequence classifier, with a scalarization rule for logits and convenient `.load()`.
- `saerm/sae/batchtopk_sae.py` — Minimal Batch-Top-K Sparse Autoencoder with training, dead-neuron tracking, and batched inference utilities.
- `saerm/sae/mounted.py` — Utility to mount a `BatchTopKSAE` on a layer of a base model via a forward hook and train on raw base inputs.

### Installation

Use your preferred environment; tests assume the package is importable from `src/`.

```bash
pip install -e .  # or ensure src/ on PYTHONPATH
pip install pytest
```

### Running tests

- Unit tests (no internet):

```bash
pytest
```

- Hugging Face integration tests (internet + token):

```bash
export HF_TOKEN=hf_xxx
pytest -m hf
```

Optional: set `BTRM_REPO_ID` to choose a specific model for BTRM loading; by default a tiny DistilBERT SST2 model is used to minimize downloads.

### Module notes

- `LitBenchHF` resolves `chosen`/`rejected` across multiple keys and drops blanks/identical pairs.
- `BTRM` composes prompt/body into a single string and maps logits to a scalar: squeeze if 1-dim, difference if 2-dim, else mean.
- `BatchTopKSAE` trains with batch-wide Top-K sparsification; at eval, enforces per-sample K.
- `MountedSAE` freezes the base model, captures layer activations via a hook, and trains only the SAE.


### API reference

#### `saerm/process_data/litbench_hf.py`

- `class LitBenchHF`
  - `__init__(hf_token: str | None = None, train_repo_id: str = "SAA-Lab/LitBench-Train", test_repo_id: str = "SAA-Lab/LitBench-Test-Enhanced")`
    - Inputs: optional HF token and repo IDs.
    - Side effects: downloads HF datasets for train/test.
  - `train() -> list[PreferencePair]`
    - Output: list of pairs with non-empty, non-identical `chosen/rejected` strings.
  - `test() -> list[PreferencePair]`
    - Output: as above, from the test-enhanced repo.
  - Static: `_get_first_nonempty(ex: dict, keys: list[str]) -> str`
    - Output: first non-empty trimmed field in `keys`, else empty string.

- `dataclass PreferencePair`
  - Fields: `chosen: str`, `rejected: str`.

- `class BasePreferenceDataset`
  - Abstract: `train() -> Sequence[PreferencePair]`, `test() -> Sequence[PreferencePair]`.
  - Helper: `as_text_tuples(split: str = "train") -> list[tuple[str, str]]`.

#### `saerm/rm/btrm/`

- `class BTRM`
  - `__init__(model: Any, tokenizer: Any, max_length: int = 512, device: str = "cpu")`
    - Inputs: HF `PreTrainedModel` (sequence classification) and tokenizer.
  - `reward(body: str, prompt: str | None = None) -> float`
    - Inputs: response text; optional prompt.
    - Output: scalar Python `float` score from model logits.
  - `train(dataset: Any) -> None`
    - Raises `NotImplementedError` in this package variant.
  - `@classmethod load(path: str, config: Mapping[str, Any]) -> BTRM`
    - Inputs: HF repo ID or local directory; config keys: `hf_token`, `max_length`, `device`, `trust_remote_code`.
    - Output: initialized `BTRM` with loaded tokenizer/model on device.
  - Internals:
    - `_compose_input(body, prompt) -> str`: joins prompt/response for scoring.
    - `_scalar_score(logits) -> tensor-like`: squeeze if 1-d, diff if 2-d, else mean over last dim.

- `class BaseRewardModel`
  - Abstract: `reward(body: str, prompt: str | None = None) -> float`, `train(dataset: Any) -> None`, `@classmethod load(path, config) -> BaseRewardModel`.

#### `saerm/sae/batchtopk_sae.py`

- `def criterion(x, x_hat, pre_codes, codes, dictionary) -> torch.Tensor`
  - Inputs: reconstruction terms and activations; includes small revival term.
  - Output: scalar tensor loss.

- `class BatchTopKSAE(nn.Module)`
  - `__init__(input_dim: int, num_neurons: int, k_active: int, *, device: str | None = None)`
    - Inputs: dimensions and K; picks CPU/CUDA by default.
  - `forward(x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]`
    - Input: `x` of shape `(B, input_dim)`.
    - Outputs: `(x_hat: (B, input_dim), info: {pre_codes, codes, dictionary})` with `codes` shape `(B, num_neurons)`.
  - `fit(X_train: torch.Tensor, X_val: torch.Tensor | None = None, *, batch_size=512, learning_rate=5e-4, n_epochs=100, patience=5, clip_grad=1.0, show_progress=True) -> dict[str, list]`
    - Output: history dict with `train_loss` and optional `val_loss` lists.
  - `get_activations(inputs: list | np.ndarray | torch.Tensor, batch_size=8192, show_progress=True) -> np.ndarray`
    - Output: dense codes `(N, num_neurons)` computed in eval mode.

#### `saerm/sae/mounted.py`

- `class MountedSAE(nn.Module)`
  - `__init__(base_model: nn.Module, layer: str | nn.Module, *, sae: BatchTopKSAE | None = None, num_neurons: int | None = None, k_active: int | None = None, device: str | None = None, flatten: bool = True, activation_transform: callable | None = None)`
    - Inputs: base model and layer (dotted path or module); provide `sae` or `(num_neurons, k_active)` for lazy creation; optional transform and flattening.
    - Behavior: freezes base model params and registers a forward hook on the target layer.
  - `forward(x, *, return_base_output: bool = False) -> tuple[torch.Tensor, dict]`
    - Inputs: base-model inputs (tensor, dict of tensors, list/tuple as expected by `base_model`).
    - Outputs: `(x_hat, info)` where `info['codes']` are SAE codes; includes `base_output` if requested.
  - `fit(X_train, X_val: object | None = None, *, batch_size=64, learning_rate=5e-4, n_epochs=10, patience=3, clip_grad=1.0, show_progress=True) -> dict[str, list]`
    - Inputs: raw base-model inputs (tensor, dict with tensors, or list/tuple items).
    - Output: history dict with `train_loss` and optional `val_loss`.
  - `get_activations(inputs: torch.Tensor, batch_size: int = 256, show_progress: bool = True) -> torch.Tensor`
    - Output: SAE codes `(N, num_neurons)`.


