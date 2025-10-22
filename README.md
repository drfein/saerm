# SAERM — Interpretable Reward Models Pipeline

This repository now focuses on the "Interpretable Reward Models for Robustness" experiment stack. The codebase is organized around three reusable phases:

1. **Embedding caching** – extract hidden representations from transformer checkpoints and persist them with rich metadata.
2. **Sparse feature learning** – train batch Top-K sparse autoencoders (with learning-rate warmup + cosine decay) on cached features to build interpretable latent spaces.
3. **Reward head training / evaluation** – fit controllable reward heads (linear, decision tree, XGBoost, GAM, …) on top of the learned features and evaluate on curated preference datasets.

The entire workflow is driven by a single `config.yaml` so that successive scripts compose without manual wiring.

## Project layout

```
src/saerm/
  config.py        # YAML-backed experiment configuration objects
  data/            # Dataset adapters (Hugging Face, Skywork, RewardBench)
  embeddings/     # Embedding caching logic (HF model + tokenizer)
  sae/            # Sparse autoencoder models, datasets, and trainers
  heads/          # Linear, tree, XGBoost, GAM heads + training helpers
  pipeline/       # Orchestration utilities for end-to-end runs
  storage/        # Filesystem layout helpers for cached artifacts
  utils/          # Shared utilities (slug builders, etc.)
```

`scripts/` holds thin CLI wrappers that call into the modules above:

- `cache_embeddings.py` – run one or more embedding jobs.
- `train_sae.py` – train sparse autoencoders on cached embeddings.
- `train_head.py` – fit configurable reward heads.
- `eval_head.py` – evaluate a stored head on a labelled split.
- `run_pipeline.py` – execute all configured jobs in sequence.

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[gam]  # add [gam] only if you plan to use GAM heads
```

All scripts read `config.yaml` by default; override with `--config path/to/file.yaml`.

### Example workflow

1. Cache embeddings:
   ```bash
   python scripts/cache_embeddings.py --job-id flan-t5-small-skywork-last
   ```
2. Train a sparse autoencoder on those embeddings:
   ```bash
   python scripts/train_sae.py --job-id flan-t5-small-skywork-sae
   ```
3. Fit a reward head:
   ```bash
   python scripts/train_head.py --job-id flan-t5-small-skywork-linear
   ```
4. Evaluate the head:
   ```bash
   python scripts/eval_head.py --job-id flan-t5-small-skywork-linear
   ```

`run_pipeline.py` chains steps 1–3 using the jobs defined in the configuration file and is useful for CI smoke tests or quick reruns.

## Configuration schema

`config.yaml` drives the entire experiment. Key sections:

- `storage`: root directory plus subdirectory names for embeddings, SAEs, and heads.
- `datasets`: named dataset entries (typically Hugging Face ids) used by jobs. Use `field_mapping` to rename raw columns (e.g. map Hugging Face fields to `prompt`, `chosen`, `rejected`).
- `embedding_jobs`: which model/layer/dataset combinations to cache. Each job describes the model id, target layer, dataset key, tokenizer override, and sampling controls. For chat-based sources supply `chat_messages_field` (plus optional `chat_add_generation_prompt` and `chat_template_kwargs`) so `tokenizer.apply_chat_template` can render the conversation faithfully. When you want paired embeddings (e.g. chosen/rejected) from the same dataset row, set `paired_chat_messages_field` or `paired_text_field`; the cache will store both tensors alongside aligned `example_ids` metadata for downstream joins.
- `sae_jobs`: sparse autoencoder jobs that reference an embedding job id and list training hyperparameters.
  - Optional keys: `log_interval`, `wandb_project`, `wandb_entity`, `wandb_name` (enable Weights & Biases logging with live dead-neuron % and batch R² metrics).
- `head_jobs`: reward head jobs that reference embedding/SAE job ids, choose a head type, and name the dataset column that contains numeric targets for supervised training.

The default `config.yaml` gives a working template for Skywork preferences (training) and RewardBench (evaluation). Adjust dataset field mappings or targets to suit your needs.

## Dataset adapters

- `saerm.data.iter_skywork_pairs` – iterate Skywork/Skywork-Reward-Preference-80K-v0.2 examples as `(prompt, chosen, rejected)` triples with metadata.
- `saerm.data.iter_reward_bench_examples` – iterate RewardBench v2 entries (prompt + competing responses + human label).

Use these adapters when crafting custom preprocessing scripts or sanity checks.

## Extending the pipeline

- Add new embedding backbones by inserting entries into `embedding_jobs`.
- Implement alternate feature learners by creating modules under `saerm/sae/` and reusing the storage + config plumbing.
- Introduce new reward heads by subclassing `heads.base.PredictionHead` and registering with `HeadFactory.register("your_head", YourHeadClass)`.
- Build domain-specific evaluation flows by composing dataset adapters with `scripts/eval_head.py` or your own CLI.

## Testing

```bash
pytest
```

Current tests cover config parsing and storage layout helpers. Add integration tests per job when you plug in real models/datasets (these often require Hugging Face auth + large downloads, so keep them opt-in).

## Notes

- Hugging Face datasets/models usually require authentication tokens for private artifacts; export `HF_TOKEN` if necessary.
- Embedding caching defaults to the last hidden state CLS token. Modify `EmbeddingCacheManager` if you need pooled representations or per-token features.
- Decision tree and XGBoost heads provide simple knob-based control. Consider authoring custom heads to add veto/monotonic constraints tailored to your domain.
