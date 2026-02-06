# Mechanistic Reward Shaping

Evaluates and mitigates spurious biases in reward models using null-space projection.

## Biases Evaluated

- **Position**: Preference for answer positions (A/B/C/D) in MCQ
- **Sycophancy**: Agreement with user's stated opinion
- **Length**: Preference for longer responses
- **Uncertainty**: Penalizing hedged/uncertain language

## Method

1. Build a **probe direction** from contrastive pairs (e.g., same content at position A vs B)
2. **Project out** the probe direction from hidden states via null-space projection
3. Evaluate whether bias is reduced without harming accuracy

## Usage

```bash
# Run a single experiment
python experiments/run_experiment.py --config experiments/configs/position_skywork_gsm8k.yaml

# Run with CLI overrides
python experiments/run_experiment.py \
    --bias_type position \
    --model_path Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --dataset_source guipenedo/gsm8k-mc
```

## Structure

```
src/nb/
├── datasets/       # Dataset loading & formatting
├── experiments/    # Experiment orchestration
└── nullbias/       # Probe building & projection

experiments/
├── configs/        # YAML experiment configs
├── run_experiment.py
└── run_all.py
```

## Requirements

```bash
pip install -r requirements.txt
```
