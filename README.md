# Paper Experiments

This branch contains the code and results for the paper experiments. All links below point to files on this branch.

---

## Prompts / Datasets

| Experiment | Dataset | Prompt format |
|------------|---------|---------------|
| Length bias (probe + eval) | GSM8K questions from `data/gsm8k_soln.json` (generated via `meta-llama/Meta-Llama-3.1-8B-Instruct`). Each item has a `question`, a concise correct solution, a verbose correct solution, and short/long incorrect solutions. | `[{"role":"user","content":<question>},{"role":"assistant","content":<response>}]` — rendered with the reward model's own chat template via `apply_chat_template`. DeBERTa uses pair encoding `tokenizer(prompt, response)`. |
| Huang et al. LWR baseline | Same GSM8K data as above. | Same as length bias. |
| BoN candidate generation | AlpacaEval instructions (`alpaca_eval.json`, field `instruction`). | Chat template of the generator model (Llama-3.2-1B-Instruct). |
| PPO training | AlpacaEval `instruction` field. First 512 examples = train split; remainder = eval. | Chat template of the policy model (Llama-3.2-1B-Instruct). |
| Style bias (NLL split) | `allenai/tulu-3-wildchat-reused-on-policy-8b` (train split, 2400 prompts, seed 10). Each item is a `(prompt, completion)` pair. | Completion NLL scored per-byte: `s_m = -NLL_m(completion \| prompt) / bytes(completion)` for each of 10 LMs; label = 1 if `s_qwen3-8B − s_llama2-7b > median`. |

**Generative LMs used for style NLL scoring:** `google/gemma-2-2b-it`, `google/gemma-2-9b-it`, `google/gemma-3-12b-it`, `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf`, `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-8B`.

---

## Core Methodology

| File | Description |
|------|-------------|
| [src/nb/nullbias/probe.py](https://github.com/drfein/saerm/blob/paper-experiments/src/nb/nullbias/probe.py) | Null-space projection (Gram-Schmidt, difference-of-means probe) |
| [src/nb/experiments/base.py](https://github.com/drfein/saerm/blob/paper-experiments/src/nb/experiments/base.py) | Train/test split logic, experiment runner |
| [src/nb/datasets/base.py](https://github.com/drfein/saerm/blob/paper-experiments/src/nb/datasets/base.py) | Hash-based deterministic splitting |
| [src/nb/datasets/length.py](https://github.com/drfein/saerm/blob/paper-experiments/src/nb/datasets/length.py) | Length bias dataset & metrics |
| [src/nb/experiments/length.py](https://github.com/drfein/saerm/blob/paper-experiments/src/nb/experiments/length.py) | Length bias experiment |

---


## Huang et al. (2025) Comparison

> Allen: 65.1% (Huang) vs 74.5% (ours). DeBERTa: 55.5% (Huang) vs 46.1% (ours).

| File | Description |
|------|-------------|
| [scripts/compare_saved_debias_vs_lwr_probefit.py](https://github.com/drfein/saerm/blob/paper-experiments/scripts/compare_saved_debias_vs_lwr_probefit.py) | LWR (Huang et al.) baseline implementation |
| [slurm/run_huang_comparison.sbatch](https://github.com/drfein/saerm/blob/paper-experiments/slurm/run_huang_comparison.sbatch) | Job to regenerate and save 65.1%/55.5% numbers |

The LWR curve is fit on the probe split only (`lwr-frac=0.5`, `lwr-alpha=1.0`), evaluated on the held-out test split — same data partition as our method.

---

## Best-of-N (BoN)

> DeBERTa baseline 228.6 words → debiased 227.5 words at N=32. GPT-5 win rate: debiased 17.19% vs baseline 13.28%.

| File | Description |
|------|-------------|
| [scripts/generate_bon_candidates.py](https://github.com/drfein/saerm/blob/paper-experiments/scripts/generate_bon_candidates.py) | Candidate pool generation |
| [scripts/score_saved_candidates_multi_rm.py](https://github.com/drfein/saerm/blob/paper-experiments/scripts/score_saved_candidates_multi_rm.py) | BoN scoring (DeBERTa + Allen, N=1,2,4,8,16,32,64) |
| [scripts/evaluate_openai_win_rate.py](https://github.com/drfein/saerm/blob/paper-experiments/scripts/evaluate_openai_win_rate.py) | GPT-5 judge win-rate evaluation |
| [scripts/export_multi_rm_n_winners.py](https://github.com/drfein/saerm/blob/paper-experiments/scripts/export_multi_rm_n_winners.py) | Export winning responses per N |
| [slurm/run_multi_rm_bon_score.sbatch](https://github.com/drfein/saerm/blob/paper-experiments/slurm/run_multi_rm_bon_score.sbatch) | BoN scoring job |
| [slurm/run_hh_openai_winrate_eval.sbatch](https://github.com/drfein/saerm/blob/paper-experiments/slurm/run_hh_openai_winrate_eval.sbatch) | Win-rate eval job |
| [results/maty/bon_ppo_deberta_alpacaeval/quick_length_summary.json](https://github.com/drfein/saerm/blob/paper-experiments/results/maty/bon_ppo_deberta_alpacaeval/quick_length_summary.json) | BoN length results (baseline 228.6 → debiased 227.5 words) |
| [results/maty/bon_ppo_deberta_alpacaeval/baseline_summary.json](https://github.com/drfein/saerm/blob/paper-experiments/results/maty/bon_ppo_deberta_alpacaeval/baseline_summary.json) | Baseline policy generation summary |
| [results/maty/bon_ppo_deberta_alpacaeval/debiased_summary.json](https://github.com/drfein/saerm/blob/paper-experiments/results/maty/bon_ppo_deberta_alpacaeval/debiased_summary.json) | Debiased policy generation summary |

---

## PPO

> Llama-3.2-1B-Instruct, 512 steps, batch size 16. AlpacaEval train/eval split: 512 train, remainder eval.

| File | Description |
|------|-------------|
| [scripts/train_trl_rm_ppo.py](https://github.com/drfein/saerm/blob/paper-experiments/scripts/train_trl_rm_ppo.py) | PPO training script (TRL, Llama-3.2-1B-Instruct) |
| [scripts/prepare_alpacaeval_ppo_split.py](https://github.com/drfein/saerm/blob/paper-experiments/scripts/prepare_alpacaeval_ppo_split.py) | AlpacaEval train/eval split (512 train, remainder eval) |
| [scripts/measure_generation_length.py](https://github.com/drfein/saerm/blob/paper-experiments/scripts/measure_generation_length.py) | Response length measurement |
| [slurm/run_trl_rm_ppo_standalone.sbatch](https://github.com/drfein/saerm/blob/paper-experiments/slurm/run_trl_rm_ppo_standalone.sbatch) | PPO training job |
| [slurm/run_hh_length_pipeline_ppo.sbatch](https://github.com/drfein/saerm/blob/paper-experiments/slurm/run_hh_length_pipeline_ppo.sbatch) | HH pipeline PPO job |
| [results/maty/ppo_sycophancy_llama32_3b/ppo_debiased_eval_summary.json](https://github.com/drfein/saerm/blob/paper-experiments/results/maty/ppo_sycophancy_llama32_3b/ppo_debiased_eval_summary.json) | Debiased PPO eval results |
| [results/maty/ppo_sycophancy_llama32_3b/ppo_vanilla_nulled_eval_summary.json](https://github.com/drfein/saerm/blob/paper-experiments/results/maty/ppo_sycophancy_llama32_3b/ppo_vanilla_nulled_eval_summary.json) | Vanilla nulled PPO eval results |

---

## INLP / Iterative Nullspace Projection

> Simple biases: AUC > 0.95 before projection, < 0.6 after. Style/complex: AUC ≈ 0.582 (not linearly separable).

| File | Description |
|------|-------------|
| [scripts/analyze_iterative_nulling.py](https://github.com/drfein/saerm/blob/paper-experiments/scripts/analyze_iterative_nulling.py) | INLP analysis — iterative AUC table |
| [scripts/plot_inlp_tsne.py](https://github.com/drfein/saerm/blob/paper-experiments/scripts/plot_inlp_tsne.py) | INLP + t-SNE panel plot |
| [scripts/rebuild_all_rms_probes_with_activations.py](https://github.com/drfein/saerm/blob/paper-experiments/scripts/rebuild_all_rms_probes_with_activations.py) | Rebuild bias probes with activations (all 5 RMs) |
| [slurm/run_all_inlp_and_style.sbatch](https://github.com/drfein/saerm/blob/paper-experiments/slurm/run_all_inlp_and_style.sbatch) | Master job: bias INLP + style INLP for all 5 RMs |

**Style probe (NLL-based)**

| File | Description |
|------|-------------|
| [scripts/compute_style_nll_scores.py](https://github.com/drfein/saerm/blob/paper-experiments/scripts/compute_style_nll_scores.py) | Per-byte NLL scores for 10 generative LMs on tulu-3-wildchat |
| [scripts/build_style_probe_per_rm.py](https://github.com/drfein/saerm/blob/paper-experiments/scripts/build_style_probe_per_rm.py) | Style probe activations per RM (split by Qwen3-8B vs Llama-2-7b delta) |
