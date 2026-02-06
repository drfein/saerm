
"""
Script to evaluate model-style sensitivty data

Note: The analysis calculates not actual perplexity, as it never exponentiates the NLL, but still calls it ppl in parts

run with 
uv run experiments/eval_perplexity.py
"""

from typing import Dict, Optional, Union, List, Any

from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_ppl = {
    "gemma-2-2b-it": {"model_id": "google/gemma-2-2b-it", "type": "perplexity", "file": "perplexity_google__gemma-2-2b-it.csv"},
    "gemma-2-9b-it": {"model_id": "google/gemma-2-9b-it", "type": "perplexity", "file": "perplexity_google__gemma-2-9b-it.csv"},
    "gemma-3-12b-it": {"model_id": "google/gemma-3-12b-it", "type": "perplexity", "file": "perplexity_google__gemma-3-12b-it.csv"},
    "Llama-2-13b-chat-hf": {"model_id": "meta-llama/Llama-2-13b-chat-hf", "type": "perplexity", "file": "perplexity_meta-llama__Llama-2-13b-chat-hf.csv"},
    "Llama-2-7b-chat-hf": {"model_id": "meta-llama/Llama-2-7b-chat-hf", "type": "perplexity", "file": "perplexity_meta-llama__Llama-2-7b-chat-hf.csv"},
    "Llama-3.1-8B-Instruct": {"model_id": "meta-llama/Llama-3.1-8B-Instruct", "type": "perplexity", "file": "perplexity_meta-llama__Llama-3.1-8B-Instruct.csv"},
    "Qwen2.5-0.5B-Instruct": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "type": "perplexity", "file": "perplexity_Qwen__Qwen2.5-0.5B-Instruct.csv"},
    "Qwen2.5-7B-Instruct": {"model_id": "Qwen/Qwen2.5-7B-Instruct", "type": "perplexity", "file": "perplexity_Qwen__Qwen2.5-7B-Instruct.csv"},
    # "Qwen3-0.6B_thinking": {"model_id": "Qwen/Qwen3-0.6B", "type": "perplexity", "file": "perplexity_Qwen__Qwen3-0.6B_thinking.csv"},
    "Qwen3-0.6B_nothink": {"model_id": "Qwen/Qwen3-0.6B", "type": "perplexity", "file": "perplexity_Qwen__Qwen3-0.6B.csv"},
    # "Qwen3-8B_thinking": {"model_id": "Qwen/Qwen3-8B", "type": "perplexity", "file": "perplexity_Qwen__Qwen3-8B_thinking.csv"},
    "Qwen3-8B_nothink": {"model_id": "Qwen/Qwen3-8B", "type": "perplexity", "file": "perplexity_Qwen__Qwen3-8B.csv"},
}
data_rewards = {
    "RM_Llama-3.1-8B": {"model_id": "allenai/Llama-3.1-8B-Instruct-RM-RB2", "type": "reward", "file": "rewards_allenai__Llama-3.1-8B-Instruct-RM-RB2.csv"},
    "Skywork_Llama-3.1-8B": {"model_id": "Skywork/Skywork-Reward-V2-Llama-3.1-8B", "type": "reward", "file": "rewards_Skywork__Skywork-Reward-V2-Llama-3.1-8B.csv"},
    "Skywork_Qwen3-0.6B": {"model_id": "Skywork/Skywork-Reward-V2-Qwen3-0.6B", "type": "reward", "file": "rewards_Skywork__Skywork-Reward-V2-Qwen3-0.6B.csv"},
    "Skywork_Qwen3-8B": {"model_id": "Skywork/Skywork-Reward-V2-Qwen3-8B", "type": "reward", "file": "rewards_Skywork__Skywork-Reward-V2-Qwen3-8B.csv"},
    "RM_Deberta": {"model_id": "OpenAssistant/reward-model-deberta-v3-large-v2", "type": "reward", "file": "rewards_OpenAssistant__reward-model-deberta-v3-large-v2.csv"},
}

path_to_data = "perplexity_data/"

@dataclass
class AnalysisPair:
    base_key: str
    models: List[str]
    reward: Dict[str, Any]
    family_name: Optional[str] = None

analysis_pairs = [
    AnalysisPair(
        base_key="Llama-3.1-8B-Instruct",
        models=[k for k in data_ppl], # [k for k in data_ppl if k != "RM_Llama-3.1-8B"],
        reward=data_rewards["RM_Llama-3.1-8B"],
        family_name="Llama",
    ),
    AnalysisPair(
        base_key="Llama-3.1-8B-Instruct",
        models=[k for k in data_ppl],
        reward=data_rewards["Skywork_Llama-3.1-8B"],
        family_name="Llama",
    ),
    AnalysisPair(
        base_key="Qwen3-8B_thinking", # "Qwen3-8B_nothink",
        models=[k for k in data_ppl],
        reward=data_rewards["Skywork_Qwen3-8B"],
        family_name="Qwen",
    ),
    AnalysisPair(
        base_key="Qwen3-0.6B_thinking", #, "Qwen3-0.6B_nothink",
        models=[k for k in data_ppl],
        reward=data_rewards["Skywork_Qwen3-0.6B"],
        family_name="Qwen",
    ),
    AnalysisPair(
        base_key="None",
        models=[k for k in data_ppl],
        reward=data_rewards["RM_Deberta"],
    ),
]

def bootstrap_spearman_ci(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    group_col: str = "dataset_id",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Cluster bootstrap Spearman correlation with percentile CI.
    Resamples group_col (dataset_id) with replacement.
    """
    rng = np.random.default_rng(seed)
    groups = df[group_col].unique()
    if len(groups) < 2:
        return {"rho": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    # Point estimate
    rho = float(df[x_col].corr(df[y_col], method="spearman"))

    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        # keep both chosen/rejected rows for each sampled id
        bs = df[df[group_col].isin(sampled)]
        boot[b] = bs[x_col].corr(bs[y_col], method="spearman")

    lo = float(np.nanquantile(boot, alpha / 2))
    hi = float(np.nanquantile(boot, 1 - alpha / 2))
    return {"rho": rho, "ci_lo": lo, "ci_hi": hi}


def main():

    NLL_NORM = "bytes" # use  "token", "bytes", "chars"
    CORR_KIND = "spearman"   # use "pearson" or "spearman"
    DO_FAMILY_SPLIT = False
    PANEL_FUNC = "median" # use "mean or "median"


    print("-------------------------------")
    # Do analysis here.
    for a_pair in analysis_pairs:
        # ---- Load reward + perplexity data once per analysis pair ----
        reward_df = pd.read_csv(path_to_data + a_pair.reward["file"])
        reward_df = reward_df[["dataset_id", "which", "reward"]].dropna()
        reward_s = reward_df.set_index(["dataset_id", "which"])["reward"]

        # Sanity: reward keys should be unique
        if not reward_s.index.is_unique:
            dup = reward_s.index[reward_s.index.duplicated()].unique()
            raise ValueError(f"Reward file has duplicate (dataset_id, which) keys, e.g. {list(dup[:5])}")

        # Load all perplexity CSVs into one wide table: index=(dataset_id, which), columns=model_key, values=s
        ppl_long_parts = []
        for k, cfg in data_ppl.items():
            df = pd.read_csv(path_to_data + cfg["file"])

            needed = {"dataset_id", "which", "nll_sum", "nll_mean_token", "norm_n_y_bytes", "norm_n_y_chars"}
            missing = needed - set(df.columns)
            if missing:
                raise KeyError(f"{cfg['file']} missing required columns: {sorted(missing)}")

            df = df[["dataset_id", "which", "nll_sum", "nll_mean_token", "norm_n_y_bytes", "norm_n_y_chars"]].copy()

            if NLL_NORM == "token":
                df["s"] = -df["nll_mean_token"]
            elif NLL_NORM == "bytes":
                denom = df["norm_n_y_bytes"].astype(float)
                df["s"] = -(df["nll_sum"].astype(float) / denom.replace(0.0, np.nan))
            elif NLL_NORM == "chars":
                denom = df["norm_n_y_chars"].astype(float)
                df["s"] = -(df["nll_sum"].astype(float) / denom.replace(0.0, np.nan))
            else:
                raise ValueError(f"Unknown NLL_NORM={NLL_NORM!r}. Use 'token', 'bytes', or 'chars'.")

            df["model_key"] = k
            ppl_long_parts.append(df[["dataset_id", "which", "model_key", "s"]])

        ppl_long = pd.concat(ppl_long_parts, ignore_index=True)
        ppl_wide = ppl_long.pivot_table(
            index=["dataset_id", "which"],
            columns="model_key",
            values="s",
            aggfunc="first",
        )

        # Sanity: perplexity keys should be unique (pivot_table with 'first' can hide duplicates)
        if ppl_long.duplicated(subset=["dataset_id", "which", "model_key"]).any():
            raise ValueError("Perplexity data has duplicate (dataset_id, which, model_key) rows; pivot_table would hide this.")


        # ---- Build family mapping from data_ppl metadata (best-effort) ----
        def _infer_family(model_id: str) -> str:
            mid = model_id.lower()
            if "llama" in mid:
                return "Llama"
            if "qwen" in mid:
                return "Qwen"
            if "gemma" in mid:
                return "Gemma"
            return "Other"

        model_key_to_family = {k: _infer_family(cfg["model_id"]) for k, cfg in data_ppl.items()}
        family = a_pair.family_name  # e.g., "Llama" or "Qwen"

        # Prepare two plots per reward model: (A) full panel, (B) other-family-only panel
        plot_specs = [
                ("all_models_panel", "Panel = all other models", None),  # None => use all except m
            ]
        if DO_FAMILY_SPLIT:
            plot_specs.append(("other_family_panel", "Panel = only other-family models", "other_family_only"))

        for plot_tag, plot_title_suffix, panel_mode in plot_specs:
            fig, ax = plt.subplots(figsize=(10, 6))
            corr_rows = []

            for m in a_pair.models:
                # ---- Panel construction ----
                panel_all = [k for k in data_ppl if k != m]

                if panel_mode is None:
                    panel = panel_all
                else:
                    # Only models NOT in the same family as a_pair.family_name
                    # (and still exclude m, even if it is not in-family)
                    panel = [k for k in panel_all if model_key_to_family.get(k, "Other") != family]

                # Need base model column + at least 1 panel column
                if m not in ppl_wide.columns:
                    continue
                panel = [p for p in panel if p in ppl_wide.columns]
                if len(panel) == 0:
                    continue

                # Compute s_delta on intersection of available rows
                if PANEL_FUNC == "mean":
                    s_delta = ppl_wide[m] - ppl_wide[panel].mean(axis=1)
                elif PANEL_FUNC == "median":
                    s_delta = ppl_wide[m] - ppl_wide[panel].median(axis=1)
                else:
                    raise ValueError(f"Unknown PANEL_FUNC={PANEL_FUNC!r}. Use 'mean' or 'median'.")

                # Join with reward scores; drop NaNs/infs
                tmp = pd.concat([s_delta.rename("s_delta"), reward_s.rename("reward")], axis=1, join="inner")
                tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["s_delta", "reward"])
                if len(tmp) == 0:
                    continue

                # tmp currently has index=(dataset_id, which). Turn dataset_id into a column for bootstrapping.
                tmp2 = tmp.reset_index()  # gives columns: dataset_id, which, s_delta, reward

                if CORR_KIND == "spearman":
                    ci = bootstrap_spearman_ci(
                        tmp2, "s_delta", "reward",
                        group_col="dataset_id",
                        n_boot=2000,
                        alpha=0.05,
                        seed=0,
                    )
                    corr = ci["rho"]
                    corr_rows.append({
                        "model": m,
                        "spearman_r": corr,
                        "ci95_lo": ci["ci_lo"],
                        "ci95_hi": ci["ci_hi"],
                        "n": int(len(tmp2)),
                        "n_ids": int(tmp2["dataset_id"].nunique()),
                    })
                else:
                    corr = float(tmp["s_delta"].corr(tmp["reward"], method=CORR_KIND))
                    corr_rows.append({"model": m, f"{CORR_KIND}_r": corr, "n": int(len(tmp))})


                ax.scatter(
                    tmp["s_delta"].to_numpy(),
                    tmp["reward"].to_numpy(),
                    s=10,
                    alpha=0.35,
                    label=f"{m} ({CORR_KIND[0]}={corr:.3f}, n={len(tmp)})",
                )

            ax.set_xlabel(f"Panel-relative log-prob (s_delta = s_base - {PANEL_FUNC}(panel)) [ ]")
            ax.set_ylabel("Reward model score [ ]")
            ax.set_title(
                f"Reward vs panel-relative perplexity: {a_pair.base_key} | {plot_title_suffix} | "
                f"NLL_NORM={NLL_NORM} | CORR={CORR_KIND}"
            )
            ax.set_xlim(-3., 3.)
            ax.grid(True, alpha=0.2)
            if len(corr_rows) > 0:
                ax.legend(fontsize=8, loc="best", frameon=True)
            plt.tight_layout()

            out_plot = (
                f"scatter_reward_vs_sdelta__{a_pair.reward['model_id'].replace('/','__')}__"
                f"{a_pair.base_key}__{plot_tag}__{NLL_NORM}__{CORR_KIND}.png"
            )
            plt.savefig(out_plot, dpi=300)

            corr_df = pd.DataFrame(corr_rows).sort_values(f"{CORR_KIND}_r", ascending=False)
            corr_df["ci95_minus"] = corr_df["spearman_r"] - corr_df["ci95_lo"]
            corr_df["ci95_plus"] = corr_df["ci95_hi"] - corr_df["spearman_r"]
            out_corr = (
                f"corr_reward_vs_sdelta__{a_pair.reward['model_id'].replace('/','__')}__"
                f"{a_pair.base_key}__{plot_tag}__{NLL_NORM}__{CORR_KIND}.csv"
            )
            corr_df.to_csv(out_corr, index=False)

            print(f"\nReward Model {a_pair.reward['model_id']}")
            mean_abs_corr = corr_df["spearman_r"].abs().mean()
            print(f"Mean absolute correlation {mean_abs_corr}")
            print(corr_df)


if __name__ == "__main__":
    main()


"""
Reward Model allenai/Llama-3.1-8B-Instruct-RM-RB2
                   model  spearman_r   ci95_lo   ci95_hi     n  n_ids  ci95_minus  ci95_plus
7    Qwen2.5-7B-Instruct    0.271152  0.248057  0.294966  4790   2400    0.023095   0.023814
5  Llama-3.1-8B-Instruct    0.113644  0.090922  0.138533  4790   2400    0.022722   0.024888
9       Qwen3-8B_nothink    0.064027  0.038683  0.089209  4790   2400    0.025344   0.025182
3    Llama-2-13b-chat-hf    0.043216  0.016395  0.069784  4790   2400    0.026822   0.026567
4     Llama-2-7b-chat-hf    0.014120 -0.012630  0.040703  4790   2400    0.026749   0.026583
8     Qwen3-0.6B_nothink   -0.012246 -0.038052  0.013489  4790   2400    0.025806   0.025735
2         gemma-3-12b-it   -0.020939 -0.047839  0.003846  4790   2400    0.026900   0.024784
0          gemma-2-2b-it   -0.030736 -0.054955 -0.006732  4790   2400    0.024219   0.024003
6  Qwen2.5-0.5B-Instruct   -0.111846 -0.137316 -0.086470  4790   2400    0.025470   0.025376
1          gemma-2-9b-it   -0.120676 -0.146460 -0.094896  4790   2400    0.025784   0.025779

Reward Model Skywork/Skywork-Reward-V2-Llama-3.1-8B
                   model  spearman_r   ci95_lo   ci95_hi     n  n_ids  ci95_minus  ci95_plus
7    Qwen2.5-7B-Instruct    0.249792  0.227009  0.273200  4790   2400    0.022784   0.023407
5  Llama-3.1-8B-Instruct    0.190900  0.168269  0.212894  4790   2400    0.022632   0.021994
9       Qwen3-8B_nothink    0.075041  0.050064  0.100345  4790   2400    0.024977   0.025303
8     Qwen3-0.6B_nothink    0.053615  0.027994  0.079363  4790   2400    0.025621   0.025748
1          gemma-2-9b-it    0.050142  0.024139  0.075018  4790   2400    0.026004   0.024875
2         gemma-3-12b-it    0.010387 -0.015785  0.036390  4790   2400    0.026172   0.026003
6  Qwen2.5-0.5B-Instruct   -0.035134 -0.059779 -0.009474  4790   2400    0.024645   0.025660
0          gemma-2-2b-it   -0.044720 -0.069587 -0.022258  4790   2400    0.024867   0.022462
3    Llama-2-13b-chat-hf   -0.183012 -0.207715 -0.158097  4790   2400    0.024703   0.024914
4     Llama-2-7b-chat-hf   -0.214426 -0.239964 -0.189495  4790   2400    0.025537   0.024932

Reward Model Skywork/Skywork-Reward-V2-Qwen3-8B
                   model  spearman_r   ci95_lo   ci95_hi     n  n_ids  ci95_minus  ci95_plus
7    Qwen2.5-7B-Instruct    0.239462  0.216017  0.263234  4790   2400    0.023445   0.023773
5  Llama-3.1-8B-Instruct    0.166201  0.141812  0.189627  4790   2400    0.024390   0.023426
8     Qwen3-0.6B_nothink    0.077730  0.053650  0.102351  4790   2400    0.024080   0.024620
9       Qwen3-8B_nothink    0.058757  0.031998  0.083658  4790   2400    0.026760   0.024900
1          gemma-2-9b-it   -0.014692 -0.039764  0.010102  4790   2400    0.025072   0.024793
0          gemma-2-2b-it   -0.016373 -0.040781  0.006766  4790   2400    0.024408   0.023139
6  Qwen2.5-0.5B-Instruct   -0.024934 -0.049846 -0.000090  4790   2400    0.024912   0.024844
2         gemma-3-12b-it   -0.088071 -0.113696 -0.062378  4790   2400    0.025624   0.025693
3    Llama-2-13b-chat-hf   -0.092691 -0.117960 -0.067419  4790   2400    0.025269   0.025272
4     Llama-2-7b-chat-hf   -0.121637 -0.148711 -0.096683  4790   2400    0.027074   0.024954

Reward Model Skywork/Skywork-Reward-V2-Qwen3-0.6B
                   model  spearman_r   ci95_lo   ci95_hi     n  n_ids  ci95_minus  ci95_plus
7    Qwen2.5-7B-Instruct    0.369994  0.347817  0.391438  4790   2400    0.022177   0.021445
9       Qwen3-8B_nothink    0.150324  0.125757  0.174911  4790   2400    0.024567   0.024587
5  Llama-3.1-8B-Instruct    0.099439  0.075458  0.123759  4790   2400    0.023981   0.024320
2         gemma-3-12b-it    0.010865 -0.015454  0.036433  4790   2400    0.026319   0.025568
8     Qwen3-0.6B_nothink   -0.046793 -0.070955 -0.021588  4790   2400    0.024162   0.025205
0          gemma-2-2b-it   -0.047554 -0.072092 -0.022943  4790   2400    0.024539   0.024611
1          gemma-2-9b-it   -0.048579 -0.075037 -0.023146  4790   2400    0.026458   0.025433
3    Llama-2-13b-chat-hf   -0.084922 -0.110308 -0.059568  4790   2400    0.025385   0.025354
4     Llama-2-7b-chat-hf   -0.121986 -0.148102 -0.097805  4790   2400    0.026117   0.024180
6  Qwen2.5-0.5B-Instruct   -0.198550 -0.222778 -0.174160  4790   2400    0.024228   0.024390

Reward Model OpenAssistant/reward-model-deberta-v3-large-v2
                   model  spearman_r   ci95_lo   ci95_hi     n  n_ids  ci95_minus  ci95_plus
7    Qwen2.5-7B-Instruct    0.179020  0.155119  0.202353  4790   2400    0.023901   0.023333
4     Llama-2-7b-chat-hf    0.158342  0.133708  0.182419  4790   2400    0.024635   0.024077
3    Llama-2-13b-chat-hf    0.136089  0.111813  0.160780  4790   2400    0.024276   0.024691
9       Qwen3-8B_nothink    0.129333  0.106320  0.153184  4790   2400    0.023014   0.023850
8     Qwen3-0.6B_nothink    0.043074  0.017619  0.068405  4790   2400    0.025455   0.025331
0          gemma-2-2b-it    0.032198  0.007425  0.055519  4790   2400    0.024773   0.023321
5  Llama-3.1-8B-Instruct   -0.008894 -0.032618  0.013788  4790   2400    0.023724   0.022682
2         gemma-3-12b-it   -0.042500 -0.066389 -0.016955  4790   2400    0.023890   0.025545
6  Qwen2.5-0.5B-Instruct   -0.070996 -0.095525 -0.045834  4790   2400    0.024528   0.025162
1          gemma-2-9b-it   -0.226576 -0.250829 -0.202886  4790   2400    0.024253   0.023690

reward models are sensitive to model-specific “likelihood/style” features
    - not just an artifact of a particular tokenization norm
    - good representation of tested inputs to avoid model dependences
    - also holds with different correlation metrics (pearson/spearman) and aggregations (mean/median)
    - Model capability does not seem to dictate correlation strength

This can depend on two things: 
- Which models were used during RM training synthetic data generation
- Potentially on which base model was used to create RM
"""