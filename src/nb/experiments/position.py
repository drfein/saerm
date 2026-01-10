"""
Position bias experiment for multiple-choice questions.

Tests whether reward models prefer certain answer positions (A/B/C/D).

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from src.nb.datasets.base import EvalExample, ProbeDataset
from src.nb.datasets.position import (
    CorrectnessPositionFreeformDataset,
    CorrectnessPositionFreeformPlausibleQADataset,
    CorrectnessPositionPlausibleQADataset,
    CorrectnessPositionPlausibleQAMCQDataset,
    CorrectnessPositionMCQDataset,
    PositionBiasDataset, 
    PositionPlausibleQADataset,
    PositionPlausibleQAMCQDataset,
    PositionFreeformDataset,
    PositionFreeformPlausibleQADataset,
    compute_position_metrics, 
    compute_binary_position_metrics,
    compute_freeform_position_metrics,
    POSITION_LABELS,
)
from src.nb.datasets.bigbench import (
    PositionBigBenchDataset,
    CorrectnessPositionBigBenchDataset,
    PositionFreeformBigBenchDataset,
    CorrectnessPositionFreeformBigBenchDataset,
)
from src.nb.experiments.base import BiasExperiment, ExperimentConfig, ExperimentResults
from src.nb.experiments.plotting import (
    create_position_bias_plot,
    create_binary_position_plot,
    create_freeform_position_plot,
    create_accuracy_plot,
)
from src.nb.nullbias.probe import (
    build_probe_direction,
    clean_probe_basis_with_correctness,
    clean_probe_with_correctness,
    get_base_model,
    get_embeddings,
    get_score_head,
    gram_schmidt,
)

logger = logging.getLogger(__name__)


class PositionBiasExperiment(BiasExperiment):
    """Experiment for evaluating position bias on MCQ.
    
    Tests whether RM prefers certain answer positions regardless of content.
    Uses a multi-probe basis capturing each position (A/B/C/D) vs the other positions.
    
    Key metrics:
    - Accuracy: Does highest-reward position match correct answer?
    - Accuracy per position: Accuracy when correct is at A, B, C, D
    - Position distribution: Should be uniform (25% each) without bias
    """
    
    @property
    def bias_type(self) -> str:
        return "position"
    
    def _create_dataset(self) -> ProbeDataset:
        """Create position bias dataset."""
        extra = self.config.extra
        dataset_class = extra.get("dataset_class", "position")
        
        # Check for BigBench dataset
        if dataset_class == "position_bigbench":
            return PositionBigBenchDataset(
                source=self.config.dataset_source,
                probe_tasks=extra.get("probe_tasks"),
                eval_tasks=extra.get("eval_tasks"),
                max_per_task_probe=extra.get("max_per_task_probe", 20),
                max_per_task_eval=extra.get("max_per_task_eval"),
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
        elif dataset_class == "position_plausibleqa_mcq":
            return PositionPlausibleQAMCQDataset(
                source=self.config.dataset_source,
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
        else:
            # Default: MCQ dataset (GSM8K-MC, MMLU, etc.)
            return PositionBiasDataset(
                source=extra.get("dataset_id", "guipenedo/gsm8k-mc"),
                split=extra.get("train_split", "train"),
                eval_split=extra.get("eval_split", "test"),
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
                subset=extra.get("subset", None),
            )
    
    def build_probe(self) -> Dict[str, Any]:
        """Build position bias probe basis using position-vs-rest directions.
        
        Computes up to 4 directions:
          p_A = mean(pos=A) - mean(pos in {B,C,D})
          p_B = mean(pos=B) - mean(pos in {A,C,D})
          p_C = mean(pos=C) - mean(pos in {A,B,D})
          p_D = mean(pos=D) - mean(pos in {A,B,C})
        and orthonormalizes them via Gram–Schmidt to form a basis.
        
        Optionally projects out a correctness direction learned from a correctness probe
        dataset that is balanced across positions.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")
        
        logger.info("Building position bias probe basis (A/B/C/D vs rest)...")
        
        # Get texts organized by position
        texts_by_pos = self.dataset.get_position_embeddings_texts(self.tokenizer)
        
        # Extract embeddings for all positions
        embeddings_by_pos = {}
        for pos in [0, 1, 2, 3]:
            logger.info("Extracting embeddings for position %s (%d texts)...", 
                       POSITION_LABELS[pos], len(texts_by_pos[pos]))
            embeddings_by_pos[pos] = get_embeddings(
                model=self.model,
                tokenizer=self.tokenizer,
                texts=texts_by_pos[pos],
                batch_size=self.config.batch_size,
                device=self.config.device,
                max_length=self.config.max_length,
            )
        
        # Build position-vs-rest probe directions (one per position)
        probes: List[torch.Tensor] = []
        raw_norms_by_pos: Dict[str, float] = {}
        for pos in [0, 1, 2, 3]:
            mean_pos = embeddings_by_pos[pos].mean(dim=0)
            rest = torch.cat([embeddings_by_pos[p] for p in [0, 1, 2, 3] if p != pos], dim=0)
            mean_rest = rest.mean(dim=0)
            v = mean_pos - mean_rest
            raw_norms_by_pos[POSITION_LABELS[pos]] = float(v.norm().item())
            v = v / (v.norm() + 1e-8)
            probes.append(v)
        
        # Optionally clean with correctness (balanced across positions)
        if self.config.extra.get("clean_with_correctness", True):
            logger.info("Building correctness probe for cleaning (balanced across positions)...")
            extra = self.config.extra

            # Choose correctness dataset matching the position dataset type
            if isinstance(self.dataset, PositionBigBenchDataset):
                correctness_dataset = CorrectnessPositionBigBenchDataset(
                    source=self.config.dataset_source,
                    probe_tasks=extra.get("probe_tasks"),
                    eval_tasks=extra.get("eval_tasks"),
                    max_per_task_probe=extra.get("max_per_task_probe", 20),
                    max_per_task_eval=extra.get("max_per_task_eval"),
                    probe_size=self.config.probe_size,
                    split_seed=self.config.split_seed,
                    max_test_examples=self.config.max_test_examples,
                )
            elif isinstance(self.dataset, PositionPlausibleQAMCQDataset):
                correctness_dataset = CorrectnessPositionPlausibleQAMCQDataset(
                    source=self.config.dataset_source,
                    probe_size=self.config.probe_size,
                    split_seed=self.config.split_seed,
                    max_test_examples=self.config.max_test_examples,
                )
            else:
                correctness_dataset = CorrectnessPositionMCQDataset(
                    source=extra.get("dataset_id", "guipenedo/gsm8k-mc"),
                    split=extra.get("train_split", "train"),
                    eval_split=extra.get("eval_split", "test"),
                    probe_size=self.config.probe_size,
                    split_seed=self.config.split_seed,
                    max_test_examples=self.config.max_test_examples,
                    subset=extra.get("subset", None),
                )
            correctness_pairs = correctness_dataset.get_probe_pairs(self.tokenizer)
            
            basis, clean_meta, correctness_probe = clean_probe_basis_with_correctness(
                probes=probes,
                model=self.model,
                tokenizer=self.tokenizer,
                correctness_pairs=correctness_pairs,
                batch_size=self.config.batch_size,
                device=self.config.device,
                max_length=self.config.max_length,
            )
            self.correctness_probe = correctness_probe  # type: ignore[attr-defined]
            self.probe = basis
            logger.info(
                "Cleaned position probe basis: removed correctness direction (mean overlap %.1f%%), kept %d basis vectors",
                clean_meta.get("correctness_overlap_mean", 0.0) * 100,
                int(self.probe.shape[0]),
            )
        else:
            # Orthonormalize raw directions to form a basis
            self.probe = gram_schmidt(probes)
            clean_meta = {"cleaned_with_correctness": False}
        
        # Compute metadata
        hidden_dim = int(self.probe.shape[-1])
        metadata: Dict[str, Any] = {
            "hidden_dim": hidden_dim,
            "probe_type": "pos_vs_rest_basis",
            "n_embeddings_by_pos": {
                "A": len(texts_by_pos[0]),
                "B": len(texts_by_pos[1]),
                "C": len(texts_by_pos[2]),
                "D": len(texts_by_pos[3]),
            },
            "raw_direction_norms_by_pos": raw_norms_by_pos,
            "n_basis_vectors": int(self.probe.shape[0]),
        }
        metadata.update(clean_meta)
        
        # Save probe if configured
        if self.config.save_probe:
            probe_dir = Path(self.config.artifacts_dir) / "probes" / "position" / self.config.name
            probe_dir.mkdir(parents=True, exist_ok=True)
            
            import json
            torch.save(self.probe, probe_dir / "probe.pt")
            with open(probe_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Saved probe to %s", probe_dir)
        
        return metadata
    
    def _compute_metrics(
        self,
        rewards: Dict[str, List[float]],
        eval_examples: List[EvalExample],
    ) -> Dict[str, float]:
        """Compute position bias metrics."""
        correct_positions = [ex.metadata["correct_idx"] for ex in eval_examples]
        return compute_position_metrics(
            rewards=rewards,
            correct_positions=correct_positions,
        )
    
    def _create_plot(
        self,
        results: ExperimentResults,
        output_path: Path,
    ) -> None:
        """Create position bias plot."""
        create_position_bias_plot(
            baseline_metrics=results.baseline_metrics,
            nulled_metrics=results.nulled_metrics or {},
            output_path=output_path,
            title=f"{self.config.name} Position Bias",
            n_examples=results.n_eval_examples,
        )
        accuracy_path = output_path.parent / f"{self.config.name}_accuracy.png"
        create_accuracy_plot(
            baseline_metrics=results.baseline_metrics,
            nulled_metrics=results.nulled_metrics or {},
            output_path=accuracy_path,
            title=f"{self.config.name} Accuracy",
            n_examples=results.n_eval_examples,
        )
    
    def fast_eval_from_cache(
        self,
        cache_path: Path,
        null_mode: bool = False,
    ) -> Dict[str, float]:
        """Fast evaluation using cached embeddings.
        
        Args:
            cache_path: Path to cached embeddings
            null_mode: Whether to apply null projection
            
        Returns:
            Metrics dictionary
        """
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")
        
        cache = torch.load(cache_path, map_location="cpu")
        embeddings = cache["embeddings"]
        metadata = cache["metadata"]
        
        logger.info("Loaded %d cached questions", len(metadata))
        
        # Load score head
        score_head = get_score_head(self.model)
        score_head = score_head.to(self.config.device)
        score_head.eval()
        
        # Prepare null directions if needed (supports [d] or [k,d])
        null_directions = None
        if null_mode and self.probe is not None:
            null_directions = self.probe.to(self.config.device).float()
            if null_directions.dim() == 1:
                null_directions = null_directions.unsqueeze(0)
            null_directions = null_directions / (null_directions.norm(dim=1, keepdim=True) + 1e-8)
        
        # Evaluate
        position_counts = [0, 0, 0, 0]
        correct_count = 0
        correct_at_pos = [0, 0, 0, 0]  # How many times correct is at each position
        correct_when_at_pos = [0, 0, 0, 0]  # Correct predictions when answer at that position
        
        for idx, meta in enumerate(tqdm(metadata, desc="Fast eval")):
            q_embeddings = embeddings[idx].to(self.config.device)
            
            if null_directions is not None:
                # Project out null directions
                dots = q_embeddings @ null_directions.T  # [n_choices, k]
                projection = dots @ null_directions      # [n_choices, d]
                q_embeddings = q_embeddings - projection
            
            with torch.inference_mode():
                scores = score_head(q_embeddings).squeeze(-1)
            
            max_idx = scores.argmax().item()
            correct_pos = meta["correct_idx"]
            position_counts[max_idx] += 1
            correct_at_pos[correct_pos] += 1
            
            if max_idx == correct_pos:
                correct_count += 1
                correct_when_at_pos[correct_pos] += 1
        
        total = len(metadata)
        accuracy = correct_count / total
        position_dist = [c / total * 100 for c in position_counts]
        
        # Compute accuracy per position
        result = {
            "accuracy": accuracy,
            "position_distribution": position_dist,
            "position_A_pct": position_dist[0],
            "position_B_pct": position_dist[1],
            "position_C_pct": position_dist[2],
            "position_D_pct": position_dist[3],
            "max_position_bias": max(abs(p - 25) for p in position_dist),
            "n_examples": total,
        }
        
        # Add accuracy per position
        for pos, label in enumerate(POSITION_LABELS):
            if correct_at_pos[pos] > 0:
                result[f"accuracy_when_{label}"] = correct_when_at_pos[pos] / correct_at_pos[pos]
            else:
                result[f"accuracy_when_{label}"] = 0.0
            result[f"n_correct_at_{label}"] = correct_at_pos[pos]
        
        return result


class BinaryPositionBiasExperiment(BiasExperiment):
    """Experiment for binary position bias (first vs second mention).
    
    Uses PlausibleQA format: "It might be {first}, but it could also be {second}."
    Tests if RM prefers whichever answer is mentioned first.
    
    Set `extra.clean_with_correctness: true` to project out the "correct vs
    incorrect response" direction from the position probe.
    """
    
    correctness_probe: Optional[torch.Tensor] = None
    
    @property
    def bias_type(self) -> str:
        return "position"
    
    def _create_correctness_dataset(self) -> ProbeDataset:
        """Create correctness probe dataset."""
        extra = self.config.extra
        dataset_class = extra.get("dataset_class", "position_plausibleqa")
        
        if dataset_class == "position_bigbench":
            return CorrectnessPositionBigBenchDataset(
                source=self.config.dataset_source,
                probe_tasks=extra.get("probe_tasks"),
                eval_tasks=extra.get("eval_tasks"),
                max_per_task_probe=extra.get("max_per_task_probe", 20),
                max_per_task_eval=extra.get("max_per_task_eval"),
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
        else:
            return CorrectnessPositionPlausibleQADataset(
                source=self.config.dataset_source,
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
    
    def build_probe(self) -> Dict[str, Any]:
        """Build position probe, optionally cleaned with correctness probe."""
        metadata = super().build_probe()
        
        if not self.config.extra.get("clean_with_correctness", True):
            return metadata
        
        logger.info("Building correctness probe for cleaning...")
        correctness_dataset = self._create_correctness_dataset()
        correctness_pairs = correctness_dataset.get_probe_pairs(self.tokenizer)

        cleaned, clean_meta, correctness_probe = clean_probe_with_correctness(
            probe=self.probe,
            model=self.model,
            tokenizer=self.tokenizer,
            correctness_pairs=correctness_pairs,
            batch_size=self.config.batch_size,
            device=self.config.device,
            max_length=self.config.max_length,
        )
        self.probe = cleaned
        self.correctness_probe = correctness_probe

        logger.info(
            "Cleaned position probe: removed %.1f%% overlap with correctness direction",
            clean_meta["correctness_overlap"] * 100,
        )
        metadata.update(clean_meta)
        
        return metadata
    
    def _create_dataset(self) -> PositionPlausibleQADataset:
        """Create binary position bias dataset."""
        return PositionPlausibleQADataset(
            source=self.config.dataset_source,
            probe_size=self.config.probe_size,
            split_seed=self.config.split_seed,
            max_test_examples=self.config.max_test_examples,
        )
    
    def _compute_metrics(
        self,
        rewards: Dict[str, List[float]],
        eval_examples: List[EvalExample],
    ) -> Dict[str, float]:
        """Compute binary position bias metrics."""
        return compute_binary_position_metrics(
            rewards=rewards,
            n_examples=len(eval_examples),
        )
    
    def _create_plot(
        self,
        results: ExperimentResults,
        output_path: Path,
    ) -> None:
        """Create position bias plot (same as freeform, first vs last)."""
        create_freeform_position_plot(
            baseline=results.baseline_metrics,
            nulled=results.nulled_metrics,
            output_path=output_path,
            title=f"Position Bias: {self.config.name}",
            n_examples=results.n_eval_examples,
        )
        accuracy_path = output_path.parent / f"{self.config.name}_accuracy.png"
        create_accuracy_plot(
            baseline_metrics=results.baseline_metrics,
            nulled_metrics=results.nulled_metrics or {},
            output_path=accuracy_path,
            title=f"{self.config.name} Accuracy",
            n_examples=results.n_eval_examples,
        )


class FreeformPositionBiasExperiment(BiasExperiment):
    """Experiment for freeform position bias (first vs last in list).
    
    Prompt lists all 4 choices: "The answer is either {a}, {b}, {c}, or {d}."
    Tests if RM accuracy differs when correct answer is first vs last in list.
    
    Set `extra.clean_with_correctness: true` to project out the "correct vs
    incorrect response" direction from the position probe.
    """
    
    correctness_probe: Optional[torch.Tensor] = None
    
    @property
    def bias_type(self) -> str:
        return "position"
    
    def _create_correctness_dataset(self) -> ProbeDataset:
        """Create correctness probe dataset."""
        extra = self.config.extra
        dataset_class = extra.get("dataset_class", "position_freeform")
        
        if dataset_class == "position_freeform_bigbench":
            return CorrectnessPositionFreeformBigBenchDataset(
                source=self.config.dataset_source,
                probe_tasks=extra.get("probe_tasks"),
                eval_tasks=extra.get("eval_tasks"),
                max_per_task_probe=extra.get("max_per_task_probe", 20),
                max_per_task_eval=extra.get("max_per_task_eval"),
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
        elif dataset_class == "position_freeform_plausibleqa":
            return CorrectnessPositionFreeformPlausibleQADataset(
                source=self.config.dataset_source,
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
        else:
            # Default: GSM8K-MC or MMLU style freeform
            return CorrectnessPositionFreeformDataset(
                source=extra.get("dataset_id", "guipenedo/gsm8k-mc"),
                split=extra.get("train_split", "train"),
                eval_split=extra.get("eval_split", "test"),
                num_choices=int(extra.get("num_choices", 4)),
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
                subset=extra.get("subset", None),
            )
    
    def build_probe(self) -> Dict[str, Any]:
        """Build position probe, optionally cleaned with correctness probe."""
        metadata = super().build_probe()
        
        if not self.config.extra.get("clean_with_correctness", True):
            return metadata
        
        logger.info("Building correctness probe for cleaning...")
        correctness_dataset = self._create_correctness_dataset()
        correctness_pairs = correctness_dataset.get_probe_pairs(self.tokenizer)

        cleaned, clean_meta, correctness_probe = clean_probe_with_correctness(
            probe=self.probe,
            model=self.model,
            tokenizer=self.tokenizer,
            correctness_pairs=correctness_pairs,
            batch_size=self.config.batch_size,
            device=self.config.device,
            max_length=self.config.max_length,
        )
        self.probe = cleaned
        self.correctness_probe = correctness_probe

        logger.info(
            "Cleaned position probe: removed %.1f%% overlap with correctness direction",
            clean_meta["correctness_overlap"] * 100,
        )
        metadata.update(clean_meta)
        
        return metadata
    
    def _create_dataset(self) -> ProbeDataset:
        """Create freeform position bias dataset."""
        extra = self.config.extra
        dataset_class = extra.get("dataset_class", "position_freeform")
        
        if dataset_class == "position_freeform_bigbench":
            return PositionFreeformBigBenchDataset(
                source=self.config.dataset_source,
                probe_tasks=extra.get("probe_tasks"),
                eval_tasks=extra.get("eval_tasks"),
                max_per_task_probe=extra.get("max_per_task_probe", 100),
                max_per_task_eval=extra.get("max_per_task_eval", 500),
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
        elif dataset_class == "position_freeform_plausibleqa":
            return PositionFreeformPlausibleQADataset(
                source=self.config.dataset_source,
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
        else:
            # Default: GSM8K-MC or MMLU style freeform
            return PositionFreeformDataset(
                source=extra.get("dataset_id", "guipenedo/gsm8k-mc"),
                split=extra.get("train_split", "train"),
                eval_split=extra.get("eval_split", "test"),
                num_choices=int(extra.get("num_choices", 4)),
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
                subset=extra.get("subset", None),
            )
    
    def _compute_metrics(
        self,
        rewards: Dict[str, List[float]],
        eval_examples: List[EvalExample],
    ) -> Dict[str, float]:
        """Compute freeform position bias metrics."""
        num_choices = 4
        if eval_examples:
            num_choices = int(eval_examples[0].metadata.get("num_choices", 4))
        return compute_freeform_position_metrics(
            rewards=rewards,
            n_examples=len(eval_examples),
            num_choices=num_choices,
        )
    
    def _create_plot(
        self,
        results: ExperimentResults,
        output_path: Path,
    ) -> None:
        """Create freeform position bias plot."""
        create_freeform_position_plot(
            baseline=results.baseline_metrics,
            nulled=results.nulled_metrics,
            output_path=output_path,
            title=f"Position Bias (First vs Last): {self.config.name}",
            n_examples=results.n_eval_examples,
        )
        accuracy_path = output_path.parent / f"{self.config.name}_accuracy.png"
        create_accuracy_plot(
            baseline_metrics=results.baseline_metrics,
            nulled_metrics=results.nulled_metrics or {},
            output_path=accuracy_path,
            title=f"{self.config.name} Accuracy",
            n_examples=results.n_eval_examples,
        )


def run_position_experiment(config_path: Path) -> ExperimentResults:
    """Run position bias experiment from config file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Experiment results
    """
    config = ExperimentConfig.from_yaml(config_path)
    
    # Choose experiment class based on dataset
    extra = config.extra
    dataset_class = extra.get("dataset_class", "")
    
    if "plausibleqa" in config.dataset_source.lower() and dataset_class != "position_bigbench":
        experiment = BinaryPositionBiasExperiment(config)
    elif dataset_class == "position_freeform":
        experiment = FreeformPositionBiasExperiment(config)
    else:
        # Default MCQ position experiment (GSM8K-MC, MMLU, BigBench)
        experiment = PositionBiasExperiment(config)
    
    return experiment.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run position bias experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_position_experiment(args.config)


