"""
Sycophancy bias experiment.

Tests whether reward models cave to user opinions regardless of correctness.

Supports multiple dataset types:
- sycophancy: Preference pairs with user opinions (original)
- sycophancy_mcq: MCQ datasets (GSM8K-MC, MMLU) with injected opinions
- sycophancy_plausibleqa: PlausibleQA with injected opinions

Supports probe cleaning via `extra.clean_with_correctness: true` to remove
the "correct vs incorrect response" direction from the sycophancy probe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from src.nb.datasets.base import EvalExample, ProbeDataset
from src.nb.datasets.sycophancy import (
    CorrectnessMCQDataset,
    CorrectnessPlausibleQADataset,
    SycophancyBiasDataset,
    SycophancyMCQDataset,
    SycophancyPlausibleQADataset,
    compute_sycophancy_metrics,
)
from src.nb.datasets.bigbench import (
    SycophancyBigBenchDataset,
    CorrectnessBigBenchDataset,
)
from src.nb.experiments.base import BiasExperiment, ExperimentConfig, ExperimentResults
from src.nb.experiments.plotting import create_sycophancy_plot
from src.nb.nullbias.probe import clean_probe_with_correctness, get_rewards_both

logger = logging.getLogger(__name__)

# Map dataset class names to classes
SYCOPHANCY_DATASET_CLASSES = {
    "sycophancy": SycophancyBiasDataset,
    "sycophancy_mcq": SycophancyMCQDataset,
    "sycophancy_plausibleqa": SycophancyPlausibleQADataset,
    "sycophancy_bigbench": SycophancyBigBenchDataset,
}


class SycophancyBiasExperiment(BiasExperiment):
    """Experiment for evaluating sycophancy bias.
    
    Tests whether RM accuracy differs based on user opinion:
    - High accuracy when user suggests correct answer (agrees)
    - Lower accuracy when user suggests incorrect answer (caves to user)
    
    The accuracy gap measures sycophancy: larger gap = more sycophantic.
    
    Set `extra.clean_with_correctness: true` to project out the "correct vs
    incorrect response" direction from the sycophancy probe.
    """
    
    correctness_probe: Optional[torch.Tensor] = None
    
    @property
    def bias_type(self) -> str:
        return "sycophancy"
    
    def _create_correctness_dataset(self) -> ProbeDataset:
        """Create correctness probe dataset (same data, different pairing)."""
        extra = self.config.extra
        
        if isinstance(self.probe_dataset, SycophancyMCQDataset):
            return CorrectnessMCQDataset(
                source=self.config.dataset_source,
                dataset_id=extra.get("dataset_id", self.config.dataset_source),
                train_split=extra.get("train_split", "train"),
                eval_split=extra.get("eval_split", "test"),
                subset=extra.get("subset"),
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
        elif isinstance(self.probe_dataset, SycophancyPlausibleQADataset):
            return CorrectnessPlausibleQADataset(
                source=self.config.dataset_source,
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
        elif isinstance(self.probe_dataset, SycophancyBigBenchDataset):
            return CorrectnessBigBenchDataset(
                source=self.config.dataset_source,
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
        else:
            raise ValueError(f"No correctness dataset for {type(self.probe_dataset)}")
    
    def build_probe(self) -> Dict[str, Any]:
        """Build sycophancy probe using only baseline-correct examples.
        
        Strategy:
        1. Compute baseline correctness mask (which probe examples RM gets right)
        2. Build probe only on those examples, using wrong user suggestions only
        3. Optionally clean with correctness probe
        
        This eliminates correctness confounding by ensuring probe only captures
        "resisting bad advice" signal on questions the model already knows.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.probe_dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")
        
        dataset_for_probe = self.probe_dataset
        
        # Step 1: Compute baseline correctness mask (only for supported datasets)
        baseline_correct_mask = None
        if isinstance(dataset_for_probe, (SycophancyMCQDataset, SycophancyPlausibleQADataset, SycophancyBigBenchDataset)):
            logger.info("Computing baseline correctness for probe examples...")
            baseline_correct_mask = dataset_for_probe.compute_baseline_correctness_mask(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.config.device,
                batch_size=self.config.batch_size,
            )
        
        # Step 2: Build probe using filtered examples
        logger.info("Building probe from %d examples (dataset: %s)...", 
                   self.config.probe_size, dataset_for_probe.name)
        
        contrastive_pairs = dataset_for_probe.get_probe_pairs(
            self.tokenizer,
            baseline_correct_mask=baseline_correct_mask
        )
        
        from src.nb.nullbias.probe import build_probe_direction
        import torch
        import json
        from pathlib import Path
        
        self.probe, metadata = build_probe_direction(
            model=self.model,
            tokenizer=self.tokenizer,
            contrastive_pairs=contrastive_pairs,
            batch_size=self.config.batch_size,
            device=self.config.device,
            max_length=self.config.max_length,
        )
        
        # Add baseline filtering info to metadata
        if baseline_correct_mask is not None:
            n_correct = sum(baseline_correct_mask)
            metadata["baseline_correct_examples"] = n_correct
            metadata["baseline_total_examples"] = len(baseline_correct_mask)
            metadata["baseline_accuracy"] = n_correct / len(baseline_correct_mask)
        
        # Step 3: Optionally clean with correctness probe
        if self.config.extra.get("clean_with_correctness", False):
            if not isinstance(self.probe_dataset, (SycophancyMCQDataset, SycophancyPlausibleQADataset, SycophancyBigBenchDataset)):
                logger.warning("clean_with_correctness only works with sycophancy_mcq, sycophancy_plausibleqa, or sycophancy_bigbench datasets")
            else:
                logger.info("Building correctness probe for cleaning...")
                correctness_dataset = self._create_correctness_dataset()
                correctness_pairs = correctness_dataset.get_probe_pairs(self.tokenizer)

                from src.nb.nullbias.probe import clean_probe_with_correctness
                
                # Clean sycophancy probe by projecting out correctness direction
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
                    "Cleaned sycophancy probe: removed %.1f%% overlap with correctness direction",
                    clean_meta["correctness_overlap"] * 100,
                )
                metadata.update(clean_meta)
        
        # Save probe if configured
        if self.config.save_probe:
            artifacts_root = Path(self.config.artifacts_dir)
            probe_dir = artifacts_root / "probes" / self.config.bias_type / self.config.name
            probe_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(self.probe, probe_dir / "probe.pt")
            with open(probe_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Saved probe to %s", probe_dir)
        
        return metadata
    
    def _create_dataset(self) -> ProbeDataset:
        """Create sycophancy bias dataset based on config."""
        extra = self.config.extra
        dataset_class_name = extra.get("dataset_class", "sycophancy")
        
        # Also check top-level config for dataset_class
        if hasattr(self.config, "dataset_class") and self.config.dataset_class:
            dataset_class_name = self.config.dataset_class
        
        # Map format to dataset class if not explicitly set
        format_type = extra.get("format")
        if format_type == "mcq":
            dataset_class_name = "sycophancy_mcq"
        elif format_type == "plausibleqa":
            dataset_class_name = "sycophancy_plausibleqa"
        
        dataset_cls = SYCOPHANCY_DATASET_CLASSES.get(dataset_class_name, SycophancyBiasDataset)
        
        logger.info("Using dataset class: %s", dataset_cls.__name__)
        
        if dataset_cls == SycophancyMCQDataset:
            return SycophancyMCQDataset(
                source=self.config.dataset_source,
                dataset_id=extra.get("dataset_id", self.config.dataset_source),
                train_split=extra.get("train_split", "train"),
                eval_split=extra.get("eval_split", "test"),
                subset=extra.get("subset"),
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
        elif dataset_cls == SycophancyPlausibleQADataset:
            return SycophancyPlausibleQADataset(
                source=self.config.dataset_source,
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
        elif dataset_cls == SycophancyBigBenchDataset:
            return SycophancyBigBenchDataset(
                source=self.config.dataset_source,
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
        else:
            return SycophancyBiasDataset(
                source=self.config.dataset_source,
                probe_size=self.config.probe_size,
                split_seed=self.config.split_seed,
                max_test_examples=self.config.max_test_examples,
            )
    
    def _compute_metrics(
        self,
        rewards: Dict[str, List[float]],
        eval_examples: List[EvalExample],
        easy_mask: Optional[List[bool]] = None,
    ) -> Dict[str, float]:
        """Compute sycophancy metrics.
        
        Args:
            rewards: Organized rewards dict
            eval_examples: List of evaluation examples
            easy_mask: Optional pre-computed mask from baseline model defining
                       which questions are "easy". If None, computed from this
                       model's no_opinion performance.
        """
        correct_indices = [ex.metadata["correct_idx"] for ex in eval_examples]
        # Pass per-example num_choices for variable-choice datasets like BigBench
        num_choices_list = [int(ex.metadata.get("num_choices", 4)) for ex in eval_examples]
        
        return compute_sycophancy_metrics(
            rewards=rewards,
            correct_indices=correct_indices,
            num_choices=num_choices_list,
            easy_mask=easy_mask,
        )
    
    def evaluate(self) -> ExperimentResults:
        """Run full evaluation pipeline with consistent easy/hard split.
        
        Overrides base class to ensure the easy/hard question split is defined
        by baseline model performance and used consistently for both baseline
        and nulled metrics.
        
        Returns:
            ExperimentResults with baseline and nulled metrics
        """
        if self.model is None:
            self.load_model()
        if self.dataset is None:
            self.load_dataset()
        
        # Build probe
        probe_metadata = self.build_probe()
        
        # Get evaluation examples
        eval_examples = self.dataset.get_eval_examples(self.tokenizer)
        n_eval = len(eval_examples)
        logger.info("Evaluating on %d examples", n_eval)
        
        # Get all texts
        all_texts, text_meta = self._get_all_texts_and_variants(eval_examples)
        
        # Evaluate BOTH baseline and nulled in single forward pass (2x faster)
        logger.info("Running evaluation (baseline + nulled in single pass, alpha=%.2f)...", 
                   self.config.null_alpha)
        baseline_rewards, nulled_rewards = get_rewards_both(
            model=self.model,
            tokenizer=self.tokenizer,
            texts=all_texts,
            probe=self.probe,
            batch_size=self.config.batch_size,
            device=self.config.device,
            max_length=self.config.max_length,
        )
        baseline_organized = self._organize_rewards(baseline_rewards, text_meta, n_eval)
        nulled_organized = self._organize_rewards(nulled_rewards, text_meta, n_eval)
        
        # Compute baseline metrics first - this determines the easy/hard split
        baseline_metrics = self._compute_metrics(baseline_organized, eval_examples)
        
        # Extract the easy_mask from baseline and use it for nulled
        easy_mask = baseline_metrics.pop("easy_mask", None)
        
        # Compute nulled metrics using baseline's easy/hard split
        nulled_metrics = self._compute_metrics(nulled_organized, eval_examples, easy_mask=easy_mask)
        # Remove easy_mask from nulled metrics (not needed in output)
        nulled_metrics.pop("easy_mask", None)
        
        # Create results
        self.results = ExperimentResults(
            config=self.config,
            baseline_metrics=baseline_metrics,
            nulled_metrics=nulled_metrics,
            probe_metadata=probe_metadata,
            n_probe_examples=self.probe_dataset.probe_size_actual if self.probe_dataset else 0,
            n_eval_examples=n_eval,
        )
        
        # Add cross-dataset info if applicable
        if self.probe_dataset is not self.dataset:
            self.results.probe_metadata["cross_dataset"] = True
            self.results.probe_metadata["probe_dataset"] = self.probe_dataset.name
            self.results.probe_metadata["eval_dataset"] = self.dataset.name
        
        # Save raw per-example data
        self._save_raw_data(eval_examples, baseline_organized, nulled_organized)
        
        return self.results
    
    def _create_plot(
        self,
        results: ExperimentResults,
        output_path: Path,
    ) -> None:
        """Create sycophancy plot."""
        create_sycophancy_plot(
            baseline=results.baseline_metrics,
            nulled=results.nulled_metrics,
            output_path=output_path,
            title=f"Sycophancy Evaluation: {self.config.name}",
            n_examples=results.n_eval_examples,
        )


def run_sycophancy_experiment(config_path: Path) -> ExperimentResults:
    """Run sycophancy experiment from config file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Experiment results
    """
    config = ExperimentConfig.from_yaml(config_path)
    experiment = SycophancyBiasExperiment(config)
    return experiment.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run sycophancy experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_sycophancy_experiment(args.config)




