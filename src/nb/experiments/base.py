"""
Base classes for bias evaluation experiments.

Each experiment follows the same pattern:
1. Load config and dataset
2. Build probe direction from probe split
3. Evaluate baseline (no projection)
4. Evaluate with null-space projection
5. Plot results with error bars
6. Save results
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.nb.datasets.base import ProbeDataset
from src.nb.nullbias.probe import build_probe_direction, get_rewards_with_nulling, get_rewards_both

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a bias evaluation experiment."""
    
    # Experiment identification
    name: str
    """Experiment name (e.g., 'length_bias_skywork')"""
    
    bias_type: str
    """Type of bias: 'length', 'sycophancy', 'uncertainty', 'position'"""
    
    # Model settings
    model_path: str
    """Path or HuggingFace ID of the reward model"""
    
    trust_remote_code: bool = True
    """Whether to trust remote code when loading model"""
    
    # Dataset settings
    dataset_source: str = ""
    """Path to dataset file or HuggingFace dataset ID"""
    
    dataset_name: str = ""
    """Dataset name for organizing output (e.g., 'gsm8k_mc', 'mmlu', 'plausibleqa')"""
    
    dataset_class: str = ""
    """Dataset class to use (e.g., 'sycophancy_mcq', 'uncertainty_mcq')"""
    
    probe_size: int = 500
    """Number of examples for probe training"""
    
    max_test_examples: Optional[int] = None
    """Maximum number of test examples (None = use all)"""
    
    split_seed: int = 42
    """Seed for deterministic train/test split"""
    
    # Evaluation settings
    null_alpha: float = 1.0
    """Nullification strength (0=none, 1=full)"""
    
    batch_size: int = 8
    """Batch size for inference"""
    
    max_length: int = 2048
    """Maximum sequence length"""
    
    device: str = "cuda"
    """Device for inference"""
    
    # Output settings
    raw_data_dir: str = "artifacts/raw_data"
    """Directory for raw per-example data (rewards, predictions)"""
    
    artifacts_dir: str = "artifacts"
    """Root directory for non-plot artifacts (results JSONs, probes)"""
    
    plots_dir: str = "plots"
    """Root directory for plots"""
    
    save_probe: bool = True
    """Whether to save the probe direction"""
    
    # Additional dataset-specific settings
    extra: Dict[str, Any] = field(default_factory=dict)
    """Additional settings for specific dataset types"""
    
    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        # Drop legacy output_dir if present
        data.pop("output_dir", None)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        data = dict(data)
        data.pop("output_dir", None)
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "bias_type": self.bias_type,
            "model_path": self.model_path,
            "trust_remote_code": self.trust_remote_code,
            "dataset_source": self.dataset_source,
            "dataset_class": self.dataset_class,
            "probe_size": self.probe_size,
            "max_test_examples": self.max_test_examples,
            "split_seed": self.split_seed,
            "null_alpha": self.null_alpha,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "device": self.device,
            "raw_data_dir": self.raw_data_dir,
            "artifacts_dir": self.artifacts_dir,
            "plots_dir": self.plots_dir,
            "save_probe": self.save_probe,
            "extra": self.extra,
        }
    
    def save_yaml(self, path: Path) -> None:
        """Save config to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


@dataclass
class ExperimentResults:
    """Results from a bias evaluation experiment."""
    
    config: ExperimentConfig
    """Experiment configuration"""
    
    baseline_metrics: Dict[str, float]
    """Metrics without nullification"""
    
    nulled_metrics: Optional[Dict[str, float]] = None
    """Metrics with nullification (if probe was built)"""
    
    probe_metadata: Optional[Dict[str, Any]] = None
    """Metadata about the probe (separation, accuracy, etc.)"""
    
    n_probe_examples: int = 0
    """Number of examples used for probe training"""
    
    n_eval_examples: int = 0
    """Number of examples used for evaluation"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": self.config.to_dict(),
            "baseline": self.baseline_metrics,
            "nulled": self.nulled_metrics,
            "probe_metadata": self.probe_metadata,
            "n_probe_examples": self.n_probe_examples,
            "n_eval_examples": self.n_eval_examples,
        }
    
    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "ExperimentResults":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            config=ExperimentConfig.from_dict(data["config"]),
            baseline_metrics=data["baseline"],
            nulled_metrics=data.get("nulled"),
            probe_metadata=data.get("probe_metadata"),
            n_probe_examples=data.get("n_probe_examples", 0),
            n_eval_examples=data.get("n_eval_examples", 0),
        )


class BiasExperiment(ABC):
    """Base class for bias evaluation experiments.
    
    Subclasses implement:
    - _create_dataset: Create the appropriate ProbeDataset instance
    - _compute_metrics: Compute bias-specific metrics from rewards
    - _create_plot: Generate the experiment-specific plot
    
    Supports cross-dataset generalization: train probe on one dataset,
    evaluate on another by setting config.extra["probe_source"].
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.dataset: Optional[ProbeDataset] = None
        self.probe_dataset: Optional[ProbeDataset] = None  # For cross-dataset
        self.probe: Optional[torch.Tensor] = None
        self.results: Optional[ExperimentResults] = None
    
    @property
    @abstractmethod
    def bias_type(self) -> str:
        """Return the bias type name."""
        pass
    
    @abstractmethod
    def _create_dataset(self) -> ProbeDataset:
        """Create the dataset instance. Override in subclass."""
        pass
    
    @abstractmethod
    def _compute_metrics(
        self,
        rewards: Dict[str, List[float]],
        eval_examples: List[Any],
    ) -> Dict[str, float]:
        """Compute bias metrics from rewards. Override in subclass.
        
        Args:
            rewards: Dictionary mapping variant names to reward lists
            eval_examples: List of EvalExample objects
            
        Returns:
            Dictionary of computed metrics
        """
        pass
    
    @abstractmethod
    def _create_plot(
        self,
        results: ExperimentResults,
        output_path: Path,
    ) -> None:
        """Create experiment-specific plot. Override in subclass.
        
        Args:
            results: Experiment results
            output_path: Path to save plot
        """
        pass
    
    def load_model(self) -> None:
        """Load the reward model and tokenizer."""
        logger.info("Loading model from %s", self.config.model_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch.bfloat16,
        )
        self.model = self.model.to(self.config.device)
        
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
    
    def load_dataset(self) -> None:
        """Load and split the dataset(s).
        
        If config.extra["probe_source"] is set, loads a separate probe dataset
        for cross-dataset generalization experiments.
        """
        logger.info("Loading evaluation dataset...")
        self.dataset = self._create_dataset()
        
        # Check for cross-dataset setup
        probe_source = self.config.extra.get("probe_source")
        if probe_source:
            logger.info("Loading probe dataset (cross-dataset): %s", probe_source)
            self.probe_dataset = self._create_probe_dataset(probe_source)
        else:
            self.probe_dataset = self.dataset
    
    def _create_probe_dataset(self, source: str) -> ProbeDataset:
        """Create probe dataset for cross-dataset experiments.
        
        Override in subclass if probe dataset needs special handling.
        Default implementation swaps the source in config.
        """
        # Store original and swap
        original_source = self.config.dataset_source
        original_extra = self.config.extra.copy()
        
        self.config.dataset_source = source
        probe_extra = self.config.extra.get("probe_extra", {})
        for k, v in probe_extra.items():
            self.config.extra[k] = v
        
        probe_dataset = self._create_dataset()
        
        # Restore
        self.config.dataset_source = original_source
        self.config.extra = original_extra
        
        return probe_dataset
    
    def build_probe(self) -> Dict[str, Any]:
        """Build probe direction from contrastive pairs.
        
        Uses probe_dataset if set (cross-dataset), otherwise uses main dataset.
        
        Returns:
            Probe metadata dictionary
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.probe_dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")
        
        # Use probe_dataset for building probe
        dataset_for_probe = self.probe_dataset
        logger.info("Building probe from %d examples (dataset: %s)...", 
                   self.config.probe_size, dataset_for_probe.name)
        
        contrastive_pairs = dataset_for_probe.get_probe_pairs(self.tokenizer)
        
        self.probe, metadata = build_probe_direction(
            model=self.model,
            tokenizer=self.tokenizer,
            contrastive_pairs=contrastive_pairs,
            batch_size=self.config.batch_size,
            device=self.config.device,
            max_length=self.config.max_length,
        )
        
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
    
    def _get_all_texts_and_variants(
        self,
        eval_examples: List[Any],
    ) -> tuple[List[str], List[tuple[int, str]]]:
        """Extract all texts and their metadata from eval examples.
        
        Returns:
            Tuple of (all_texts, text_metadata) where metadata is (example_idx, variant_name)
        """
        all_texts = []
        text_meta = []
        
        for idx, example in enumerate(eval_examples):
            for variant_name, text in example.texts.items():
                all_texts.append(text)
                text_meta.append((idx, variant_name))
        
        return all_texts, text_meta
    
    def _organize_rewards(
        self,
        rewards: torch.Tensor,
        text_meta: List[tuple[int, str]],
        n_examples: int,
    ) -> Dict[str, List[float]]:
        """Organize flat rewards list into variant-keyed dictionary.
        
        Args:
            rewards: Flat tensor of rewards
            text_meta: List of (example_idx, variant_name) tuples
            n_examples: Number of evaluation examples
            
        Returns:
            Dictionary mapping variant names to reward lists
        """
        # First pass: collect all variant names
        variant_names = set(variant for _, variant in text_meta)
        
        # Initialize reward lists
        organized = {name: [None] * n_examples for name in variant_names}
        
        # Fill in rewards
        for (idx, variant), reward in zip(text_meta, rewards.tolist()):
            organized[variant][idx] = reward
        
        return organized
    
    def evaluate(self) -> ExperimentResults:
        """Run full evaluation pipeline.
        
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
        baseline_metrics = self._compute_metrics(baseline_organized, eval_examples)
        
        nulled_organized = self._organize_rewards(nulled_rewards, text_meta, n_eval)
        nulled_metrics = self._compute_metrics(nulled_organized, eval_examples)
        
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
        self._save_raw_data(
            eval_examples=eval_examples,
            baseline_rewards=baseline_organized,
            nulled_rewards=nulled_organized,
        )
        
        return self.results
    
    def _save_raw_data(
        self,
        eval_examples: List[Any],
        baseline_rewards: Dict[str, List[float]],
        nulled_rewards: Dict[str, List[float]],
    ) -> None:
        """Save raw per-example data for later analysis.
        
        Saves:
        - Per-example rewards for each variant (baseline and nulled)
        - Example metadata (question, correct_idx, etc.)
        """
        raw_data_dir = Path(self.config.raw_data_dir)
        out_dir = raw_data_dir / self.config.bias_type / self.config.name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Build per-example data
        examples_data = []
        for i, ex in enumerate(eval_examples):
            example_dict = {
                "idx": i,
                "metadata": ex.metadata,
                "baseline_rewards": {k: v[i] for k, v in baseline_rewards.items()},
                "nulled_rewards": {k: v[i] for k, v in nulled_rewards.items()},
            }
            examples_data.append(example_dict)
        
        # Save as JSON
        raw_data_path = out_dir / "raw_data.json"
        with open(raw_data_path, "w") as f:
            json.dump({
                "config": self.config.to_dict(),
                "n_examples": len(examples_data),
                "variant_names": list(baseline_rewards.keys()),
                "examples": examples_data,
            }, f, indent=2)
        
        logger.info("Saved raw data to %s", raw_data_path)
    
    def save_results(self, output_dir: Optional[Path] = None) -> Path:
        """Save experiment results.
        
        Args:
            output_dir: Override output directory
            
        Returns:
            Path to saved results
        """
        if self.results is None:
            raise RuntimeError("No results to save. Call evaluate() first.")
        
        artifacts_root = Path(output_dir or self.config.artifacts_dir)
        out_dir = artifacts_root / "results" / self.config.bias_type
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / f"{self.config.name}_results.json"
        
        self.results.save(results_path)
        logger.info("Saved results to %s", results_path)
        
        return results_path
    
    def create_plot(self, output_dir: Optional[Path] = None) -> Path:
        """Create and save experiment plot.
        
        Args:
            output_dir: Override output directory
            
        Returns:
            Path to saved plot
        """
        if self.results is None:
            raise RuntimeError("No results to plot. Call evaluate() first.")
        
        plots_root = Path(output_dir or self.config.plots_dir)
        # Organize by bias_type/dataset_name/
        if self.config.dataset_name:
            out_dir = plots_root / self.config.bias_type / self.config.dataset_name
        else:
            out_dir = plots_root / self.config.bias_type
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / f"{self.config.name}_plot.png"
        
        self._create_plot(self.results, plot_path)
        logger.info("Saved plot to %s", plot_path)
        
        return plot_path
    
    def run(self) -> ExperimentResults:
        """Run complete experiment: evaluate, save, and plot.
        
        Returns:
            Experiment results
        """
        results = self.evaluate()
        self.save_results()
        self.create_plot()
        
        # Print summary
        self._print_summary()
        
        return results
    
    def _print_summary(self) -> None:
        """Print experiment summary to logger."""
        if self.results is None:
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT SUMMARY: %s", self.config.name)
        logger.info("=" * 60)
        logger.info("Probe examples: %d", self.results.n_probe_examples)
        logger.info("Eval examples: %d", self.results.n_eval_examples)
        
        if self.results.probe_metadata:
            logger.info("Probe accuracy: %.2f%%", 100 * self.results.probe_metadata.get("probe_accuracy", 0))
            logger.info("Probe separation: %.4f", self.results.probe_metadata.get("separation", 0))
        
        logger.info("\nBaseline metrics:")
        for key, val in self.results.baseline_metrics.items():
            if isinstance(val, float):
                logger.info("  %s: %.4f", key, val)
        
        if self.results.nulled_metrics:
            logger.info("\nNulled metrics (alpha=%.2f):", self.config.null_alpha)
            for key, val in self.results.nulled_metrics.items():
                if isinstance(val, float):
                    baseline_val = self.results.baseline_metrics.get(key, 0)
                    delta = val - baseline_val
                    logger.info("  %s: %.4f (%+.4f)", key, val, delta)
        
        logger.info("=" * 60)




