from __future__ import annotations

import logging
import math
from typing import Dict

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

import torch
from torch.utils.data import DataLoader

from ..config import SAETrainingConfig
from ..embeddings.cache import EmbeddingCacheManager
from ..storage import StorageManager
from .dataset import EmbeddingTensorDataset
from .models import BatchTopKSAE

logger = logging.getLogger(__name__)


class SAETrainer:
    """Train a sparse autoencoder using cached embeddings."""

    def __init__(
        self,
        storage: StorageManager,
        cache: EmbeddingCacheManager,
        job: SAETrainingConfig,
    ) -> None:
        self._storage = storage
        self._cache = cache
        self._job = job
        self._device = torch.device(job.device if job.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._input_dim: int | None = None
        self._wandb_run = self._setup_wandb()

    def train(self) -> Dict[str, float]:
        payload = self._cache.load_embeddings(self._job.embedding_job)
        tensor = payload["embeddings"].float()
        if tensor.numel() == 0:
            logger.warning("No embeddings found for job %s", self._job.embedding_job)
            return {"loss": 0.0, "recon": 0.0, "l1": 0.0, "active": 0.0, "r2": 0.0, "dead_pct": 0.0}

        dataset = EmbeddingTensorDataset(tensor)
        dataloader = DataLoader(dataset, batch_size=self._job.batch_size, shuffle=True, drop_last=False)
        dataset_size = len(dataset)
        self._input_dim = tensor.shape[1]

        batches_per_epoch = max(1, math.ceil(dataset_size / self._job.batch_size))
        steps_limit = self._job.steps
        if self._job.epochs is not None:
            planned_epochs = self._job.epochs
        elif steps_limit is not None:
            planned_epochs = max(1, math.ceil(steps_limit / batches_per_epoch))
        else:
            raise ValueError(f"SAE job {self._job.job_id} must specify epochs or steps")
        total_schedule_steps = planned_epochs * batches_per_epoch
        if steps_limit is not None:
            total_schedule_steps = min(total_schedule_steps, steps_limit)

        model = BatchTopKSAE(self._input_dim, self._job.hidden_size, self._job.k_active).to(self._device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._job.learning_rate)
        warmup_steps = self._compute_warmup_steps(total_schedule_steps)

        activation_counts = torch.zeros(self._job.hidden_size, device=self._device)
        global_step = 0
        completed_epochs = 0
        last_epoch_metrics: Dict[str, float] | None = None

        model.train()
        for epoch_index in range(planned_epochs):
            if steps_limit is not None and global_step >= steps_limit:
                break

            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_l1 = 0.0
            epoch_active = 0.0
            epoch_corr_stats = self._init_corr_stats()
            num_batches = 0

            progress = tqdm(dataloader, desc=f"Epoch {epoch_index + 1}/{planned_epochs}", leave=False) if tqdm is not None else None
            iterator = progress if progress is not None else dataloader

            for batch in iterator:
                if steps_limit is not None and global_step >= steps_limit:
                    break

                batch = batch.to(self._device)
                optimizer.zero_grad()
                self._apply_lr_schedule(optimizer, global_step, warmup_steps, total_schedule_steps)
                recon, codes = model(batch)
                recon_loss = torch.nn.functional.mse_loss(recon, batch)
                sparsity_loss = codes.abs().mean()
                loss = recon_loss + self._job.l1_coef * sparsity_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_l1 += sparsity_loss.item()
                epoch_active += self._average_activations(codes)
                self._accumulate_corr_stats(epoch_corr_stats, batch, recon)
                activation_counts += (codes != 0).float().sum(dim=0)
                global_step += 1
                num_batches += 1

                if progress is not None:
                    progress.set_postfix({"loss": f"{loss.item():.4f}", "recon": f"{recon_loss.item():.4f}"})

                if self._job.checkpoint_interval and self._job.checkpoint_interval > 0 and global_step % self._job.checkpoint_interval == 0:
                    self._save_checkpoint(model)

            if progress is not None:
                progress.close()

            if num_batches == 0:
                continue

            completed_epochs += 1
            epoch_avg_loss = epoch_loss / num_batches
            epoch_avg_recon = epoch_recon / num_batches
            epoch_avg_l1 = epoch_l1 / num_batches
            epoch_avg_active = epoch_active / num_batches
            epoch_corr = self._corr_from_stats(epoch_corr_stats)
            epoch_r2 = epoch_corr * epoch_corr
            dead_pct = self._dead_neuron_percent(activation_counts)

            epoch_metrics = {
                "loss": epoch_avg_loss,
                "recon": epoch_avg_recon,
                "l1": epoch_avg_l1,
                "active": epoch_avg_active,
                "r2": epoch_r2,
                "corr": epoch_corr,
                "dead_pct": dead_pct,
            }
            last_epoch_metrics = epoch_metrics

            log_metrics = dict(epoch_metrics)
            log_metrics["epoch"] = epoch_index + 1
            log_metrics["global_step"] = global_step
            log_metrics["lr"] = optimizer.param_groups[0]["lr"]
            self._log_wandb(log_metrics, commit=True)
            logger.info(
                "SAE %s epoch %d/%d | loss %.4f | recon %.4f | L1 %.4f | active %.2f | dead %.2f%% | corr %.4f | R2 %.4f",
                self._job.job_id,
                epoch_index + 1,
                planned_epochs,
                epoch_avg_loss,
                epoch_avg_recon,
                epoch_avg_l1,
                epoch_avg_active,
                dead_pct,
                epoch_corr,
                epoch_r2,
            )

        final_metrics = last_epoch_metrics or {"loss": 0.0, "recon": 0.0, "l1": 0.0, "active": 0.0, "r2": 0.0, "dead_pct": self._dead_neuron_percent(activation_counts)}
        logger.info("SAE %s finished after %d epoch(s) and %d step(s): %s", self._job.job_id, completed_epochs, global_step, final_metrics)

        metadata_path = self._storage.sae_metadata_path(self._job.job_id)
        self._storage.write_metadata(metadata_path, {
            "job_id": self._job.job_id,
            "embedding_job": self._job.embedding_job,
            "hidden_size": self._job.hidden_size,
            "k_active": self._job.k_active,
            "input_dim": self._input_dim,
            "epochs": completed_epochs,
            "steps": global_step,
            "learning_rate": self._job.learning_rate,
            "l1_coef": self._job.l1_coef,
            "warmup_steps": warmup_steps,
            "warmup_ratio": self._job.warmup_ratio,
            "min_lr_scale": self._job.min_lr_scale,
            "use_cosine_decay": self._job.use_cosine_decay,
            "dead_neuron_percent": final_metrics.get("dead_pct", 0.0),
            "metrics": final_metrics,
        })
        summary_metrics = dict(final_metrics)
        summary_metrics["step"] = global_step
        summary_metrics["epoch"] = completed_epochs
        self._log_wandb(summary_metrics, commit=True)
        self._finish_wandb()
        return final_metrics

    def _save_checkpoint(self, model: BatchTopKSAE) -> None:
        path = self._storage.sae_checkpoint_path(self._job.job_id)
        payload = {
            "state_dict": model.state_dict(),
            "input_dim": self._input_dim,
            "hidden_dim": self._job.hidden_size,
            "k_active": self._job.k_active,
        }
        torch.save(payload, path)
        logger.debug("Saved SAE checkpoint to %s", path)

    def _compute_warmup_steps(self, total_steps: int) -> int:
        if total_steps <= 1:
            return 0
        if self._job.warmup_steps is not None:
            warmup_steps = max(0, min(self._job.warmup_steps, total_steps - 1))
        else:
            warmup_steps = int(total_steps * self._job.warmup_ratio)
            warmup_steps = max(0, min(warmup_steps, total_steps - 1))
        return warmup_steps

    def _apply_lr_schedule(self, optimizer: torch.optim.Optimizer, step: int, warmup_steps: int, total_steps: int) -> None:
        scale = self._lr_scale(step, warmup_steps, total_steps)
        lr = self._job.learning_rate * scale
        for group in optimizer.param_groups:
            group["lr"] = lr

    def _lr_scale(self, step: int, warmup_steps: int, total_steps: int) -> float:
        total_steps = max(1, total_steps)
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-6, (step + 1) / warmup_steps)
        progress_denominator = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / progress_denominator if progress_denominator > 0 else 1.0
        progress = min(max(progress, 0.0), 1.0)
        if self._job.use_cosine_decay:
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(self._job.min_lr_scale, cosine)
        linear = 1.0 - progress
        return max(self._job.min_lr_scale, linear)

    def _average_activations(self, codes: torch.Tensor) -> float:
        if codes.numel() == 0:
            return 0.0
        active = (codes != 0).float().sum(dim=1).mean()
        return float(active.item())

    def _dead_neuron_percent(self, activation_counts: torch.Tensor) -> float:
        if activation_counts.numel() == 0:
            return 0.0
        dead = (activation_counts == 0).sum().item()
        return float(dead / activation_counts.numel() * 100.0)

    def _init_corr_stats(self) -> Dict[str, float]:
        return {
            "count": 0.0,
            "x_sum": 0.0,
            "y_sum": 0.0,
            "x_sq_sum": 0.0,
            "y_sq_sum": 0.0,
            "xy_sum": 0.0,
        }

    def _accumulate_corr_stats(self, stats: Dict[str, float], target: torch.Tensor, reconstruction: torch.Tensor) -> None:
        with torch.no_grad():
            x = target.detach()
            y = reconstruction.detach()
            stats["count"] += float(x.numel())
            stats["x_sum"] += float(torch.sum(x, dtype=torch.float64).item())
            stats["y_sum"] += float(torch.sum(y, dtype=torch.float64).item())
            stats["x_sq_sum"] += float(torch.sum(x * x, dtype=torch.float64).item())
            stats["y_sq_sum"] += float(torch.sum(y * y, dtype=torch.float64).item())
            stats["xy_sum"] += float(torch.sum(x * y, dtype=torch.float64).item())

    def _corr_from_stats(self, stats: Dict[str, float]) -> float:
        n = stats["count"]
        if n <= 1.0:
            return 0.0
        numerator = stats["xy_sum"] - (stats["x_sum"] * stats["y_sum"]) / n
        x_var = stats["x_sq_sum"] - (stats["x_sum"] * stats["x_sum"]) / n
        y_var = stats["y_sq_sum"] - (stats["y_sum"] * stats["y_sum"]) / n
        if x_var <= 0.0 or y_var <= 0.0:
            return 0.0
        denominator = math.sqrt(x_var * y_var)
        if denominator <= 0.0:
            return 0.0
        correlation = numerator / denominator
        correlation = max(min(correlation, 1.0), -1.0)
        return float(correlation)

    def _setup_wandb(self):
        if self._job.wandb_project is None:
            return None
        try:  # pragma: no cover - optional dependency
            import wandb  # type: ignore
        except Exception as exc:  # pragma: no cover
            logger.warning("wandb import failed: %s", exc)
            return None

        run = wandb.init(
            project=self._job.wandb_project,
            entity=self._job.wandb_entity,
            name=self._job.wandb_name or self._job.job_id,
            config={
                "job_id": self._job.job_id,
                "hidden_size": self._job.hidden_size,
                "k_active": self._job.k_active,
                "learning_rate": self._job.learning_rate,
                "steps": self._job.steps,
                "epochs": self._job.epochs,
                "batch_size": self._job.batch_size,
                "l1_coef": self._job.l1_coef,
            },
            reinit=True,
        )
        return run

    def _log_wandb(self, metrics: Dict[str, float], commit: bool = False) -> None:
        if self._wandb_run is None:
            return
        try:  # pragma: no cover
            import wandb  # type: ignore

            wandb.log(metrics, commit=commit)
        except Exception as exc:  # pragma: no cover
            logger.warning("wandb.log failed: %s", exc)

    def _finish_wandb(self) -> None:
        if self._wandb_run is None:
            return
        try:  # pragma: no cover
            self._wandb_run.finish()
        except Exception as exc:  # pragma: no cover
            logger.warning("wandb finish failed: %s", exc)
        finally:
            self._wandb_run = None
