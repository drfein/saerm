from __future__ import annotations

import logging
import math
from typing import Dict

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
        dataset = EmbeddingTensorDataset(tensor)
        dataloader = DataLoader(dataset, batch_size=self._job.batch_size, shuffle=True, drop_last=False)
        self._input_dim = tensor.shape[1]
        model = BatchTopKSAE(self._input_dim, self._job.hidden_size, self._job.k_active).to(self._device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._job.learning_rate)
        warmup_steps = self._compute_warmup_steps()
        log_interval = max(1, self._job.log_interval)

        global_step = 0
        total_loss = 0.0
        total_recon = 0.0
        total_l1 = 0.0
        total_active = 0.0
        total_r2 = 0.0

        activation_counts = torch.zeros(self._job.hidden_size, device=self._device)
        model.train()
        while global_step < self._job.steps:
            for batch in dataloader:
                if global_step >= self._job.steps:
                    break
                batch = batch.to(self._device)
                optimizer.zero_grad()
                self._apply_lr_schedule(optimizer, global_step, warmup_steps)
                recon, codes = model(batch)
                recon_loss = torch.nn.functional.mse_loss(recon, batch)
                sparsity_loss = codes.abs().mean()
                loss = recon_loss + self._job.l1_coef * sparsity_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_l1 += sparsity_loss.item()
                total_active += self._average_activations(codes)
                r2 = self._batch_r2(batch, recon)
                total_r2 += r2
                activation_counts += (codes != 0).float().sum(dim=0)
                global_step += 1

                if global_step % self._job.checkpoint_interval == 0 or global_step == self._job.steps:
                    self._save_checkpoint(model)
                if global_step % log_interval == 0 or global_step == self._job.steps:
                    dead_pct = self._dead_neuron_percent(activation_counts)
                    metrics = {
                        "step": global_step,
                        "loss": loss.item(),
                        "recon_loss": recon_loss.item(),
                        "sparsity_loss": sparsity_loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "dead_neuron_pct": dead_pct,
                        "r2_batch": r2,
                    }
                    logger.info(
                        "SAE %s step %d | loss %.4f | recon %.4f | L1 %.4f | dead %.2f%% | R2 %.4f",
                        self._job.job_id,
                        global_step,
                        metrics["loss"],
                        metrics["recon_loss"],
                        metrics["sparsity_loss"],
                        metrics["dead_neuron_pct"],
                        metrics["r2_batch"],
                    )
                    self._log_wandb(metrics, commit=True)

        avg_loss = total_loss / global_step if global_step else 0.0
        avg_recon = total_recon / global_step if global_step else 0.0
        avg_l1 = total_l1 / global_step if global_step else 0.0
        avg_active = total_active / global_step if global_step else 0.0
        avg_r2 = total_r2 / global_step if global_step else 0.0
        final_dead_pct = self._dead_neuron_percent(activation_counts)
        metrics = {"loss": avg_loss, "recon": avg_recon, "l1": avg_l1, "active": avg_active, "r2": avg_r2, "dead_pct": final_dead_pct}
        logger.info("SAE %s finished: %s", self._job.job_id, metrics)
        metadata_path = self._storage.sae_metadata_path(self._job.job_id)
        self._storage.write_metadata(metadata_path, {
            "job_id": self._job.job_id,
            "embedding_job": self._job.embedding_job,
            "hidden_size": self._job.hidden_size,
            "k_active": self._job.k_active,
            "input_dim": self._input_dim,
            "steps": self._job.steps,
            "learning_rate": self._job.learning_rate,
            "l1_coef": self._job.l1_coef,
            "warmup_steps": warmup_steps,
            "warmup_ratio": self._job.warmup_ratio,
            "min_lr_scale": self._job.min_lr_scale,
            "use_cosine_decay": self._job.use_cosine_decay,
            "dead_neuron_percent": final_dead_pct,
            "metrics": metrics,
        })
        self._log_wandb({
            "step": global_step,
            "final_loss": avg_loss,
            "final_recon": avg_recon,
            "final_r2": avg_r2,
            "dead_neuron_pct": final_dead_pct,
        }, commit=True)
        self._finish_wandb()
        return metrics

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

    def _compute_warmup_steps(self) -> int:
        if self._job.steps <= 1:
            return 0
        if self._job.warmup_steps is not None:
            warmup_steps = max(0, min(self._job.warmup_steps, self._job.steps - 1))
        else:
            warmup_steps = int(self._job.steps * self._job.warmup_ratio)
            warmup_steps = max(0, min(warmup_steps, self._job.steps - 1))
        return warmup_steps

    def _apply_lr_schedule(self, optimizer: torch.optim.Optimizer, step: int, warmup_steps: int) -> None:
        scale = self._lr_scale(step, warmup_steps)
        lr = self._job.learning_rate * scale
        for group in optimizer.param_groups:
            group["lr"] = lr

    def _lr_scale(self, step: int, warmup_steps: int) -> float:
        total_steps = max(1, self._job.steps)
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

    def _batch_r2(self, target: torch.Tensor, reconstruction: torch.Tensor) -> float:
        ss_res = torch.sum((target - reconstruction) ** 2).item()
        mean = target.mean(dim=0, keepdim=True)
        ss_tot = torch.sum((target - mean) ** 2).item()
        if ss_tot == 0.0:
            return 0.0
        return float(1.0 - ss_res / (ss_tot + 1e-8))

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
