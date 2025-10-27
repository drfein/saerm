from __future__ import annotations

import heapq
import json
import logging
import math
from typing import Dict, List, Optional, Sequence

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
from .models import SparseAutoencoder

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
            return {"loss": 0.0, "recon": 0.0, "active": 0.0, "r2": 0.0, "corr": 0.0, "dead_pct": 0.0}
        records = payload.get("records") or []
        total_examples = tensor.shape[0]

        combined_example_ids: Optional[List[int]] = None
        combined_texts: Optional[List[Optional[str]]] = None
        combined_sources: List[str] = []
        if records:
            if len(records) != total_examples:
                logger.warning(
                    "Record count (%s) does not match embedding rows (%s) for job %s; ignoring record metadata",
                    len(records),
                    total_examples,
                    self._job.embedding_job,
                )
                records = []
            else:
                combined_example_ids = []
                combined_texts = []
                for entry in records:
                    combined_sources.append(str(entry.get("choice", "chosen")))
                    combined_example_ids.append(int(entry.get("example_index", 0)))
                    text_value = entry.get("text")
                    combined_texts.append(str(text_value) if text_value is not None else None)
        if not combined_sources:
            combined_sources = ["chosen"] * total_examples
        if combined_texts is not None and all(text is None for text in combined_texts):
            combined_texts = None

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

        model = SparseAutoencoder(
            input_dim=self._input_dim,
            m_total_neurons=self._job.hidden_size,
            k_active_neurons=self._job.k_active,
            aux_k=self._job.aux_k,
            dead_neuron_threshold_steps=self._job.dead_neuron_threshold_steps,
            prefix_lengths=self._job.prefix_lengths,
            batch_topk_threshold_lr=self._job.batch_topk_threshold_lr,
            activation=self._job.activation,
            device=str(self._device),
        )

        # Default: initialize with K-Means centroids (clusters == hidden_size)
        if tensor.numel() > 0:
            logger.info(
                "Initializing SAE with k-means (k=%d) over %d examples...",
                self._job.hidden_size,
                tensor.shape[0],
            )
            # Use a reasonably large batch for distance computations
            kmeans_batch = max(self._job.batch_size, 8192)
            model.initialize_weights_kmeans_(
                tensor,
                num_iters=50,
                tol=1e-4,
                batch_size=kmeans_batch,
                seed=0,
                kmeans_plus_plus=True,
            )

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
            epoch_active = 0.0
            epoch_corr_stats = self._init_corr_stats()
            num_batches = 0

            progress = tqdm(dataloader, desc=f"Epoch {epoch_index + 1}/{planned_epochs}", leave=False) if tqdm is not None else None
            iterator = progress if progress is not None else dataloader

            for batch in iterator:
                if steps_limit is not None and global_step >= steps_limit:
                    break

                batch = batch.to(self._device)
                optimizer.zero_grad(set_to_none=True)
                self._apply_lr_schedule(optimizer, global_step, warmup_steps, total_schedule_steps)
                recon, info = model(batch)
                loss = model.compute_loss(
                    batch,
                    recon,
                    info,
                    aux_coef=self._job.aux_loss_coef,
                )
                loss.backward()
                model.adjust_decoder_gradient_()
                if self._job.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self._job.grad_clip_norm)
                optimizer.step()
                if self._job.normalize_decoder:
                    model.normalize_decoder_()

                epoch_loss += loss.item()
                recon_loss = torch.nn.functional.mse_loss(recon, batch).item()
                epoch_recon += recon_loss
                codes = info["activations"].detach()
                epoch_active += self._average_activations(codes)
                self._accumulate_corr_stats(epoch_corr_stats, batch, recon)
                activation_counts += (codes != 0).float().sum(dim=0)
                global_step += 1
                num_batches += 1

                if progress is not None:
                    postfix = {"loss": f"{loss.item():.4f}", "recon": f"{recon_loss:.4f}", "threshold": f"{model.threshold.item():.2e}"}
                    progress.set_postfix(postfix)

                if self._job.checkpoint_interval and self._job.checkpoint_interval > 0 and global_step % self._job.checkpoint_interval == 0:
                    self._save_checkpoint(model)

            if progress is not None:
                progress.close()

            if num_batches == 0:
                continue

            completed_epochs += 1
            epoch_avg_loss = epoch_loss / num_batches
            epoch_avg_recon = epoch_recon / num_batches
            epoch_avg_active = epoch_active / num_batches
            epoch_corr = self._corr_from_stats(epoch_corr_stats)
            count = epoch_corr_stats["count"]
            ss_tot = epoch_corr_stats["x_sq_sum"] - (epoch_corr_stats["x_sum"] * epoch_corr_stats["x_sum"]) / max(count, 1.0)
            if ss_tot <= 0.0:
                epoch_r2 = 0.0
            else:
                ss_res = epoch_corr_stats["ss_res"]
                epoch_r2 = 1.0 - ss_res / ss_tot
            dead_pct = self._dead_neuron_percent(activation_counts)

            epoch_metrics = {
                "loss": epoch_avg_loss,
                "recon": epoch_avg_recon,
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
            log_metrics["threshold"] = float(model.threshold.item())
            self._log_wandb(log_metrics, commit=True)
            logger.info(
                "SAE %s epoch %d/%d | loss %.4f | recon %.4f | active %.2f | dead %.2f%% | corr %.4f | R2 %.4f",
                self._job.job_id,
                epoch_index + 1,
                planned_epochs,
                epoch_avg_loss,
                epoch_avg_recon,
                epoch_avg_active,
                dead_pct,
                epoch_corr,
                epoch_r2,
            )

        model.eval()
        self._save_feature_examples(model, tensor, combined_example_ids, combined_texts, combined_sources)

        final_metrics = last_epoch_metrics or {"loss": 0.0, "recon": 0.0, "active": 0.0, "r2": 0.0, "corr": 0.0, "dead_pct": self._dead_neuron_percent(activation_counts)}
        logger.info("SAE %s finished after %d epoch(s) and %d step(s): %s", self._job.job_id, completed_epochs, global_step, final_metrics)

        metadata_path = self._storage.sae_metadata_path(self._job.job_id)
        self._storage.write_metadata(metadata_path, {
            "job_id": self._job.job_id,
            "embedding_job": self._job.embedding_job,
            "hidden_size": self._job.hidden_size,
            "k_active": self._job.k_active,
            "activation": self._job.activation,
            "input_dim": self._input_dim,
            "epochs": completed_epochs,
            "steps": global_step,
            "learning_rate": self._job.learning_rate,
            "warmup_steps": warmup_steps,
            "warmup_ratio": self._job.warmup_ratio,
            "min_lr_scale": self._job.min_lr_scale,
            "use_cosine_decay": self._job.use_cosine_decay,
            "batch_topk_threshold_lr": self._job.batch_topk_threshold_lr,
            "aux_k": self._job.aux_k,
            "aux_loss_coef": self._job.aux_loss_coef,
            "dead_neuron_threshold_steps": self._job.dead_neuron_threshold_steps,
            "prefix_lengths": self._job.prefix_lengths,
            "grad_clip_norm": self._job.grad_clip_norm,
            "normalize_decoder": self._job.normalize_decoder,
            "dead_neuron_percent": final_metrics.get("dead_pct", 0.0),
            "threshold": float(model.threshold.item()),
            "metrics": final_metrics,
        })
        summary_metrics = dict(final_metrics)
        summary_metrics["step"] = global_step
        summary_metrics["epoch"] = completed_epochs
        summary_metrics["threshold"] = float(model.threshold.item())
        self._log_wandb(summary_metrics, commit=True)
        self._finish_wandb()
        return final_metrics

    def _save_checkpoint(self, model: SparseAutoencoder) -> None:
        path = self._storage.sae_checkpoint_path(self._job.job_id)
        payload = {
            "state_dict": model.state_dict(),
            "input_dim": self._input_dim,
            "hidden_dim": self._job.hidden_size,
            "k_active": self._job.k_active,
            "activation": self._job.activation,
            "batch_topk_threshold_lr": self._job.batch_topk_threshold_lr,
            "aux_k": self._job.aux_k,
            "dead_neuron_threshold_steps": self._job.dead_neuron_threshold_steps,
            "prefix_lengths": self._job.prefix_lengths,
        }
        torch.save(payload, path)
        logger.debug("Saved SAE checkpoint to %s", path)

    def _save_feature_examples(
        self,
        model: SparseAutoencoder,
        embeddings: torch.Tensor,
        example_ids: Optional[Sequence[int]],
        texts: Optional[Sequence[Optional[str]]],
        sources: Optional[Sequence[str]],
    ) -> None:
        top_k = max(0, self._job.top_k_feature_examples)
        if top_k == 0 or embeddings.numel() == 0:
            return

        heaps: List[List[tuple[float, int, Dict[str, object]]]] = [[] for _ in range(model.encoder.out_features)]
        batch_size = max(1, self._job.batch_size)
        effective_ids: Optional[List[object]] = None
        if example_ids is not None:
            effective_ids = []
            for idx in example_ids:
                if isinstance(idx, torch.Tensor):
                    if idx.numel() == 1:
                        effective_ids.append(idx.item())
                    else:
                        effective_ids.append(idx.detach().cpu().tolist())
                else:
                    effective_ids.append(idx)
        effective_texts: Optional[List[Optional[str]]] = list(texts) if texts is not None else None
        if effective_texts is not None:
            effective_texts = [str(text) if text is not None else None for text in effective_texts]
        effective_sources: Optional[List[str]] = list(sources) if sources is not None else None
        if effective_sources is not None:
            effective_sources = [str(source) for source in effective_sources]
        decoder_weights = model.decoder.weight.detach().cpu()
        decoder_column_norms = decoder_weights.norm(dim=0)

        with torch.no_grad():
            for start in range(0, embeddings.shape[0], batch_size):
                end = min(start + batch_size, embeddings.shape[0])
                batch = embeddings[start:end].to(self._device)
                pre_activations, codes = model.encode(batch, training=False)
                codes_cpu = codes.cpu()
                pre_cpu = pre_activations.cpu()
                for row_idx in range(codes_cpu.shape[0]):
                    example_pos = start + row_idx
                    example_identifier: object = example_pos
                    if effective_ids is not None and example_pos < len(effective_ids):
                        example_identifier = effective_ids[example_pos]
                    text_value: Optional[str] = None
                    if effective_texts is not None and example_pos < len(effective_texts):
                        candidate = effective_texts[example_pos]
                        text_value = candidate if candidate is not None else None
                    source_value: Optional[str] = None
                    if effective_sources is not None and example_pos < len(effective_sources):
                        source_value = effective_sources[example_pos]
                    row = codes_cpu[row_idx]
                    pre_row = pre_cpu[row_idx]
                    row_norm = float(row.norm(p=2).item())
                    active_indices = row.nonzero(as_tuple=False).view(-1)
                    for feature_idx in active_indices.tolist():
                        value = float(row[feature_idx].item())
                        if value <= 0.0:
                            continue
                        pre_value = float(pre_row[feature_idx].item())
                        relative = float(value / row_norm) if row_norm > 0.0 else 0.0
                        decoder_norm = float(decoder_column_norms[feature_idx].item())
                        decoder_scaled = float(value * decoder_norm)
                        entry: Dict[str, object] = {
                            "example_id": example_identifier,
                            "activation": float(value),
                            "pre_activation": float(pre_value),
                            "relative_activation": float(relative),
                            "code_l2_norm": float(row_norm),
                            "dataset_index": int(example_pos),
                            "decoder_column_norm": float(decoder_norm),
                            "decoder_scaled_activation": float(decoder_scaled),
                        }
                        if text_value is not None:
                            entry["text"] = text_value
                        if source_value is not None:
                            entry["source"] = source_value
                        heap = heaps[feature_idx]
                        item = (value, example_pos, entry)
                        if len(heap) < top_k:
                            heapq.heappush(heap, item)
                        elif value > heap[0][0]:
                            heapq.heapreplace(heap, item)

        examples_path = self._storage.sae_feature_examples_path(self._job.job_id)
        examples_path.parent.mkdir(parents=True, exist_ok=True)
        examples: Dict[str, List[Dict[str, object]]] = {}
        for feature_idx, heap in enumerate(heaps):
            if not heap:
                # Ensure a key exists for every feature even if no examples (empty list)
                examples[str(feature_idx)] = []
                continue
            sorted_entries = sorted(heap, key=lambda pair: (-pair[0], pair[1]))
            feature_examples = [entry for _, _, entry in sorted_entries]
            examples[str(feature_idx)] = feature_examples

        # Redundant safety: include empty lists for any features not covered above
        total_features = model.encoder.out_features
        for feature_idx in range(total_features):
            key = str(feature_idx)
            if key not in examples:
                examples[key] = []

        with examples_path.open("w", encoding="utf-8") as handle:
            json.dump(examples, handle, indent=2, sort_keys=True)
        logger.info("Saved top activations for SAE %s to %s", self._job.job_id, examples_path)

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
            "ss_res": 0.0,
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
            diff = x - y
            stats["ss_res"] += float(torch.sum(diff * diff, dtype=torch.float64).item())

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
