#!/usr/bin/env python3
"""
Training Module for Continual Pretraining Experiments

Implements training procedures per docs/specs/RUNBOOK.md:
1. Train on Domain A
2. Save checkpoint (theta_A)
3. [Optional] Compute Fisher for EWC, build replay buffer
4. Train on Domain B (with optional EWC/Replay/MER)
5. Save final checkpoint (theta_AB)

Supports:
- Baseline: Sequential fine-tuning (no mitigation)
- EWC: Elastic Weight Consolidation (Kirkpatrick2017)
- Replay: Experience Replay (Rolnick2019)
- MER: Meta-Experience Replay with Reptile (Abbes2025)

Supports gradient checkpointing for laptop VRAM constraints.
"""

import gc
import os
import time
import json
import math
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

from cl_methods import (
    ReplayBuffer, create_replay_buffer, create_mixed_batch,
    EWC, MER, BanditReplay, RMGS,
    create_probe_set, evaluate_probe,
    get_method_name, get_method_params,
    is_replay_method, is_ewc_method, is_mer_method,
    is_bandit_replay_method, is_rmgs_method,
)

logger = logging.getLogger(__name__)


def get_gpu_memory_mb() -> Optional[float]:
    """Get current GPU memory usage in MB (CUDA only). Returns None if unavailable."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return None


def get_ram_usage_mb() -> float:
    """Get current RAM usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


class CPTTrainer:
    """
    Continual Pretraining Trainer.

    Handles sequential training on Domain A then Domain B with optional
    continual learning methods (EWC, Replay, MER).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Dict[str, Any],
        device: str = "cuda",
        output_dir: Optional[Path] = None
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.output_dir = output_dir or Path("outputs")

        # Resource tracking
        self.peak_vram_mb = None  # type: Optional[float]
        self.peak_ram_mb = 0.0
        self.training_log = []

        # CL method state
        self.method = get_method_name(config)
        self.method_params = get_method_params(config)
        self.ewc: Optional[EWC] = None
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.mer: Optional[MER] = None
        self.bandit: Optional[BanditReplay] = None
        self.rmgs: Optional[RMGS] = None

        # RL method tracking (populated during Domain B training)
        self._total_effective_lr = 0.0
        self._domain_b_final_loss = 0.0

        logger.info(f"Method: {self.method}, params: {self.method_params}")

        # Enable gradient checkpointing if configured
        if config.get("model", {}).get("gradient_checkpointing", True):
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")

    def _create_dataloader(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        shuffle: bool = True
    ) -> DataLoader:
        """Create DataLoader from tokenized data."""
        # Reshape into sequences
        n_tokens = len(tokens)
        n_seqs = n_tokens // sequence_length
        tokens = tokens[:n_seqs * sequence_length].view(n_seqs, sequence_length)

        dataset = TensorDataset(tokens)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _update_resource_tracking(self):
        """Update peak resource usage."""
        if torch.cuda.is_available():
            current_vram = get_gpu_memory_mb()  # Optional[float]
            if current_vram is not None:
                if self.peak_vram_mb is None:
                    self.peak_vram_mb = current_vram
                else:
                    self.peak_vram_mb = max(self.peak_vram_mb, current_vram)
        else:
            # Non-CUDA (e.g., MPS): don't report misleading 0.0
            self.peak_vram_mb = None

        current_ram = get_ram_usage_mb()
        self.peak_ram_mb = max(self.peak_ram_mb, current_ram)

    def train_domain(
        self,
        tokens: torch.Tensor,
        domain_config: Dict[str, Any],
        domain_name: str,
        max_steps: Optional[int] = None,
        eval_callback: Optional[Callable] = None,
        log_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Train on a single domain.

        Args:
            tokens: Tokenized training data
            domain_config: Training hyperparameters
            domain_name: 'domain_a' or 'domain_b'
            max_steps: Override max steps (computed from tokens if None)
            eval_callback: Optional callback for periodic evaluation
            log_steps: Steps between logging

        Returns:
            Dictionary with training stats
        """
        self.model.train()

        # Config
        lr = domain_config.get("learning_rate", 5e-5)
        batch_size = domain_config.get("batch_size", 4)
        grad_accum = domain_config.get("gradient_accumulation_steps", 4)
        warmup_ratio = domain_config.get("warmup_ratio", 0.05)
        weight_decay = domain_config.get("weight_decay", 0.01)
        max_grad_norm = domain_config.get("max_grad_norm", 1.0)
        seq_len = self.config.get("data", {}).get("sequence_length", 512)

        # Create dataloader
        loader = self._create_dataloader(tokens, batch_size, seq_len)
        effective_batch = batch_size * grad_accum

        # Calculate steps
        steps_per_epoch = len(loader) // grad_accum
        if max_steps is None:
            max_steps = steps_per_epoch  # 1 epoch by default

        total_steps = max_steps
        warmup_steps = int(total_steps * warmup_ratio)

        logger.info(f"Training {domain_name}: {len(tokens):,} tokens, "
                    f"{total_steps} steps, batch_size={effective_batch}")

        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Scheduler: linear warmup + cosine decay
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        decay_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=lr * 0.1
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps]
        )

        # Training loop
        start_time = time.time()
        global_step = 0
        microbatch_count = 0
        total_loss = 0.0
        tokens_processed = 0

        pbar = tqdm(total=total_steps, desc=f"Training {domain_name}")
        data_iter = iter(loader)

        optimizer.zero_grad()

        while global_step < total_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch[0].to(self.device)

            # Forward pass
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss / grad_accum

            # Backward pass
            loss.backward()
            tokens_processed += input_ids.numel()
            microbatch_count += 1

            # Gradient accumulation
            if microbatch_count % grad_accum == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_grad_norm
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                total_loss += loss.item() * grad_accum

                # Update resource tracking
                self._update_resource_tracking()

                # Logging
                if global_step % log_steps == 0:
                    avg_loss = total_loss / global_step
                    elapsed = time.time() - start_time
                    tokens_per_sec = tokens_processed / elapsed

                    log_entry = {
                        "step": global_step,
                        "loss": loss.item() * grad_accum,
                        "avg_loss": avg_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "tokens_per_sec": tokens_per_sec,
                        "elapsed_sec": elapsed,
                        "domain": domain_name
                    }
                    self.training_log.append(log_entry)

                    pbar.set_postfix({
                        "loss": f"{loss.item() * grad_accum:.4f}",
                        "tok/s": f"{tokens_per_sec:.0f}"
                    })

                # Evaluation callback
                if eval_callback and global_step % self.config.get("logging", {}).get("eval_steps", 500) == 0:
                    self.model.eval()
                    eval_callback(global_step, domain_name)
                    self.model.train()

                pbar.update(1)

        pbar.close()

        # Final stats
        elapsed = time.time() - start_time
        avg_loss = total_loss / global_step if global_step > 0 else 0

        return {
            "domain": domain_name,
            "steps": global_step,
            "tokens_processed": tokens_processed,
            "final_loss": loss.item() * grad_accum if loss else 0,
            "avg_loss": avg_loss,
            "elapsed_hours": elapsed / 3600,
            "tokens_per_sec": tokens_processed / elapsed if elapsed > 0 else 0
        }

    def save_checkpoint(self, name: str, extra_info: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{name}.pt"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
        if extra_info:
            checkpoint.update(extra_info)

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint

    def _prune_old_snapshots(self, keep: int = 2):
        """Keep only the `keep` most-recent intermediate Domain-B snapshots.

        Files matching `theta_AB_step*.pt` (intermediate checkpoints written
        every domain_b_checkpoint_every_steps) are sorted by mtime and the
        oldest are deleted. The final `theta_AB.pt` and `theta_A.pt` are
        never touched by this method.
        """
        ckpt_dir = self.output_dir / "checkpoints"
        if not ckpt_dir.exists():
            return
        snaps = sorted(
            ckpt_dir.glob("theta_AB_step*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        for old in snaps[:-keep]:
            try:
                old.unlink()
            except OSError as e:
                logger.warning(f"Failed to prune snapshot {old}: {e}")

    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        peak_vram_gb = (self.peak_vram_mb / 1024) if self.peak_vram_mb is not None else None
        return {
            "peak_vram_gb": peak_vram_gb,
            "peak_ram_gb": self.peak_ram_mb / 1024
        }

    def get_training_log(self) -> list:
        """Get training log entries."""
        return self.training_log

    def save_training_log(self, path: Optional[Path] = None):
        """Save training log to JSONL file."""
        if path is None:
            path = self.output_dir / "training_log.jsonl"

        with open(path, 'w') as f:
            for entry in self.training_log:
                f.write(json.dumps(entry) + "\n")

        logger.info(f"Training log saved: {path}")

    # =========================================================================
    # CL Method Setup (call after Domain A training)
    # =========================================================================

    def setup_cl_after_domain_a(
        self,
        tokens_a: torch.Tensor,
        valid_tokens_a: Optional[torch.Tensor] = None
    ):
        """
        Set up continual learning components after Domain A training.

        For EWC: Compute Fisher diagonal and store anchor weights.
        For Replay/MER: Build replay buffer from Domain A tokens.
        For MER: Initialize Reptile state.
        For BanditReplay: Build replay buffer + initialize bandit with probe set.
        For RMGS: Initialize gradient scaling controller with probe set.

        Args:
            tokens_a: Domain A training tokens
            valid_tokens_a: Domain A validation tokens (required for RL methods)
        """
        seq_len = self.config.get("data", {}).get("sequence_length", 512)

        # Free stale MPS allocations from Domain A training
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()

        # EWC setup
        if is_ewc_method(self.method):
            logger.info("Setting up EWC...")
            self.ewc = EWC(self.model, self.device)
            fisher_samples = self.method_params.get("fisher_samples", 1000)
            batch_size = self.config.get("training", {}).get("domain_a", {}).get("batch_size", 4)
            self.ewc.compute_fisher(
                tokens_a,
                n_samples=fisher_samples,
                batch_size=batch_size,
                sequence_length=seq_len
            )
            # Flush stale allocations from Fisher computation before Domain B
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()

        # Replay buffer setup
        if is_replay_method(self.method):
            logger.info("Setting up Replay buffer...")
            buffer_size_pct = self.method_params.get("buffer_size_pct", 10)
            self.replay_buffer = create_replay_buffer(
                tokens_a,
                buffer_size_pct=buffer_size_pct,
                sequence_length=seq_len
            )

        # MER setup (on top of replay)
        if is_mer_method(self.method):
            logger.info("Setting up MER (Reptile)...")
            reptile_interval = self.method_params.get("reptile_interval", 100)
            reptile_epsilon = self.method_params.get("reptile_epsilon", 0.1)
            self.mer = MER(
                self.model,
                reptile_interval=reptile_interval,
                reptile_epsilon=reptile_epsilon
            )

        # Bandit Replay setup
        if is_bandit_replay_method(self.method):
            if valid_tokens_a is None:
                raise ValueError("valid_tokens_a required for bandit_replay method")
            logger.info("Setting up Bandit Replay...")

            # Build replay buffer (same infra as replay25)
            buffer_size_pct = self.method_params.get("buffer_size_pct", 10)
            self.replay_buffer = create_replay_buffer(
                tokens_a,
                buffer_size_pct=buffer_size_pct,
                sequence_length=seq_len
            )

            # Build probe set from validation tokens
            probe_size = self.method_params.get("probe_size", 100)
            seed = self.config.get("seed", 42)
            probe_set, probe_hash, _ = create_probe_set(
                valid_tokens_a, probe_size, seq_len, seed
            )

            # Initialize bandit
            arms = self.method_params.get("arms", [0.0, 0.1, 0.25, 0.5, 0.75])
            initial_weights = self.method_params.get(
                "initial_weights", [1, 1, 2, 1, 1]
            )
            self.bandit = BanditReplay(
                model=self.model,
                device=self.device,
                replay_buffer=self.replay_buffer,
                arms=arms,
                initial_weights=initial_weights,
                probe_set=probe_set,
                probe_hash=probe_hash,
                probe_interval=self.method_params.get("probe_interval", 50),
                exp3_gamma=self.method_params.get("exp3_gamma", 0.1),
                exp3_eta=self.method_params.get("exp3_eta", None),
                seed=seed,
            )
            self.bandit.initialize()

        # RMGS setup
        if is_rmgs_method(self.method):
            if valid_tokens_a is None:
                raise ValueError("valid_tokens_a required for rmgs method")
            logger.info("Setting up RMGS...")

            # Build probe set from validation tokens
            probe_size = self.method_params.get("probe_size", 100)
            seed = self.config.get("seed", 42)
            probe_set, probe_hash, _ = create_probe_set(
                valid_tokens_a, probe_size, seq_len, seed
            )

            self.rmgs = RMGS(
                model=self.model,
                device=self.device,
                probe_set=probe_set,
                probe_hash=probe_hash,
                probe_interval=self.method_params.get("probe_interval", 50),
                ema_alpha=self.method_params.get("ema_alpha", 0.3),
                beta=self.method_params.get("beta", 2.0),
                min_scale=self.method_params.get("min_scale", 0.05),
            )
            self.rmgs.initialize()

    # =========================================================================
    # Domain B Training with CL Methods
    # =========================================================================

    def train_domain_b(
        self,
        tokens: torch.Tensor,
        domain_config: Dict[str, Any],
        max_steps: Optional[int] = None,
        eval_callback: Optional[Callable] = None,
        log_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Train on Domain B with continual learning methods.

        Applies EWC penalty, replay mixing, and/or MER updates based on config.

        Args:
            tokens: Domain B training tokens
            domain_config: Training hyperparameters
            max_steps: Override max steps
            eval_callback: Optional callback for periodic evaluation
            log_steps: Steps between logging

        Returns:
            Dictionary with training stats
        """
        # For baseline, just use standard training
        if self.method == "baseline":
            return self.train_domain(tokens, domain_config, "domain_b",
                                     max_steps, eval_callback, log_steps)

        self.model.train()

        # Config
        lr = domain_config.get("learning_rate", 5e-5)
        batch_size = domain_config.get("batch_size", 4)
        grad_accum = domain_config.get("gradient_accumulation_steps", 4)
        warmup_ratio = domain_config.get("warmup_ratio", 0.05)
        weight_decay = domain_config.get("weight_decay", 0.01)
        max_grad_norm = domain_config.get("max_grad_norm", 1.0)
        seq_len = self.config.get("data", {}).get("sequence_length", 512)

        # EWC params
        ewc_lambda = self.method_params.get("ewc_lambda", 100.0)

        # Replay params
        replay_rate = self.method_params.get("mixing_ratio_replay",
                      self.method_params.get("replay_rate", 0.25))

        # Dedicated RNG for replay-fractional-rounding draws (AMENDMENT-004).
        # Kept separate from torch's global generator so that adding the
        # Bernoulli draw does not perturb data-shuffle / dropout streams in
        # otherwise-identical runs. Seed precedence: explicit
        # `replay_rng_seed` config > base seed + 1 > 43.
        base_seed = self.config.get("seed", 42)
        replay_rng_seed = self.config.get("replay_rng_seed", base_seed + 1)
        replay_rng = random.Random(replay_rng_seed)

        # Track nominal (configured/bandit-current) and effective (realized)
        # replay accounting separately. The pre-AMENDMENT-004 logs only had
        # `replay_samples` which conflated the two and silently read 0.
        nominal_microbatches_with_replay = 0
        sum_nominal_replay_rate = 0.0

        # Create dataloader for Domain B
        loader = self._create_dataloader(tokens, batch_size, seq_len)
        effective_batch = batch_size * grad_accum

        # Calculate steps
        steps_per_epoch = len(loader) // grad_accum
        if max_steps is None:
            max_steps = steps_per_epoch

        total_steps = max_steps
        warmup_steps = int(total_steps * warmup_ratio)

        method_desc = f"{self.method}"
        if is_ewc_method(self.method):
            method_desc += f" (λ={ewc_lambda})"
        if is_replay_method(self.method):
            method_desc += f" (replay={replay_rate:.0%})"
        if is_mer_method(self.method):
            method_desc += f" (reptile every {self.mer.reptile_interval} steps)"
        if is_bandit_replay_method(self.method) and self.bandit is not None:
            method_desc += f" (bandit arms={len(self.bandit.arms)})"
        if is_rmgs_method(self.method) and self.rmgs is not None:
            method_desc += f" (beta={self.rmgs.beta}, min_scale={self.rmgs.min_scale})"

        logger.info(f"Training domain_b with {method_desc}: {len(tokens):,} tokens, "
                    f"{total_steps} steps, batch_size={effective_batch}")

        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Scheduler
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        decay_scheduler = CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=lr * 0.1
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps]
        )

        # Training loop
        start_time = time.time()
        global_step = 0
        microbatch_count = 0
        total_loss = 0.0
        total_ewc_penalty = 0.0
        tokens_processed = 0
        replay_samples_used = 0
        reptile_updates = 0

        pbar = tqdm(total=total_steps, desc=f"Training domain_b ({self.method})")
        data_iter = iter(loader)

        optimizer.zero_grad()

        while global_step < total_steps:
            # Get batch from Domain B
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch[0].to(self.device)

            # Apply replay mixing if enabled (fixed-rate methods)
            if is_replay_method(self.method) and self.replay_buffer is not None:
                input_ids, n_new, n_replay = create_mixed_batch(
                    input_ids, self.replay_buffer, replay_rate, rng=replay_rng
                )
                replay_samples_used += n_replay
                sum_nominal_replay_rate += replay_rate
                nominal_microbatches_with_replay += 1

            # Apply bandit-guided replay mixing (adaptive rate)
            if is_bandit_replay_method(self.method) and self.bandit is not None:
                bandit_rate = self.bandit.get_current_rate()
                input_ids, n_new, n_replay = create_mixed_batch(
                    input_ids, self.replay_buffer, bandit_rate, rng=replay_rng
                )
                replay_samples_used += n_replay
                sum_nominal_replay_rate += bandit_rate
                nominal_microbatches_with_replay += 1

            # Forward pass
            outputs = self.model(input_ids, labels=input_ids)
            ce_loss = outputs.loss

            # CE loss: scaled by 1/grad_accum so accumulated grads = avg(∇CE).
            loss = ce_loss / grad_accum
            loss.backward()

            # EWC penalty: applied ONCE per gradient step (last microbatch of
            # each accumulation window). Uses analytic gradient via
            # apply_penalty_grad() — bypasses autograd to avoid building a
            # 4-6 GB intermediate graph for ~1B-param models, which on
            # Apple-Silicon unified memory pushes runs into MPS swap (~6x
            # slowdown observed pre-fix). Mathematically equivalent to the
            # previous penalty().backward() path.
            is_accum_boundary = (microbatch_count + 1) % grad_accum == 0
            if is_ewc_method(self.method) and self.ewc is not None and is_accum_boundary:
                penalty_value = self.ewc.apply_penalty_grad(ewc_lambda)
                total_ewc_penalty += penalty_value

            tokens_processed += input_ids.numel()
            microbatch_count += 1

            # Gradient accumulation step
            if microbatch_count % grad_accum == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                total_loss += loss.item() * grad_accum

                # Periodic MPS cache release: prevents allocator fragmentation
                # accumulation on multi-hour 1B-class runs. Bit-equivalent to
                # not calling it (no tensor data is altered); only frees the
                # allocator's free-block list so cached blocks can return to
                # the OS. Cost is small (<100 ms per call). Disabled when not
                # on MPS or when the empty_cache_every config is 0/missing.
                empty_cache_every = self.config.get(
                    "empty_cache_every_steps", 200
                )
                if (
                    empty_cache_every > 0
                    and global_step % empty_cache_every == 0
                    and self.device == "mps"
                    and hasattr(torch.mps, "empty_cache")
                ):
                    torch.mps.empty_cache()

                # Mid-Domain-B checkpointing: lets us recover from an OOM
                # kill mid-run instead of losing 5-10h of work. The final
                # theta_AB.pt is still written on successful completion;
                # these intermediate snapshots are pruned to keep only the
                # latest two (~4 GB disk for 1B model in bf16).
                ckpt_every = self.config.get(
                    "domain_b_checkpoint_every_steps", 200
                )
                if ckpt_every > 0 and global_step % ckpt_every == 0:
                    snap_name = f"theta_AB_step{global_step}"
                    self.save_checkpoint(
                        snap_name,
                        extra_info={"domain": "domain_b", "step": global_step}
                    )
                    self._prune_old_snapshots(keep=2)

                # MER: Apply Reptile update if at interval
                if is_mer_method(self.method) and self.mer is not None:
                    if self.mer.step():
                        reptile_updates += 1

                # Bandit Replay: evaluate probe and update bandit
                if is_bandit_replay_method(self.method) and self.bandit is not None:
                    self.bandit.step()

                # RMGS: evaluate probe, update scale, modulate LR
                if is_rmgs_method(self.method) and self.rmgs is not None:
                    self.rmgs.step()
                    scale = self.rmgs.get_scale()
                    base_lr = scheduler.get_last_lr()[0]
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = base_lr * scale
                    self._total_effective_lr += base_lr * scale

                # Update resource tracking
                self._update_resource_tracking()

                # Logging
                if global_step % log_steps == 0:
                    avg_loss = total_loss / global_step
                    elapsed = time.time() - start_time
                    tokens_per_sec = tokens_processed / elapsed

                    log_entry = {
                        "step": global_step,
                        "loss": loss.item() * grad_accum,
                        "avg_loss": avg_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "tokens_per_sec": tokens_per_sec,
                        "elapsed_sec": elapsed,
                        "domain": "domain_b",
                        "method": self.method
                    }

                    if is_ewc_method(self.method):
                        log_entry["ewc_penalty"] = ewc_penalty.item() if 'ewc_penalty' in dir() else 0
                    if is_replay_method(self.method):
                        log_entry["replay_samples"] = replay_samples_used
                        log_entry["nominal_replay_rate"] = replay_rate
                        log_entry["effective_replay_samples"] = replay_samples_used
                        if nominal_microbatches_with_replay > 0:
                            log_entry["effective_replay_rate"] = (
                                replay_samples_used
                                / (nominal_microbatches_with_replay * batch_size)
                            )
                        log_entry["replay_rng_seed"] = replay_rng_seed
                    if is_mer_method(self.method):
                        log_entry["reptile_updates"] = reptile_updates
                    if is_bandit_replay_method(self.method) and self.bandit is not None:
                        log_entry["replay_rate"] = self.bandit.get_current_rate()
                        log_entry["nominal_replay_rate"] = self.bandit.get_current_rate()
                        log_entry["replay_samples"] = replay_samples_used
                        log_entry["effective_replay_samples"] = replay_samples_used
                        if nominal_microbatches_with_replay > 0:
                            log_entry["effective_replay_rate"] = (
                                replay_samples_used
                                / (nominal_microbatches_with_replay * batch_size)
                            )
                        log_entry["replay_rng_seed"] = replay_rng_seed
                    if is_rmgs_method(self.method) and self.rmgs is not None:
                        log_entry["gradient_scale"] = self.rmgs.get_scale()
                        log_entry["effective_lr"] = scheduler.get_last_lr()[0] * self.rmgs.get_scale()

                    self.training_log.append(log_entry)

                    # === CL METHOD INSTRUMENTATION (visible in logs) ===
                    cl_status = []
                    if is_replay_method(self.method) and self.replay_buffer is not None:
                        buf_size = len(self.replay_buffer)
                        buf_cap = self.replay_buffer.capacity
                        realized_frac = replay_samples_used / (tokens_processed / seq_len) if tokens_processed > 0 else 0
                        cl_status.append(f"Replay[{buf_size}/{buf_cap}] frac={realized_frac:.2%}")
                    if is_ewc_method(self.method) and self.ewc is not None:
                        n_params = len(self.ewc.fisher_diag) if self.ewc.fisher_diag else 0
                        avg_penalty = total_ewc_penalty / global_step if global_step > 0 else 0
                        cl_status.append(f"EWC[params={n_params}] avg_penalty={avg_penalty:.4f}")
                    if is_mer_method(self.method) and self.mer is not None:
                        last_snap = self.mer.step_counter - (self.mer.step_counter % self.mer.reptile_interval)
                        cl_status.append(f"MER[updates={reptile_updates}] last_snap={last_snap}")
                    if is_bandit_replay_method(self.method) and self.bandit is not None:
                        cl_status.append(f"Bandit[rate={self.bandit.get_current_rate():.2f}, evals={len(self.bandit.arm_history)}]")
                    if is_rmgs_method(self.method) and self.rmgs is not None:
                        cl_status.append(f"RMGS[scale={self.rmgs.get_scale():.4f}, evals={len(self.rmgs.scale_history)}]")
                    if cl_status:
                        logger.info(f"  [CL] step={global_step}: {' | '.join(cl_status)}")

                    postfix = {"loss": f"{loss.item() * grad_accum:.4f}", "tok/s": f"{tokens_per_sec:.0f}"}
                    if is_ewc_method(self.method) and 'ewc_penalty' in dir():
                        postfix["ewc"] = f"{ewc_penalty.item():.4f}"
                    pbar.set_postfix(postfix)

                # Evaluation callback
                if eval_callback and global_step % self.config.get("logging", {}).get("eval_steps", 500) == 0:
                    self.model.eval()
                    eval_callback(global_step, "domain_b")
                    self.model.train()

                pbar.update(1)

        pbar.close()

        # Final stats
        elapsed = time.time() - start_time
        avg_loss = total_loss / global_step if global_step > 0 else 0

        stats = {
            "domain": "domain_b",
            "method": self.method,
            "steps": global_step,
            "tokens_processed": tokens_processed,
            "final_loss": loss.item() * grad_accum if 'loss' in dir() else 0,
            "avg_loss": avg_loss,
            "elapsed_hours": elapsed / 3600,
            "tokens_per_sec": tokens_processed / elapsed if elapsed > 0 else 0
        }

        if is_ewc_method(self.method):
            stats["total_ewc_penalty"] = total_ewc_penalty
            stats["avg_ewc_penalty"] = total_ewc_penalty / global_step if global_step > 0 else 0
        # Replay accounting (AMENDMENT-004): nominal = configured/scheduled rate;
        # effective = realized samples / total slots. The two diverge whenever
        # batch_size * rate is non-integer, in which case stochastic rounding
        # produces an integer per microbatch whose long-run mean equals
        # the nominal rate.
        if is_replay_method(self.method) or is_bandit_replay_method(self.method):
            stats["replay_samples_used"] = replay_samples_used
            stats["effective_replay_samples"] = replay_samples_used
            stats["replay_rng_seed"] = replay_rng_seed
            if nominal_microbatches_with_replay > 0:
                stats["nominal_replay_rate_mean"] = (
                    sum_nominal_replay_rate / nominal_microbatches_with_replay
                )
                total_slots = nominal_microbatches_with_replay * batch_size
                stats["effective_replay_rate"] = replay_samples_used / total_slots
        if is_mer_method(self.method):
            stats["reptile_updates"] = reptile_updates
        if is_rmgs_method(self.method):
            self._domain_b_final_loss = stats["final_loss"]

        logger.info(f"Domain B training complete: {stats}")

        return stats

    def get_rl_method_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get RL method statistics for metrics.json under 'rl_method_stats'.

        Returns None for non-RL methods.
        """
        if is_bandit_replay_method(self.method) and self.bandit is not None:
            stats = self.bandit.get_stats()
            # Cross-run metrics (require baseline comparison, computed in analysis)
            stats["domain_b_adaptation_ratio"] = None
            stats["tradeoff_efficiency"] = None
            return stats
        elif is_rmgs_method(self.method) and self.rmgs is not None:
            stats = self.rmgs.get_stats()
            stats["total_effective_lr_integral"] = self._total_effective_lr
            stats["domain_b_training_loss_final"] = self._domain_b_final_loss
            # Cross-run metrics (require baseline comparison, computed in analysis)
            stats["domain_b_adaptation_ratio"] = None
            stats["tradeoff_efficiency"] = None
            return stats
        return None


def create_model(
    model_name: str = "gpt2",
    gradient_checkpointing: bool = True,
    torch_dtype=None,
    trust_remote_code: bool = False,
):
    """Create a causal LM from any HuggingFace model identifier."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype or torch.float32,
        trust_remote_code=trust_remote_code,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model
