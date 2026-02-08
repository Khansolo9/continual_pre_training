#!/usr/bin/env python3
"""
Training Module for Continual Pretraining Experiments

Implements training procedures per docs/RUNBOOK.md:
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

import os
import time
import json
import math
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import GPT2LMHeadModel, GPT2Config
from tqdm import tqdm

from cl_methods import (
    ReplayBuffer, create_replay_buffer, create_mixed_batch,
    EWC, MER,
    get_method_name, get_method_params, is_replay_method, is_ewc_method, is_mer_method
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

            # Gradient accumulation
            if (pbar.n + 1) % grad_accum == 0:
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

    def setup_cl_after_domain_a(self, tokens_a: torch.Tensor):
        """
        Set up continual learning components after Domain A training.

        For EWC: Compute Fisher diagonal and store anchor weights.
        For Replay/MER: Build replay buffer from Domain A tokens.
        For MER: Initialize Reptile state.

        Args:
            tokens_a: Domain A training tokens
        """
        seq_len = self.config.get("data", {}).get("sequence_length", 512)

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

            # Apply replay mixing if enabled
            if is_replay_method(self.method) and self.replay_buffer is not None:
                input_ids, n_new, n_replay = create_mixed_batch(
                    input_ids, self.replay_buffer, replay_rate
                )
                replay_samples_used += n_replay

            # Forward pass
            outputs = self.model(input_ids, labels=input_ids)
            ce_loss = outputs.loss

            # Add EWC penalty if enabled
            if is_ewc_method(self.method) and self.ewc is not None:
                ewc_penalty = self.ewc.penalty(ewc_lambda)
                loss = ce_loss + ewc_penalty
                total_ewc_penalty += ewc_penalty.item()
            else:
                loss = ce_loss

            # Scale for gradient accumulation
            loss = loss / grad_accum

            # Backward pass
            loss.backward()
            tokens_processed += input_ids.numel()

            # Gradient accumulation step
            if (pbar.n + 1) % grad_accum == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                total_loss += loss.item() * grad_accum

                # MER: Apply Reptile update if at interval
                if is_mer_method(self.method) and self.mer is not None:
                    if self.mer.step():
                        reptile_updates += 1

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
                    if is_mer_method(self.method):
                        log_entry["reptile_updates"] = reptile_updates

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
        if is_replay_method(self.method):
            stats["replay_samples_used"] = replay_samples_used
        if is_mer_method(self.method):
            stats["reptile_updates"] = reptile_updates

        logger.info(f"Domain B training complete: {stats}")

        return stats


def create_model(model_name: str = "gpt2", gradient_checkpointing: bool = True):
    """Create GPT-2 model with optional gradient checkpointing."""
    model = GPT2LMHeadModel.from_pretrained(model_name)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model
