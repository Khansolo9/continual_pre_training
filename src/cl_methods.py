#!/usr/bin/env python3
"""
Continual Learning Methods Module

Implements:
- EWC (Elastic Weight Consolidation) - Kirkpatrick2017
- Replay (Experience Replay) - Rolnick2019
- MER (Meta-Experience Replay) - Abbes2025

These methods are designed to reduce catastrophic forgetting during
continual pretraining on sequential domains.
"""

import random
import logging
import copy
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# REPLAY BUFFER (Experience Replay)
# =============================================================================

class ReplayBuffer:
    """
    Replay buffer with reservoir sampling for experience replay.

    Implements reservoir sampling to maintain a uniform random sample
    from the entire training history. Per Rolnick2019 and Abbes2025.

    Args:
        capacity: Maximum number of sequences to store
        sequence_length: Length of each sequence
    """

    def __init__(self, capacity: int, sequence_length: int = 512):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer: List[torch.Tensor] = []
        self.total_seen = 0

    def add(self, sequence: torch.Tensor):
        """
        Add a sequence to the buffer using reservoir sampling.

        Args:
            sequence: Token tensor of shape (sequence_length,)
        """
        self.total_seen += 1

        if len(self.buffer) < self.capacity:
            self.buffer.append(sequence.clone())
        else:
            # Reservoir sampling: replace with probability capacity/total_seen
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = sequence.clone()

    def add_batch(self, sequences: torch.Tensor):
        """
        Add a batch of sequences to the buffer.

        Args:
            sequences: Token tensor of shape (batch_size, sequence_length)
        """
        for i in range(sequences.shape[0]):
            self.add(sequences[i])

    def sample(self, k: int) -> torch.Tensor:
        """
        Sample k sequences from the buffer.

        Args:
            k: Number of sequences to sample

        Returns:
            Tensor of shape (k, sequence_length)
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")

        k = min(k, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), k)
        samples = [self.buffer[i] for i in indices]
        return torch.stack(samples)

    def __len__(self):
        return len(self.buffer)

    def fill_from_tokens(self, tokens: torch.Tensor):
        """
        Fill buffer from a token tensor by chunking into sequences.

        Args:
            tokens: 1D tensor of all tokens
        """
        n_tokens = len(tokens)
        n_seqs = n_tokens // self.sequence_length

        for i in range(n_seqs):
            start = i * self.sequence_length
            end = start + self.sequence_length
            self.add(tokens[start:end])

        logger.info(f"Replay buffer filled: {len(self.buffer)}/{self.capacity} sequences "
                    f"from {n_tokens:,} tokens")


def create_replay_buffer(
    tokens: torch.Tensor,
    buffer_size_pct: float,
    sequence_length: int = 512
) -> ReplayBuffer:
    """
    Create and fill a replay buffer from training tokens.

    Args:
        tokens: 1D tensor of all training tokens
        buffer_size_pct: Percentage of data to store (e.g., 10 for 10%)
        sequence_length: Length of each sequence

    Returns:
        Filled ReplayBuffer
    """
    n_tokens = len(tokens)
    n_seqs_total = n_tokens // sequence_length
    capacity = max(1, int(n_seqs_total * buffer_size_pct / 100))

    buffer = ReplayBuffer(capacity=capacity, sequence_length=sequence_length)
    buffer.fill_from_tokens(tokens)

    return buffer


# =============================================================================
# EWC (Elastic Weight Consolidation)
# =============================================================================

class EWC:
    """
    Elastic Weight Consolidation implementation.

    After training on Domain A, computes the diagonal Fisher Information Matrix
    and stores anchor weights. During Domain B training, adds a quadratic
    penalty to prevent weights from deviating too far from the anchor.

    Per Kirkpatrick2017:
        L(θ) = L_CE(θ) + (λ/2) Σ_i F_i(θ_i - θ*_i)²

    Args:
        model: The neural network model
        device: Device to use for computation
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.theta_star: Optional[Dict[str, torch.Tensor]] = None
        self.fisher_diag: Optional[Dict[str, torch.Tensor]] = None
        self._computed = False

    def compute_fisher(
        self,
        tokens: torch.Tensor,
        n_samples: int = 1000,
        batch_size: int = 4,
        sequence_length: int = 512
    ):
        """
        Compute the diagonal Fisher Information Matrix on Domain A data.

        Fisher diagonal: F_i = E[(∂log p(x|θ)/∂θ_i)²]

        IMPORTANT: Uses sum reduction over tokens (not mean) before squaring
        gradients to avoid ~T underestimation where T is sequence length.
        Per Kirkpatrick2017, Fisher should be mean of squared per-token gradients,
        not squared mean gradient.

        Args:
            tokens: Training tokens from Domain A
            n_samples: Number of samples for Fisher estimate
            batch_size: Batch size for computation
            sequence_length: Sequence length
        """
        logger.info(f"Computing Fisher diagonal on {n_samples} samples...")

        # Store anchor weights
        self.theta_star = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Initialize Fisher accumulator
        self.fisher_diag = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Create dataloader for Fisher computation
        n_tokens = len(tokens)
        n_seqs = min(n_samples, n_tokens // sequence_length)
        tokens_subset = tokens[:n_seqs * sequence_length].view(n_seqs, sequence_length)

        dataset = TensorDataset(tokens_subset)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.eval()
        n_processed = 0
        total_tokens = 0  # Track total tokens for proper normalization

        for batch in tqdm(loader, desc="Fisher computation", leave=False):
            if n_processed >= n_samples:
                break

            input_ids = batch[0].to(self.device)

            # Forward pass - get logits WITHOUT labels to avoid HF's mean reduction
            self.model.zero_grad()
            outputs = self.model(input_ids)
            logits = outputs.logits

            # Construct shifted logits/labels for causal LM (standard autoregressive setup)
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Compute cross-entropy with reduction='sum' to avoid token-level averaging
            # This ensures gradients are NOT divided by sequence length before squaring
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )

            # Backward pass to get gradients (now properly scaled)
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_diag[name] += param.grad.detach() ** 2

            n_processed += input_ids.shape[0]
            total_tokens += shift_labels.numel()  # Count actual tokens used

        # Normalize by total tokens (not samples) for proper Fisher scaling
        for name in self.fisher_diag:
            self.fisher_diag[name] /= total_tokens

        self._computed = True
        self.model.train()

        # Log statistics without concatenating all Fisher values (avoids OOM on large models)
        total_params = sum(f.numel() for f in self.fisher_diag.values())
        weighted_sum = sum(f.sum().item() for f in self.fisher_diag.values())
        mean_fisher = weighted_sum / total_params if total_params > 0 else 0.0
        max_fisher = max(f.max().item() for f in self.fisher_diag.values())

        logger.info(
            f"Fisher computed: {total_params:,} parameters, "
            f"mean={mean_fisher:.6f}, max={max_fisher:.6f}, "
            f"total_tokens={total_tokens:,}"
        )

    def penalty(self, ewc_lambda: float = 100.0) -> torch.Tensor:
        """
        Compute the EWC penalty term.

        Penalty = (λ/2) Σ_i F_i(θ_i - θ*_i)²

        Args:
            ewc_lambda: Regularization strength

        Returns:
            Scalar penalty tensor
        """
        if not self._computed:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if name in self.fisher_diag and name in self.theta_star:
                diff = param - self.theta_star[name]
                penalty += (self.fisher_diag[name] * diff ** 2).sum()

        return (ewc_lambda / 2) * penalty

    @property
    def is_computed(self) -> bool:
        """Check if Fisher has been computed."""
        return self._computed

    def get_fisher_stats(self) -> Dict[str, float]:
        """
        Get summary statistics of Fisher diagonal for diagnostics.

        Returns:
            Dict with mean, median, max, min Fisher values
        """
        if not self._computed or self.fisher_diag is None:
            return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0}

        all_fisher = torch.cat([f.flatten() for f in self.fisher_diag.values()])
        return {
            "mean": all_fisher.mean().item(),
            "median": all_fisher.median().item(),
            "max": all_fisher.max().item(),
            "min": all_fisher.min().item(),
            "n_params": all_fisher.numel(),
        }


# =============================================================================
# MER (Meta-Experience Replay with Reptile)
# =============================================================================

class MER:
    """
    Meta-Experience Replay with Reptile-style meta-updates.

    Per Abbes2025, MER combines experience replay with Reptile gradient
    alignment. Every k steps, the model weights are interpolated toward
    a snapshot taken k steps ago:

        θ ← θ_old + ε(θ - θ_old)

    This promotes gradient alignment across tasks and reduces interference.

    Args:
        model: The neural network model
        reptile_interval: Steps between meta-updates
        reptile_epsilon: Interpolation coefficient (0 < ε < 1)
    """

    def __init__(
        self,
        model: nn.Module,
        reptile_interval: int = 100,
        reptile_epsilon: float = 0.1
    ):
        self.model = model
        self.reptile_interval = reptile_interval
        self.reptile_epsilon = reptile_epsilon

        self.theta_old: Optional[Dict[str, torch.Tensor]] = None
        self.step_counter = 0
        self.update_counter = 0

    def snapshot(self):
        """Take a snapshot of current weights as θ_old."""
        self.theta_old = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        logger.debug(f"MER snapshot taken at step {self.step_counter}")

    def step(self) -> bool:
        """
        Called after each training step. Applies Reptile update if at interval.

        Returns:
            True if Reptile update was applied, False otherwise
        """
        self.step_counter += 1

        if self.step_counter % self.reptile_interval == 0:
            if self.theta_old is not None:
                self._apply_reptile_update()
                self.update_counter += 1
                return True

            # First interval - just take snapshot
            self.snapshot()

        return False

    def _apply_reptile_update(self):
        """
        Apply Reptile meta-update:
            θ ← θ_old + ε(θ - θ_old)
            = (1-ε)θ_old + ε*θ
            = θ_old + ε*(θ - θ_old)

        This interpolates toward the new weights but keeps some of the old.
        """
        if self.theta_old is None:
            return

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.theta_old and param.requires_grad:
                    # θ ← θ_old + ε(θ - θ_old)
                    diff = param.data - self.theta_old[name]
                    param.data = self.theta_old[name] + self.reptile_epsilon * diff

        # Take new snapshot for next interval
        self.snapshot()

        logger.debug(f"Reptile update applied at step {self.step_counter} "
                     f"(update #{self.update_counter})")

    def reset(self):
        """Reset counters and snapshot for a new training phase."""
        self.theta_old = None
        self.step_counter = 0
        self.update_counter = 0


# =============================================================================
# MIXED BATCH CREATION (for Replay and MER)
# =============================================================================

def create_mixed_batch(
    new_batch: torch.Tensor,
    replay_buffer: ReplayBuffer,
    replay_rate: float
) -> Tuple[torch.Tensor, int, int]:
    """
    Create a mixed batch with both new and replay samples.

    Args:
        new_batch: New domain data of shape (batch_size, seq_len)
        replay_buffer: Buffer containing old domain samples
        replay_rate: Fraction of batch that should be replay (0.0 to 1.0)

    Returns:
        Tuple of (mixed_batch, n_new, n_replay)
    """
    batch_size = new_batch.shape[0]
    n_replay = int(batch_size * replay_rate)
    n_new = batch_size - n_replay

    if n_replay == 0 or len(replay_buffer) == 0:
        return new_batch, batch_size, 0

    # Sample from replay buffer
    replay_samples = replay_buffer.sample(n_replay)
    replay_samples = replay_samples.to(new_batch.device)

    # Combine: first n_new from new, then n_replay from buffer
    mixed_batch = torch.cat([new_batch[:n_new], replay_samples], dim=0)

    return mixed_batch, n_new, n_replay


# =============================================================================
# HELPER: Get method from config
# =============================================================================

def get_method_name(config: Dict[str, Any]) -> str:
    """Extract method name from config, defaulting to 'baseline'."""
    return config.get("method", "baseline")


def get_method_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract method parameters from config."""
    return config.get("method_params", {})


def is_replay_method(method: str) -> bool:
    """Check if method uses replay buffer."""
    return method in ["replay25", "replay50", "mer25", "mer50"]


def is_ewc_method(method: str) -> bool:
    """Check if method uses EWC."""
    return method == "ewc"


def is_mer_method(method: str) -> bool:
    """Check if method uses MER (Reptile updates)."""
    return method in ["mer25", "mer50"]
