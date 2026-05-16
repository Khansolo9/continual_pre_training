#!/usr/bin/env python3
"""
Continual Learning Methods Module

Implements:
- EWC (Elastic Weight Consolidation) - Kirkpatrick2017
- Replay (Experience Replay) - Rolnick2019
- MER (Meta-Experience Replay) - Abbes2025
- BanditReplay (Bandit-Guided Replay Scheduling) - RL Phase
- RMGS (Reward-Modulated Gradient Scaling) - RL Phase

These methods are designed to reduce catastrophic forgetting during
continual pretraining on sequential domains.
"""

import random
import logging
import copy
import math
import hashlib
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

    def apply_penalty_grad(self, ewc_lambda: float = 100.0) -> float:
        """
        Add the analytic EWC gradient directly to param.grad and return the
        scalar penalty value for logging.

        For diagonal-Fisher EWC the gradient is exact and trivial:
            ∂[(λ/2) F_i (θ_i - θ*_i)²] / ∂θ_i = λ F_i (θ_i - θ*_i)

        Bypassing autograd avoids building a 4-6 GB graph of bf16
        intermediates for ~1B-param models, which on Apple-Silicon unified
        memory pushes Gemma3 1B / Llama3 1B into MPS swap (~6x slowdown
        observed pre-fix). Mathematically equivalent to penalty().backward().

        Implementation uses torch._foreach_* fused ops to collapse ~340
        per-parameter Python loop iterations into ~5 batched MPS dispatches.
        On Gemma3 1B this brings EWC tok/s from ~178 to ~300+ by eliminating
        Python-loop kernel-dispatch overhead (~140 ms/step at 340 params).
        Bit-equivalent to the per-parameter loop within bf16 reduction order.

        Must be called AFTER ce_loss.backward() and BEFORE optimizer.step().
        """
        if not self._computed:
            return 0.0

        # Gather aligned tensor lists once. _foreach_* requires same dtype
        # and device per list, which is invariant under named_parameters() here.
        params, fishers, thetas, grads = [], [], [], []
        for name, param in self.model.named_parameters():
            if name not in self.fisher_diag or name not in self.theta_star:
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            params.append(param.detach())
            fishers.append(self.fisher_diag[name])
            thetas.append(self.theta_star[name])
            grads.append(param.grad)

        if not params:
            return 0.0

        with torch.no_grad():
            # diffs_i = θ_i - θ*_i   (one fused kernel across all tensors)
            diffs = torch._foreach_sub(params, thetas)
            # weighted_i = F_i · diffs_i
            weighted = torch._foreach_mul(fishers, diffs)
            # grads_i += λ · weighted_i  (in-place fused add)
            torch._foreach_add_(grads, weighted, alpha=ewc_lambda)
            # penalty value: Σ_i (F_i · diffs_i²).sum() — log-only, single CPU sync
            sq = torch._foreach_mul(weighted, diffs)
            penalty_value = sum(t.sum() for t in sq).item()

        return (ewc_lambda / 2.0) * penalty_value

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
    replay_rate: float,
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, int, int]:
    """
    Create a mixed batch with both new and replay samples.

    Replay-count rounding is stochastic and fractional (AMENDMENT-004,
    2026-05-15). For an `expected = batch_size * replay_rate`, the realized
    count is `floor(expected) + Bernoulli(expected - floor(expected))`. This
    guarantees `E[n_replay] == expected` even when `batch_size * replay_rate`
    is not integer-valued — fixing the prior `int(...)` truncation that
    silently produced `replay_samples == 0` for the 1B configs
    (batch_size=2, replay_rate=0.25 → int(0.5) == 0).

    Args:
        new_batch: New domain data of shape (batch_size, seq_len)
        replay_buffer: Buffer containing old domain samples
        replay_rate: Fraction of batch that should be replay (0.0 to 1.0)
        rng: Optional `random.Random` instance for the fractional Bernoulli
            draw. If None, uses the module-global `random` state. Callers
            that need run-to-run reproducibility (the trainer) should pass
            a seeded instance.

    Returns:
        Tuple of (mixed_batch, n_new, n_replay)
    """
    batch_size = new_batch.shape[0]

    expected = batch_size * replay_rate
    n_floor = int(math.floor(expected))
    frac = expected - n_floor
    if frac > 0.0:
        draw = (rng or random).random()
        n_replay = n_floor + (1 if draw < frac else 0)
    else:
        n_replay = n_floor
    # Defensive cap: replay_rate ∈ [0, 1] keeps this within bounds, but a
    # caller-supplied rate slightly > 1 (e.g. from a bandit numerical edge)
    # should not crash sampling.
    n_replay = max(0, min(n_replay, batch_size))
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


def is_bandit_replay_method(method: str) -> bool:
    """Check if method uses bandit-guided replay scheduling."""
    return method == "bandit_replay"


def is_rmgs_method(method: str) -> bool:
    """Check if method uses reward-modulated gradient scaling."""
    return method == "rmgs"


# =============================================================================
# PROBE SET (shared by bandit_replay and rmgs)
# =============================================================================

def create_probe_set(
    valid_tokens: torch.Tensor,
    probe_size: int,
    sequence_length: int,
    seed: int
) -> Tuple[torch.Tensor, str, List[int]]:
    """
    Create a deterministic probe set from Domain A validation tokens.

    The probe set is derived deterministically from the seed using a
    separate RNG to avoid coupling with other random operations.
    The probe is fixed once at initialization and never re-sampled.

    Args:
        valid_tokens: 1D tensor of Domain A validation tokens
        probe_size: Number of sequences in the probe
        sequence_length: Length of each sequence
        seed: Run seed for deterministic derivation

    Returns:
        Tuple of (probe_tensor, probe_hash, probe_indices)
        - probe_tensor: shape (probe_size, sequence_length)
        - probe_hash: SHA256 hex digest of probe indices
        - probe_indices: list of selected sequence indices
    """
    n_tokens = len(valid_tokens)
    n_seqs = n_tokens // sequence_length

    if n_seqs == 0:
        raise ValueError(
            f"Not enough validation tokens ({n_tokens}) for "
            f"sequence_length={sequence_length}"
        )

    # Use a separate RNG seeded deterministically from the run seed
    # This ensures probe selection is independent of prior random operations
    probe_rng = random.Random(seed * 31 + 7919)

    actual_probe_size = min(probe_size, n_seqs)
    probe_indices = sorted(probe_rng.sample(range(n_seqs), actual_probe_size))

    # Build probe tensor
    all_seqs = valid_tokens[:n_seqs * sequence_length].view(n_seqs, sequence_length)
    probe_tensor = all_seqs[probe_indices].clone()

    # Compute SHA256 hash of probe indices
    index_bytes = str(probe_indices).encode('utf-8')
    probe_hash = hashlib.sha256(index_bytes).hexdigest()

    logger.info(
        f"Probe set created: {actual_probe_size} sequences from "
        f"{n_seqs} available, hash={probe_hash[:16]}..."
    )

    return probe_tensor, probe_hash, probe_indices


@torch.no_grad()
def evaluate_probe(
    model: nn.Module,
    probe_set: torch.Tensor,
    device: str,
    batch_size: int = 1
) -> float:
    """
    Evaluate mean NLL on a probe set. No gradients computed.

    Args:
        model: The model to evaluate
        probe_set: Tensor of shape (n_probes, seq_len) on CPU
        device: Device to use for evaluation
        batch_size: Evaluation batch size (default 1 to avoid OOM on
                    large-vocab models like Qwen3 with 151K vocab)

    Returns:
        Mean NLL (token-weighted) across the probe set
    """
    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(probe_set), batch_size):
        batch = probe_set[i:i + batch_size].to(device)
        outputs = model(batch, labels=batch)
        n_tokens = batch.numel()
        total_nll += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
        del outputs, batch

    if was_training:
        model.train()

    return total_nll / total_tokens if total_tokens > 0 else float('inf')


# =============================================================================
# BANDIT REPLAY (Bandit-Guided Replay Scheduling)
# =============================================================================

class BanditReplay:
    """
    Multi-armed bandit for adaptive replay rate scheduling.

    Uses EXP3 algorithm to select replay rates from a discrete set of arms.
    Reward is the negative change in Domain A probe loss
    (positive reward = retention improving).

    Per RL_PHASE_PLAN.md Section 2.1 and Appendix A.1.

    Args:
        model: The neural network model
        device: Device string
        replay_buffer: Filled replay buffer from Domain A
        arms: List of discrete replay rate options
        initial_weights: Warm-start weights for each arm
        probe_set: Probe tensor on CPU, shape (probe_size, seq_len)
        probe_hash: SHA256 hash of probe indices
        probe_interval: Gradient steps between probe evaluations
        exp3_gamma: EXP3 exploration parameter
        exp3_eta: EXP3 learning rate (auto-computed if None)
        seed: Run seed for deterministic bandit RNG
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        replay_buffer: ReplayBuffer,
        arms: List[float],
        initial_weights: List[float],
        probe_set: torch.Tensor,
        probe_hash: str,
        probe_interval: int,
        exp3_gamma: float = 0.1,
        exp3_eta: Optional[float] = None,
        seed: int = 42
    ):
        self.model = model
        self.device = device
        self.replay_buffer = replay_buffer
        self.arms = arms
        self.K = len(arms)
        self.probe_set = probe_set
        self.probe_hash = probe_hash
        self.probe_interval = probe_interval
        self.exp3_gamma = exp3_gamma

        # Auto-compute eta if not specified (use gamma as default)
        self.exp3_eta = exp3_eta if exp3_eta is not None else exp3_gamma

        # Bandit state
        self.weights = list(initial_weights) if initial_weights else [1.0] * self.K
        self.rng = random.Random(seed * 37 + 1009)  # deterministic bandit RNG

        # Current arm
        self.current_arm_idx = None
        self.current_rate = arms[2] if len(arms) > 2 else arms[0]

        # History tracking
        self.arm_history: List[int] = []
        self.reward_history: List[float] = []
        self.probe_loss_history: List[float] = []
        self.last_probe_loss: Optional[float] = None

        # Step counter
        self.step_counter = 0

    def initialize(self):
        """Evaluate initial probe loss and select first arm."""
        self.last_probe_loss = evaluate_probe(
            self.model, self.probe_set, self.device
        )
        self.probe_loss_history.append(self.last_probe_loss)
        self._select_arm()
        logger.info(
            f"BanditReplay initialized: probe_loss={self.last_probe_loss:.4f}, "
            f"initial_rate={self.current_rate}"
        )

    def _get_probabilities(self) -> List[float]:
        """Compute EXP3 arm probabilities with exploration."""
        total_w = sum(self.weights)
        probs = []
        for w in self.weights:
            p = (1 - self.exp3_gamma) * (w / total_w) + self.exp3_gamma / self.K
            probs.append(p)
        return probs

    def _select_arm(self):
        """Select an arm using EXP3 probabilities."""
        probs = self._get_probabilities()
        r = self.rng.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                self.current_arm_idx = i
                self.current_rate = self.arms[i]
                return
        # Fallback (numerical edge case)
        self.current_arm_idx = self.K - 1
        self.current_rate = self.arms[-1]

    def get_current_rate(self) -> float:
        """Return current replay rate."""
        return self.current_rate

    def step(self) -> Optional[Dict[str, Any]]:
        """
        Called after each gradient step. Evaluates probe and updates
        bandit at probe_interval boundaries.

        Returns:
            Dict with probe results if evaluation was performed, else None.
        """
        self.step_counter += 1

        if self.step_counter % self.probe_interval == 0:
            probe_loss = evaluate_probe(
                self.model, self.probe_set, self.device
            )

            # Reward: positive = retention improved (loss decreased)
            reward = self.last_probe_loss - probe_loss

            # Update bandit weights (EXP3)
            probs = self._get_probabilities()
            p_chosen = probs[self.current_arm_idx]

            # Clip reward to [-1, 1] for stability
            reward_clipped = max(-1.0, min(1.0, reward))
            r_hat = reward_clipped / p_chosen

            # Update weight for chosen arm
            self.weights[self.current_arm_idx] *= math.exp(
                self.exp3_eta * r_hat / self.K
            )

            # Record history
            self.arm_history.append(self.current_arm_idx)
            self.reward_history.append(reward)
            self.probe_loss_history.append(probe_loss)
            self.last_probe_loss = probe_loss

            # Select new arm for next interval
            self._select_arm()

            logger.debug(
                f"Bandit step {self.step_counter}: reward={reward:.4f}, "
                f"new_rate={self.current_rate}, probe_loss={probe_loss:.4f}"
            )

            return {
                "probe_loss": probe_loss,
                "reward": reward,
                "arm_idx": self.current_arm_idx,
                "rate": self.current_rate
            }

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Return full history and summary statistics for metrics.json."""
        arm_rates = [self.arms[i] for i in self.arm_history]

        if arm_rates:
            mean_rate = sum(arm_rates) / len(arm_rates)
            rate_std = (
                sum((r - mean_rate) ** 2 for r in arm_rates) / len(arm_rates)
            ) ** 0.5
        else:
            mean_rate = self.current_rate
            rate_std = 0.0

        probs = self._get_probabilities()

        return {
            "bandit_arm_history": self.arm_history,
            "bandit_reward_history": self.reward_history,
            "bandit_arm_weights_final": self.weights,
            "bandit_arm_probs_final": probs,
            "mean_replay_rate": mean_rate,
            "replay_rate_std": rate_std,
            "probe_loss_trajectory": self.probe_loss_history,
            "probe_set_hash": self.probe_hash,
            "n_evaluations": len(self.arm_history),
            "arms": self.arms,
        }


# =============================================================================
# RMGS (Reward-Modulated Gradient Scaling)
# =============================================================================

class RMGS:
    """
    Reward-Modulated Gradient Scaling.

    Monitors Domain A retention via a probe set during Domain B training.
    When forgetting is detected (probe loss increasing), the effective
    learning rate is scaled down proportionally. When retention is stable
    or improving, the full learning rate is maintained.

    Per RL_PHASE_PLAN.md Section 2.2 and Appendix A.2.

    Args:
        model: The neural network model
        device: Device string
        probe_set: Probe tensor on CPU, shape (probe_size, seq_len)
        probe_hash: SHA256 hash of probe indices
        probe_interval: Gradient steps between probe evaluations
        ema_alpha: Exponential moving average smoothing for reward
        beta: Sensitivity of scaling to reward signal
        min_scale: Minimum gradient scale (never fully stops learning)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        probe_set: torch.Tensor,
        probe_hash: str,
        probe_interval: int,
        ema_alpha: float = 0.3,
        beta: float = 2.0,
        min_scale: float = 0.05,
    ):
        self.model = model
        self.device = device
        self.probe_set = probe_set
        self.probe_hash = probe_hash
        self.probe_interval = probe_interval
        self.ema_alpha = ema_alpha
        self.beta = beta
        self.min_scale = min_scale

        # State
        self.last_probe_loss: Optional[float] = None
        self.ema_reward = 0.0
        self.current_scale = 1.0

        # History
        self.scale_history: List[float] = []
        self.reward_history: List[float] = []
        self.ema_reward_history: List[float] = []
        self.probe_loss_history: List[float] = []

        # Step counter
        self.step_counter = 0

    def initialize(self):
        """Evaluate initial probe loss."""
        self.last_probe_loss = evaluate_probe(
            self.model, self.probe_set, self.device
        )
        self.probe_loss_history.append(self.last_probe_loss)
        logger.info(f"RMGS initialized: probe_loss={self.last_probe_loss:.4f}")

    def step(self) -> Optional[Dict[str, float]]:
        """
        Called after each gradient step. Evaluates probe and updates
        gradient scale at probe_interval boundaries.

        Returns:
            Dict with evaluation results if performed, else None.
        """
        self.step_counter += 1

        if self.step_counter % self.probe_interval == 0:
            probe_loss = evaluate_probe(
                self.model, self.probe_set, self.device
            )

            # Reward: positive = retention improved (loss decreased)
            reward = self.last_probe_loss - probe_loss

            # Update EMA
            self.ema_reward = (
                self.ema_alpha * reward
                + (1 - self.ema_alpha) * self.ema_reward
            )

            # Compute scale: clamp(1.0 + beta * R, min_scale, 1.0)
            self.current_scale = max(
                self.min_scale,
                min(1.0, 1.0 + self.beta * self.ema_reward)
            )

            # Record history
            self.scale_history.append(self.current_scale)
            self.reward_history.append(reward)
            self.ema_reward_history.append(self.ema_reward)
            self.probe_loss_history.append(probe_loss)
            self.last_probe_loss = probe_loss

            logger.debug(
                f"RMGS step {self.step_counter}: reward={reward:.4f}, "
                f"ema={self.ema_reward:.4f}, scale={self.current_scale:.4f}"
            )

            return {
                "probe_loss": probe_loss,
                "reward": reward,
                "ema_reward": self.ema_reward,
                "scale": self.current_scale,
            }

        return None

    def get_scale(self) -> float:
        """Return current gradient scaling factor."""
        return self.current_scale

    def get_stats(self) -> Dict[str, Any]:
        """Return full history and summary statistics for metrics.json."""
        if self.scale_history:
            mean_scale = sum(self.scale_history) / len(self.scale_history)
            scale_std = (
                sum((s - mean_scale) ** 2 for s in self.scale_history)
                / len(self.scale_history)
            ) ** 0.5
        else:
            mean_scale = 1.0
            scale_std = 0.0

        return {
            "rmgs_scale_history": self.scale_history,
            "rmgs_reward_history": self.reward_history,
            "rmgs_ema_reward_history": self.ema_reward_history,
            "rmgs_probe_loss_trajectory": self.probe_loss_history,
            "mean_gradient_scale": mean_scale,
            "scale_std": scale_std,
            "probe_set_hash": self.probe_hash,
            "n_evaluations": len(self.scale_history),
        }
