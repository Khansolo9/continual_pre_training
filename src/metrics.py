#!/usr/bin/env python3
"""
Metrics Module for Continual Pretraining Experiments

Implements metrics per docs/METRICS_CATALOG.md:
- PPL (primary: token-weighted NLL; secondary: median-batch)
- Forgetting percentage
- Rep-n (repetition metrics)
- Drift metrics (JS divergence)
- Vocab overlap
- LAMBADA accuracy

PPL aggregation policy (locked for pilot):
- Primary PPL: token-weighted NLL over entire eval set, then exp()
- Secondary: median of batch-level PPL values
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MetricsComputer:
    """Computes all required metrics for continual pretraining experiments."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def compute_ppl(
        self,
        tokens: torch.Tensor,
        batch_size: int = 8,
        sequence_length: int = 512
    ) -> Dict[str, float]:
        """
        Compute perplexity metrics.

        Returns:
            dict with:
                - ppl_primary: token-weighted NLL -> PPL (official metric)
                - ppl_median_batch: median of batch PPLs (diagnostic)
                - total_tokens: number of tokens evaluated
        """
        self.model.eval()

        # Reshape into sequences
        n_tokens = len(tokens)
        n_seqs = n_tokens // sequence_length
        if n_seqs == 0:
            # Handle short sequences
            sequence_length = n_tokens
            n_seqs = 1

        tokens = tokens[:n_seqs * sequence_length].view(n_seqs, sequence_length)

        dataset = TensorDataset(tokens)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_nll = 0.0
        total_tokens = 0
        batch_ppls = []

        for batch in tqdm(loader, desc="Computing PPL", leave=False):
            input_ids = batch[0].to(self.device)

            # Forward pass
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss  # Cross-entropy loss (mean over tokens)

            # Accumulate token-weighted NLL
            batch_tokens = input_ids.numel()
            total_nll += loss.item() * batch_tokens
            total_tokens += batch_tokens

            # Track batch-level PPL for median diagnostic
            batch_ppl = math.exp(loss.item())
            batch_ppls.append(batch_ppl)

        # Primary PPL: token-weighted mean NLL -> exp
        avg_nll = total_nll / total_tokens
        ppl_primary = math.exp(avg_nll)

        # Secondary: median batch PPL
        batch_ppls_sorted = sorted(batch_ppls)
        n = len(batch_ppls_sorted)
        if n % 2 == 0:
            ppl_median_batch = (batch_ppls_sorted[n//2 - 1] + batch_ppls_sorted[n//2]) / 2
        else:
            ppl_median_batch = batch_ppls_sorted[n//2]

        return {
            "ppl_primary": ppl_primary,
            "ppl_median_batch": ppl_median_batch,
            "total_tokens": total_tokens,
            "avg_nll": avg_nll
        }

    @staticmethod
    def compute_forgetting_pct(ppl_before: float, ppl_after: float) -> float:
        """
        Compute forgetting percentage.

        Formula: Forget% = (PPL_after - PPL_before) / PPL_before * 100
        """
        if ppl_before <= 0:
            return float('nan')
        return ((ppl_after - ppl_before) / ppl_before) * 100

    @torch.no_grad()
    def compute_repetition(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Dict[str, float]:
        """
        Compute repetition metrics (Rep-4, Rep-8).

        Rep-n = fraction of n-grams that are duplicates in generated text.
        """
        self.model.eval()

        all_rep4 = []
        all_rep8 = []

        for prompt in tqdm(prompts, desc="Computing Rep-n", leave=False):
            # Encode prompt
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Generate
            if do_sample:
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            else:
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Get generated tokens (excluding prompt)
            generated = output[0, input_ids.shape[1]:].tolist()

            # Compute Rep-4 and Rep-8
            rep4 = self._compute_rep_n(generated, n=4)
            rep8 = self._compute_rep_n(generated, n=8)

            all_rep4.append(rep4)
            all_rep8.append(rep8)

        # Return median values
        return {
            "rep4": self._median(all_rep4),
            "rep8": self._median(all_rep8),
            "rep4_mean": sum(all_rep4) / len(all_rep4) if all_rep4 else 0.0,
            "rep8_mean": sum(all_rep8) / len(all_rep8) if all_rep8 else 0.0,
            "n_samples": len(prompts)
        }

    @staticmethod
    def _compute_rep_n(tokens: List[int], n: int) -> float:
        """Compute repetition rate for n-grams."""
        if len(tokens) < n:
            return 0.0

        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0

        counter = Counter(ngrams)
        total = len(ngrams)
        duplicates = sum(count - 1 for count in counter.values() if count > 1)

        return duplicates / total if total > 0 else 0.0

    @staticmethod
    def _median(values: List[float]) -> float:
        """Compute median of a list."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 0:
            return (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
        return sorted_vals[n//2]

    @torch.no_grad()
    def compute_drift_metrics(
        self,
        prompts: List[str],
        reference_distributions: Optional[Dict[str, Counter]] = None,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Tuple[Dict[str, float], Dict[str, Counter]]:
        """
        Compute output distribution drift metrics.

        Uses Jensen-Shannon divergence on token frequency distributions.
        Also computes vocab overlap.

        Returns:
            (metrics_dict, current_distributions) where current_distributions
            can be saved as reference for future comparisons.
        """
        self.model.eval()

        # Collect token distributions from generations
        token_counter = Counter()
        all_generated_tokens = set()

        for prompt in tqdm(prompts, desc="Computing drift", leave=False):
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            if do_sample:
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            else:
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated = output[0, input_ids.shape[1]:].tolist()
            token_counter.update(generated)
            all_generated_tokens.update(generated)

        current_distributions = {
            "token_counts": token_counter,
            "unique_tokens": all_generated_tokens
        }

        metrics = {
            "n_unique_tokens": len(all_generated_tokens),
            "n_total_tokens": sum(token_counter.values())
        }

        # If we have reference distributions, compute drift
        if reference_distributions is not None:
            ref_counter = reference_distributions.get("token_counts", Counter())
            ref_tokens = reference_distributions.get("unique_tokens", set())

            # JS divergence
            js_div = self._compute_js_divergence(ref_counter, token_counter)
            metrics["js_divergence"] = js_div

            # Vocab overlap
            if ref_tokens and all_generated_tokens:
                overlap = len(ref_tokens & all_generated_tokens)
                union = len(ref_tokens | all_generated_tokens)
                metrics["vocab_overlap"] = overlap / union if union > 0 else 0.0
            else:
                metrics["vocab_overlap"] = 1.0 if not ref_tokens else 0.0

        return metrics, current_distributions

    @staticmethod
    def _compute_js_divergence(counter1: Counter, counter2: Counter) -> float:
        """Compute Jensen-Shannon divergence between two token distributions."""
        # Get all tokens
        all_tokens = set(counter1.keys()) | set(counter2.keys())
        if not all_tokens:
            return 0.0

        # Normalize to probabilities
        total1 = sum(counter1.values()) or 1
        total2 = sum(counter2.values()) or 1

        p = {t: counter1.get(t, 0) / total1 for t in all_tokens}
        q = {t: counter2.get(t, 0) / total2 for t in all_tokens}

        # Compute M = (P + Q) / 2
        m = {t: (p[t] + q[t]) / 2 for t in all_tokens}

        # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        def kl_div(dist, ref):
            total = 0.0
            for t in all_tokens:
                if dist[t] > 0 and ref[t] > 0:
                    total += dist[t] * math.log(dist[t] / ref[t])
            return total

        js = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
        return js

    @torch.no_grad()
    def compute_lambada_accuracy(
        self,
        lambada_path: Path,
        max_examples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute LAMBADA accuracy (zero-shot word prediction).

        The task is to predict the final word given the context.
        """
        self.model.eval()

        with open(lambada_path, 'r') as f:
            examples = json.load(f)

        if max_examples:
            examples = examples[:max_examples]

        correct = 0
        total = 0

        multi_token_targets = 0

        for ex in tqdm(examples, desc="LAMBADA eval", leave=False):
            context = ex["context"]
            target = ex["target"].strip()

            # Encode context
            enc = self.tokenizer(context, return_tensors="pt", truncation=True)
            input_ids = enc["input_ids"].to(self.device)

            # GPT-2 tokenization: target typically begins with a leading space
            target_ids = self.tokenizer.encode(" " + target, add_special_tokens=False)
            if len(target_ids) == 0:
                continue
            target_first_id = target_ids[0]

            if len(target_ids) > 1:
                multi_token_targets += 1

            # Predict next token
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]
            pred_token_id = int(logits.argmax().item())

            if pred_token_id == target_first_id:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "multi_token_targets": multi_token_targets,
            "multi_token_rate": (multi_token_targets / total) if total > 0 else 0.0
        }


def load_prompts(prompt_file: Path) -> List[str]:
    """Load prompts from JSON file."""
    with open(prompt_file, 'r') as f:
        data = json.load(f)
    return [item["text"] for item in data]
