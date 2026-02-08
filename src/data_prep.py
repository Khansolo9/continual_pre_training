#!/usr/bin/env python3
"""
Data Preparation Script for Continual Pretraining Experiments

Prepares:
- Domain A: WikiText-103 (A-medium = 10M tokens)
- Domain B: ArXiv abstracts (B-medium = 10M tokens)
- Evaluation sets: wikitext103_valid, arxiv_valid, lambada_test

Outputs:
- Tokenized datasets in data/processed/
- Evaluation sets in data/eval/
- Manifest files with hashes in data/manifests/
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try imports - provide helpful error if missing
try:
    import torch
    from transformers import GPT2Tokenizer
    from datasets import load_dataset, Dataset
    import numpy as np
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    logger.error("Install with: pip install torch transformers datasets numpy")
    raise


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def compute_hash(data: bytes) -> str:
    """Compute SHA256 hash of data."""
    return hashlib.sha256(data).hexdigest()[:16]


def compute_dataset_hash(texts: list) -> str:
    """Compute hash of dataset content."""
    combined = "\n".join(texts[:1000])  # Hash first 1000 entries for speed
    return compute_hash(combined.encode('utf-8'))


class DataPreparer:
    """Prepares datasets for continual pretraining experiments."""

    def __init__(self, project_root: Path, token_budget_a: int = 10_000_000,
                 token_budget_b: int = 10_000_000):
        self.project_root = project_root
        self.token_budget_a = token_budget_a
        self.token_budget_b = token_budget_b

        # Initialize tokenizer
        logger.info("Loading GPT-2 tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Paths
        self.data_dir = project_root / "data"
        self.processed_dir = self.data_dir / "processed"
        self.eval_dir = self.data_dir / "eval"
        self.manifest_dir = self.data_dir / "manifests"

        # Create directories
        for d in [self.processed_dir, self.eval_dir, self.manifest_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def prepare_wikitext103(self) -> Dict[str, Any]:
        """
        Prepare WikiText-103 (Domain A).

        Returns manifest with dataset info and hash.
        """
        logger.info("Loading WikiText-103...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

        # Process training data
        train_texts = [t for t in dataset['train']['text'] if t.strip()]
        logger.info(f"WikiText-103 train: {len(train_texts)} non-empty texts")

        # Tokenize and count tokens
        logger.info("Tokenizing WikiText-103 train split...")
        all_tokens = []
        total_tokens = 0

        for text in train_texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            total_tokens += len(tokens)
            if total_tokens >= self.token_budget_a:
                break

        # Trim to exact budget
        all_tokens = all_tokens[:self.token_budget_a]
        logger.info(f"Domain A tokens: {len(all_tokens):,} (budget: {self.token_budget_a:,})")

        # Save tokenized training data
        train_path = self.processed_dir / "wikitext103_train_tokens.pt"
        torch.save(torch.tensor(all_tokens, dtype=torch.long), train_path)

        # Process validation data
        valid_texts = [t for t in dataset['validation']['text'] if t.strip()]
        valid_tokens = []
        for text in valid_texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            valid_tokens.extend(tokens)

        # Save validation set
        valid_path = self.eval_dir / "wikitext103_valid_tokens.pt"
        torch.save(torch.tensor(valid_tokens, dtype=torch.long), valid_path)
        logger.info(f"WikiText-103 valid: {len(valid_tokens):,} tokens")

        # Compute hash (convert tokens to string since token IDs exceed byte range)
        dataset_hash = compute_hash(str(all_tokens[:10000]).encode('utf-8'))

        # Create manifest
        manifest = {
            "name": "wikitext-103",
            "tier": "A-medium",
            "tokens_used": len(all_tokens),
            "tokens_budget": self.token_budget_a,
            "valid_tokens": len(valid_tokens),
            "hash": f"sha256:{dataset_hash}",
            "train_path": str(train_path.relative_to(self.project_root)),
            "valid_path": str(valid_path.relative_to(self.project_root)),
            "created": datetime.now().isoformat(),
            "source": "huggingface:wikitext/wikitext-103-raw-v1"
        }

        manifest_path = self.manifest_dir / "domain_a.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Domain A manifest saved: {manifest_path}")

        return manifest

    def prepare_arxiv(self) -> Dict[str, Any]:
        """
        Prepare ArXiv abstracts (Domain B).

        Uses the arxiv_dataset from HuggingFace or falls back to a smaller subset.
        """
        logger.info("Loading ArXiv abstracts...")

        try:
            # Try loading from HuggingFace
            dataset = load_dataset("ccdv/arxiv-summarization", split="train")
            texts = [item['abstract'] for item in dataset if item.get('abstract')]
        except Exception as e:
            logger.warning(f"Could not load arxiv-summarization: {e}")
            logger.info("Falling back to scientific_papers dataset...")
            try:
                dataset = load_dataset("scientific_papers", "arxiv", split="train")
                texts = [item['abstract'] for item in dataset if item.get('abstract')]
            except Exception as e2:
                logger.warning(f"Could not load scientific_papers: {e2}")
                logger.info("Creating synthetic ArXiv-style data for pilot testing...")
                # Fallback: create synthetic scientific text for pilot
                texts = self._create_synthetic_arxiv(self.token_budget_b)

        logger.info(f"ArXiv: {len(texts)} abstracts loaded")

        # === Deterministic train/valid split BEFORE tokenizing ===
        # This ensures validation texts are NEVER seen during training
        n_valid = max(100, len(texts) // 20)  # 5% for validation
        valid_texts = texts[-n_valid:]
        train_texts = texts[:-n_valid]  # Explicit exclusion
        logger.info(f"ArXiv split: {len(train_texts)} train, {len(valid_texts)} valid abstracts")

        # Tokenize TRAINING data only
        logger.info("Tokenizing ArXiv training abstracts...")
        all_tokens = []
        total_tokens = 0

        for text in train_texts:  # Use train_texts, not texts
            if not text or not text.strip():
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            total_tokens += len(tokens)
            if total_tokens >= self.token_budget_b:
                break

        # Trim to exact budget
        all_tokens = all_tokens[:self.token_budget_b]
        logger.info(f"Domain B tokens: {len(all_tokens):,} (budget: {self.token_budget_b:,})")

        # Save tokenized training data
        train_path = self.processed_dir / "arxiv_train_tokens.pt"
        torch.save(torch.tensor(all_tokens, dtype=torch.long), train_path)

        # Tokenize VALIDATION data (from held-out valid_texts)
        valid_tokens = []
        for text in valid_texts:
            if text and text.strip():
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                valid_tokens.extend(tokens)

        # Limit validation size
        valid_tokens = valid_tokens[:500000]  # ~500K tokens max for validation

        # Save validation set
        valid_path = self.eval_dir / "arxiv_valid_tokens.pt"
        torch.save(torch.tensor(valid_tokens, dtype=torch.long), valid_path)
        logger.info(f"ArXiv valid: {len(valid_tokens):,} tokens")

        # Compute hash (convert tokens to string since token IDs exceed byte range)
        dataset_hash = compute_hash(str(all_tokens[:10000]).encode('utf-8'))

        # Create manifest
        manifest = {
            "name": "arxiv_abstracts",
            "tier": "B-medium",
            "tokens_used": len(all_tokens),
            "tokens_budget": self.token_budget_b,
            "valid_tokens": len(valid_tokens),
            "hash": f"sha256:{dataset_hash}",
            "train_path": str(train_path.relative_to(self.project_root)),
            "valid_path": str(valid_path.relative_to(self.project_root)),
            "created": datetime.now().isoformat(),
            "source": "huggingface:ccdv/arxiv-summarization"
        }

        manifest_path = self.manifest_dir / "domain_b.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Domain B manifest saved: {manifest_path}")

        return manifest

    def _create_synthetic_arxiv(self, token_budget: int) -> list:
        """Create synthetic scientific text for testing when real data unavailable."""
        logger.warning("Using synthetic data - for testing only!")

        templates = [
            "We propose a novel method for {topic} that achieves state-of-the-art results. Our approach combines {method1} with {method2} to improve performance. Experiments on {dataset} demonstrate significant improvements over baseline methods.",
            "This paper presents an analysis of {topic} in the context of {field}. We show that {finding} under certain conditions. Our theoretical framework provides new insights into {aspect}.",
            "In this work, we investigate the problem of {topic}. Previous approaches have focused on {old_approach}, but we demonstrate that {new_approach} leads to better outcomes. We validate our method on multiple benchmarks.",
        ]

        topics = ["neural networks", "optimization", "language models", "representation learning",
                  "generalization", "deep learning", "transformer architectures", "attention mechanisms"]
        methods = ["gradient descent", "backpropagation", "regularization", "dropout",
                   "batch normalization", "layer normalization", "residual connections"]
        datasets = ["ImageNet", "CIFAR-10", "WikiText", "GLUE", "SQuAD", "MNIST"]
        fields = ["computer vision", "natural language processing", "reinforcement learning",
                  "unsupervised learning", "semi-supervised learning"]

        import random
        random.seed(42)

        texts = []
        while True:
            template = random.choice(templates)
            text = template.format(
                topic=random.choice(topics),
                method1=random.choice(methods),
                method2=random.choice(methods),
                dataset=random.choice(datasets),
                field=random.choice(fields),
                finding="performance improves significantly",
                old_approach=random.choice(methods),
                new_approach=random.choice(methods),
                aspect=random.choice(topics)
            )
            texts.append(text)

            # Check if we have enough
            if len(texts) >= 50000:
                break

        return texts

    def prepare_lambada(self) -> Dict[str, Any]:
        """
        Prepare LAMBADA evaluation set.

        LAMBADA tests long-range context understanding via word prediction.
        """
        logger.info("Loading LAMBADA dataset...")

        lambada_path = self.eval_dir / "lambada_test.json"

        try:
            dataset = load_dataset("lambada", split="test")

            # Format as context + target pairs
            examples = []
            for item in dataset:
                text = item['text']
                # Last word is the target
                words = text.rsplit(' ', 1)
                if len(words) == 2:
                    examples.append({
                        "context": words[0],
                        "target": words[1]
                    })

            logger.info(f"LAMBADA: {len(examples)} examples")

            with open(lambada_path, 'w') as f:
                json.dump(examples, f, indent=2)

            return {"path": str(lambada_path), "count": len(examples)}

        except Exception as e:
            logger.warning(f"Could not load LAMBADA: {e}")
            logger.info("Creating minimal LAMBADA substitute for testing...")

            # Create minimal test set
            examples = [
                {"context": "The sun was setting over the", "target": "horizon"},
                {"context": "She picked up her phone and started to", "target": "type"},
                {"context": "The experiment was a complete", "target": "success"},
            ] * 100  # Repeat to get reasonable size

            with open(lambada_path, 'w') as f:
                json.dump(examples, f, indent=2)

            return {"path": str(lambada_path), "count": len(examples)}

    def run(self) -> Dict[str, Any]:
        """Run full data preparation pipeline."""
        logger.info("=" * 60)
        logger.info("Starting data preparation...")
        logger.info("=" * 60)

        results = {}

        # Prepare Domain A
        logger.info("\n[1/3] Preparing Domain A (WikiText-103)...")
        results['domain_a'] = self.prepare_wikitext103()

        # Prepare Domain B
        logger.info("\n[2/3] Preparing Domain B (ArXiv)...")
        results['domain_b'] = self.prepare_arxiv()

        # Prepare LAMBADA
        logger.info("\n[3/3] Preparing LAMBADA evaluation set...")
        results['lambada'] = self.prepare_lambada()

        logger.info("\n" + "=" * 60)
        logger.info("Data preparation complete!")
        logger.info("=" * 60)

        # Summary
        logger.info("\nSummary:")
        logger.info(f"  Domain A: {results['domain_a']['tokens_used']:,} tokens")
        logger.info(f"  Domain B: {results['domain_b']['tokens_used']:,} tokens")
        logger.info(f"  LAMBADA:  {results['lambada']['count']} examples")

        return results


def main():
    parser = argparse.ArgumentParser(description="Prepare data for continual pretraining")
    parser.add_argument("--token-budget-a", type=int, default=10_000_000,
                        help="Token budget for Domain A (default: 10M)")
    parser.add_argument("--token-budget-b", type=int, default=10_000_000,
                        help="Token budget for Domain B (default: 10M)")
    parser.add_argument("--project-root", type=str, default=None,
                        help="Project root directory")
    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else get_project_root()

    preparer = DataPreparer(
        project_root=project_root,
        token_budget_a=args.token_budget_a,
        token_budget_b=args.token_budget_b
    )

    preparer.run()


if __name__ == "__main__":
    main()
