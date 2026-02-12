#!/usr/bin/env python3
"""
Invariance & Regression Checks (Section 6.5 of MULTI_MODEL_AUDIT_REPORT.md)

Runs the pending invariance tests:
1. PPL sequence_length sensitivity (256 vs 512)
2. Drift determinism (two runs, same checkpoint, same seed)
3. Rep-n seed sensitivity (two runs, record variance)

Uses the GPT-2 smoke baseline checkpoint (theta_A) as the test subject.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import MetricsComputer, load_prompts

PROJECT_ROOT = Path(__file__).parent.parent


def load_gpt2_checkpoint():
    """Load GPT-2 smoke baseline theta_A checkpoint."""
    ckpt_path = PROJECT_ROOT / "experiments" / "runs" / "gpt2_smoke_baseline_s1" / "checkpoints" / "theta_A.pt"
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def test_ppl_sequence_length_sensitivity():
    """PPL with seq_len=256 vs seq_len=512 on the same checkpoint."""
    print("\n" + "=" * 60)
    print("TEST: PPL Sequence Length Sensitivity")
    print("=" * 60)

    model, tokenizer, device = load_gpt2_checkpoint()
    mc = MetricsComputer(model=model, tokenizer=tokenizer, device=device)

    valid_tokens = torch.load(
        PROJECT_ROOT / "data" / "eval" / "wikitext103_valid_tokens.pt",
        weights_only=True
    )[:16384]  # Use enough tokens for meaningful PPL

    result_256 = mc.compute_ppl(valid_tokens, batch_size=1, sequence_length=256)
    result_512 = mc.compute_ppl(valid_tokens, batch_size=1, sequence_length=512)

    ppl_256 = result_256["ppl_primary"]
    ppl_512 = result_512["ppl_primary"]
    tokens_256 = result_256["total_tokens"]
    tokens_512 = result_512["total_tokens"]

    rel_diff = abs(ppl_256 - ppl_512) / ppl_512 * 100

    print(f"  seq_len=256: PPL={ppl_256:.4f} (tokens={tokens_256})")
    print(f"  seq_len=512: PPL={ppl_512:.4f} (tokens={tokens_512})")
    print(f"  Relative difference: {rel_diff:.2f}%")
    print(f"  Token count diff: {tokens_256 - tokens_512} (expected: non-zero due to reshaping)")

    # Expected: small difference (<5%) — longer context generally helps
    status = "PASS" if rel_diff < 5.0 else "REVIEW (>5% difference)"
    print(f"  Status: {status}")

    del model
    if device == "mps":
        torch.mps.empty_cache()

    return {"ppl_256": ppl_256, "ppl_512": ppl_512, "rel_diff_pct": rel_diff, "status": status}


def test_drift_determinism():
    """Run drift evaluation twice on the same checkpoint. Should be identical."""
    print("\n" + "=" * 60)
    print("TEST: Drift Determinism")
    print("=" * 60)

    model, tokenizer, device = load_gpt2_checkpoint()
    mc = MetricsComputer(model=model, tokenizer=tokenizer, device=device)

    drift_prompts = load_prompts(PROJECT_ROOT / "data" / "eval" / "prompts_drift_v1.json")
    # Use a small subset for speed
    drift_prompts = drift_prompts[:20]

    # Run 1
    result1, dist1 = mc.compute_drift_metrics(
        drift_prompts, reference_distributions=None,
        max_new_tokens=64, do_sample=False
    )
    # Run 2 (same checkpoint, same prompts, deterministic decoding)
    result2, dist2 = mc.compute_drift_metrics(
        drift_prompts, reference_distributions=dist1,
        max_new_tokens=64, do_sample=False
    )
    # Run 3 — compare against dist1 again (should give JS=0)
    result3, dist3 = mc.compute_drift_metrics(
        drift_prompts, reference_distributions=dist1,
        max_new_tokens=64, do_sample=False
    )

    js_self = result3.get("js_divergence", -1)

    print(f"  Run 1 vocab size: {len(dist1.get('unique_tokens', set()))}")
    print(f"  Run 2 vs Run 1 JS divergence: {result2.get('js_divergence', -1):.8f}")
    print(f"  Run 3 vs Run 1 JS divergence: {js_self:.8f}")
    print(f"  Vocab overlap (Run 2 vs Run 1): {result2.get('vocab_overlap', -1):.4f}")
    print(f"  Vocab overlap (Run 3 vs Run 1): {result3.get('vocab_overlap', -1):.4f}")

    # JS divergence of a distribution with itself should be 0 (or very near 0)
    status = "PASS" if js_self < 1e-6 else f"FAIL (JS={js_self:.8f})"
    print(f"  Status: {status}")

    del model
    if device == "mps":
        torch.mps.empty_cache()

    return {"js_run2_vs_run1": result2.get("js_divergence", -1), "js_self": js_self, "status": status}


def test_repn_seed_sensitivity():
    """Run rep-n twice with sampling. Record variance band."""
    print("\n" + "=" * 60)
    print("TEST: Rep-n Seed Sensitivity")
    print("=" * 60)

    model, tokenizer, device = load_gpt2_checkpoint()
    mc = MetricsComputer(model=model, tokenizer=tokenizer, device=device)

    quality_prompts = load_prompts(PROJECT_ROOT / "data" / "eval" / "prompts_quality_v1.json")
    quality_prompts = quality_prompts[:20]

    results = []
    for run_idx in range(3):
        torch.manual_seed(42 + run_idx)
        rep_result = mc.compute_repetition(
            quality_prompts,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        results.append(rep_result)
        print(f"  Run {run_idx+1}: rep4={rep_result['rep4']:.4f}, rep8={rep_result['rep8']:.4f}")

    rep4_values = [r["rep4"] for r in results]
    rep8_values = [r["rep8"] for r in results]

    rep4_range = max(rep4_values) - min(rep4_values)
    rep8_range = max(rep8_values) - min(rep8_values)

    print(f"  Rep4 range: {rep4_range:.4f} (values: {[f'{v:.4f}' for v in rep4_values]})")
    print(f"  Rep8 range: {rep8_range:.4f} (values: {[f'{v:.4f}' for v in rep8_values]})")

    # Rep-n with sampling has inherent variance; just record it
    status = "PASS (variance recorded)"
    print(f"  Status: {status}")

    del model
    if device == "mps":
        torch.mps.empty_cache()

    return {
        "rep4_values": rep4_values,
        "rep8_values": rep8_values,
        "rep4_range": rep4_range,
        "rep8_range": rep8_range,
        "status": status
    }


if __name__ == "__main__":
    all_results = {}

    all_results["ppl_seq_len"] = test_ppl_sequence_length_sensitivity()
    all_results["drift_determinism"] = test_drift_determinism()
    all_results["repn_seed"] = test_repn_seed_sensitivity()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, result in all_results.items():
        print(f"  {test_name}: {result['status']}")

    # Save results
    output_path = PROJECT_ROOT / "experiments" / "invariance_check_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
