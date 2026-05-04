#!/usr/bin/env python3
"""Verify three EWC penalty implementations produce equivalent gradients.

Compares element-wise on Gemma3 1B:
  (a) autograd backward through explicit (lambda/2) sum_i F_i (theta - theta*)^2
  (b) analytic gradient via per-named-parameter Python loop (the path used
      in gemma3_rq2_ewc_s42, the recent successful EWC run)
  (c) analytic gradient via torch._foreach_* fused ops (current branch)

Math equivalence: for diagonal-Fisher EWC, all three compute the same
gradient. Any element-wise discrepancy beyond bf16 ulp tolerance (~1e-3
relative) indicates an implementation bug, not just rounding.

This isolates the EWC penalty contribution: no CE loss, no optimizer steps.
The smoke parity test (gemma3_smoke_ewc) is too noisy on 12 training steps
to tell math drift from sampling chaos; this test goes direct to the math.

Usage:
    python src/adhoc/verify_ewc_gradient_equivalence.py
"""
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / "src"))


def make_test_state(model):
    """Build theta_star (cloned params) and a synthetic varied Fisher."""
    theta_star = {}
    fisher = {}
    torch.manual_seed(42)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        theta_star[name] = p.clone().detach()
        # Positive Fisher with varied per-element magnitude — sensitive to
        # any per-element handling error.
        fisher[name] = torch.abs(torch.randn_like(p)) * 0.01 + 1e-6
    return theta_star, fisher


def perturb_params(model, scale=0.001):
    """Make theta != theta_star so the gradient is non-zero everywhere."""
    torch.manual_seed(123)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * scale)


def reset_grads(model):
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            else:
                p.grad.zero_()


def snapshot_grads(model):
    """Return {name: grad.clone()} dict on CPU.

    Moving to CPU prevents holding three copies of all gradients on MPS,
    which on a 24 GB unified-memory Mac would OOM with model + 3x grads.
    """
    return {
        name: (
            p.grad.detach().to("cpu").clone() if p.grad is not None else None
        )
        for name, p in model.named_parameters()
    }


def impl_a_autograd(model, theta_star, fisher, ewc_lambda):
    """Autograd backward through the explicit penalty tensor."""
    device = next(model.parameters()).device
    penalty = torch.tensor(0.0, device=device)
    for name, p in model.named_parameters():
        if name not in fisher or name not in theta_star:
            continue
        diff = p - theta_star[name]
        penalty = penalty + (fisher[name] * diff * diff).sum()
    loss = (ewc_lambda / 2.0) * penalty
    loss.backward()


def impl_b_analytic_loop(model, theta_star, fisher, ewc_lambda):
    """Per-parameter analytic gradient (used by gemma3_rq2_ewc_s42)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name not in fisher or name not in theta_star:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            f = fisher[name]
            d = p.detach() - theta_star[name]
            p.grad.add_(f * d, alpha=ewc_lambda)


def impl_c_foreach(model, theta_star, fisher, ewc_lambda):
    """Foreach-fused analytic gradient (current branch)."""
    params, fishers, thetas, grads = [], [], [], []
    for name, p in model.named_parameters():
        if name not in fisher or name not in theta_star:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        params.append(p.detach())
        fishers.append(fisher[name])
        thetas.append(theta_star[name])
        grads.append(p.grad)
    if not params:
        return
    with torch.no_grad():
        diffs = torch._foreach_sub(params, thetas)
        weighted = torch._foreach_mul(fishers, diffs)
        torch._foreach_add_(grads, weighted, alpha=ewc_lambda)


def compare(g1, g2, label):
    """Element-wise comparison; report worst-case absolute and relative diff."""
    max_abs = 0.0
    max_rel = 0.0
    total_params = 0
    worst_param = ""
    for name in g1:
        if g1[name] is None or g2[name] is None:
            continue
        a = g1[name].float()
        b = g2[name].float()
        d = (a - b).abs()
        cur_max_abs = d.max().item()
        if cur_max_abs > max_abs:
            max_abs = cur_max_abs
            worst_param = name
        denom = a.abs()
        mask = denom > 1e-6
        if mask.any():
            rel = (d[mask] / denom[mask]).max().item()
            if rel > max_rel:
                max_rel = rel
        total_params += a.numel()
    print(f"  {label}")
    print(f"    max abs diff:   {max_abs:.4e}    (worst tensor: {worst_param})")
    print(f"    max rel diff:   {max_rel:.4e}")
    print(f"    total elements: {total_params:,}")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading Gemma3 1B on {device} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-pt", torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    print("Building synthetic Fisher and theta_star...")
    theta_star, fisher = make_test_state(model)
    print(f"  named params with Fisher: {len(fisher):,}")
    print(f"  total Fisher elements:    {sum(f.numel() for f in fisher.values()):,}")

    print("Perturbing params so theta - theta_star != 0...")
    perturb_params(model)

    ewc_lambda = 100.0
    print(f"\nRunning three EWC penalty implementations with lambda={ewc_lambda}\n")

    def _free_mps():
        if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    # (a) autograd
    print("[a] autograd backward through penalty tensor...")
    reset_grads(model)
    try:
        impl_a_autograd(model, theta_star, fisher, ewc_lambda)
        grads_a = snapshot_grads(model)  # moves to CPU
    except RuntimeError as e:
        print(f"  autograd OOM or error: {e}")
        grads_a = None
    reset_grads(model)
    _free_mps()

    # (b) analytic loop
    print("[b] analytic per-parameter loop...")
    impl_b_analytic_loop(model, theta_star, fisher, ewc_lambda)
    grads_b = snapshot_grads(model)
    reset_grads(model)
    _free_mps()

    # (c) foreach
    print("[c] foreach-fused analytic...")
    impl_c_foreach(model, theta_star, fisher, ewc_lambda)
    grads_c = snapshot_grads(model)
    reset_grads(model)
    _free_mps()

    print("\nElement-wise comparison:\n")
    if grads_a is not None:
        compare(grads_a, grads_b, "autograd (a)  vs  analytic-loop (b)")
        compare(grads_a, grads_c, "autograd (a)  vs  foreach (c)")
    compare(grads_b, grads_c, "analytic-loop (b)  vs  foreach (c)")

    print("\nVerdict reference:")
    print("  bf16 has ~3-4 decimal digits → ulp ~1e-3 relative.")
    print("  max rel diff < 1e-2  -> implementations are math-equivalent.")
    print("  max rel diff > 1e-1  -> implementation bug; investigate.")


if __name__ == "__main__":
    main()
