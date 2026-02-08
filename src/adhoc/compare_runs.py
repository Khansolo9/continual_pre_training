#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYS = [
    ("ppl_a_init", "PPL_A init"),
    ("ppl_a_before", "PPL_A post_A"),
    ("ppl_a_after", "PPL_A final"),
    ("ppl_b_init", "PPL_B init"),
    ("ppl_b_after", "PPL_B final"),
    ("rep4_before", "Rep-4 post_A"),
    ("rep4_after", "Rep-4 final"),
    ("rep8_before", "Rep-8 post_A"),
    ("rep8_after", "Rep-8 final"),
    ("forgetting_pct", "Forgetting %"),
    ("drift_value_before", "Drift JS before"),
    ("drift_value_after", "Drift JS after"),
    ("vocab_overlap_before", "Vocab overlap before"),
    ("vocab_overlap_after", "Vocab overlap after"),
    ("lambada_before", "LAMBADA before"),
    ("lambada_after", "LAMBADA after"),
]


def load_metrics(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {
        "run": d.get("run", {}),
        "inputs": d.get("inputs", {}),
        "metrics": d.get("metrics", {}),
        "resources": d.get("resources", {}),
        "anomalies": d.get("anomalies", []),
        "notes": d.get("notes", ""),
    }


def fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="run id A")
    ap.add_argument("--run-b", required=True, help="run id B")
    ap.add_argument("--project-root", default=".", help="repo root (default: .)")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()

    pa = root / "experiments" / "runs" / args.run_a / "metrics.json"
    pb = root / "experiments" / "runs" / args.run_b / "metrics.json"

    if not pa.exists():
        raise FileNotFoundError(f"Missing {pa}")
    if not pb.exists():
        raise FileNotFoundError(f"Missing {pb}")

    A = load_metrics(pa)
    B = load_metrics(pb)

    print("=" * 100)
    print(f"COMPARE\nA: {args.run_a}\nB: {args.run_b}")
    print("=" * 100)

    print("\n[RUN STATUS]")
    print(f"A status={A['run'].get('status')} method={A['run'].get('method')} seed={A['run'].get('seed')}")
    print(f"B status={B['run'].get('status')} method={B['run'].get('method')} seed={B['run'].get('seed')}")

    print("\n[METRICS DIFF]")
    for k, label in KEYS:
        va = A["metrics"].get(k)
        vb = B["metrics"].get(k)
        print(f"{label:<20}  A={fmt(va):>12}   B={fmt(vb):>12}")

    print("\n[ANOMALIES]")
    print(f"A: {A['anomalies'] or ['None']}")
    print(f"B: {B['anomalies'] or ['None']}")

    print("\n[NOTES]")
    print(f"A: {A['notes']}")
    print(f"B: {B['notes']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
