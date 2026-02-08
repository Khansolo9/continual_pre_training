#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    p = root / "experiments" / "runs" / args.run_id / "metrics.json"
    if not p.exists():
        raise FileNotFoundError(p)

    d = json.loads(p.read_text())
    m = d.get("metrics", {})
    inp = d.get("inputs", {})
    notes = d.get("notes", "")

    print("=" * 80)
    print(f"Run: {args.run_id}")
    print(f"Notes: {notes}")
    print("-" * 80)

    # Drift semantics check
    drift_before = m.get("drift_value_before")
    vocab_before = m.get("vocab_overlap_before")

    print("[DRIFT SEMANTICS]")
    print(f"drift_before={drift_before}  vocab_before={vocab_before}")

    if drift_before == 0.0 and vocab_before == 1.0:
        print("⚠️  This looks like a 'reference baseline set here' case.")
        print("    Meaning: ref_distributions was likely initialized at the 'before' checkpoint.")
        print("    If this is eval-only recovery, you probably skipped init evaluation.")
    else:
        print("✅ drift_before likely represents a real comparison vs an earlier reference distribution.")

    print("\n[GEN SETTINGS]")
    print(f"quality_do_sample={inp.get('gen_quality_do_sample')}  mode={inp.get('gen_quality_decoding_mode')}")
    print(f"drift_do_sample={inp.get('gen_drift_do_sample')}  mode={inp.get('gen_drift_decoding_mode')}")

    print("\nDone.")

if __name__ == "__main__":
    main()
