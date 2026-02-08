#!/usr/bin/env python3
"""
Diagnostic Tests for Continual Learning Method Implementations

This script verifies whether EWC, Replay, and MER are actually implemented
in the training pipeline, and checks for known issues in metrics.

Run with: python src/adhoc/_diagnostics/verify_methods.py
"""

import sys
import json
import re
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def check(name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "✗"
    print(f"  [{symbol}] {name}: {status}")
    if details:
        for line in details.split("\n"):
            print(f"      {line}")
    return passed

# ============================================================
# TEST 1: Check if EWC penalty code exists in trainer.py
# ============================================================
def test_ewc_implementation():
    header("TEST 1: EWC Implementation Check")

    trainer_path = PROJECT_ROOT / "src" / "trainer.py"
    run_exp_path = PROJECT_ROOT / "src" / "run_experiment.py"

    trainer_code = trainer_path.read_text()
    run_exp_code = run_exp_path.read_text()

    # Check for EWC-related patterns
    ewc_patterns = [
        (r'fisher', "Fisher information computation"),
        (r'ewc.*penalty', "EWC penalty term"),
        (r'ewc.*loss', "EWC loss modification"),
        (r'theta.*star|anchor', "Anchor weights storage"),
        (r'quadratic.*penalty', "Quadratic regularization"),
    ]

    found_in_trainer = []
    found_in_runner = []

    for pattern, desc in ewc_patterns:
        if re.search(pattern, trainer_code, re.IGNORECASE):
            found_in_trainer.append(desc)
        if re.search(pattern, run_exp_code, re.IGNORECASE):
            found_in_runner.append(desc)

    # Check if method config is used
    method_used = "self.config.get(\"method\")" in run_exp_code or "config['method']" in run_exp_code
    method_params_used = "method_params" in trainer_code

    print(f"\n  Searching for EWC patterns in src/trainer.py:")
    print(f"    Found EWC-related code: {len(found_in_trainer) > 0}")
    if found_in_trainer:
        for f in found_in_trainer:
            print(f"      - {f}")
    else:
        print(f"      (none found)")

    print(f"\n  Searching for EWC patterns in src/run_experiment.py:")
    print(f"    Found EWC-related code: {len(found_in_runner) > 0}")
    if found_in_runner:
        for f in found_in_runner:
            print(f"      - {f}")
    else:
        print(f"      (none found)")

    print(f"\n  Method branching logic exists: {method_used}")
    print(f"  Method params used in trainer: {method_params_used}")

    # Result
    ewc_implemented = len(found_in_trainer) >= 2  # Need at least fisher + penalty
    return check("EWC implementation", ewc_implemented,
                 "EWC requires Fisher computation and penalty term in loss" if not ewc_implemented else "")

# ============================================================
# TEST 2: Check if Replay buffer code exists
# ============================================================
def test_replay_implementation():
    header("TEST 2: Replay Implementation Check")

    trainer_path = PROJECT_ROOT / "src" / "trainer.py"
    run_exp_path = PROJECT_ROOT / "src" / "run_experiment.py"

    trainer_code = trainer_path.read_text()
    run_exp_code = run_exp_path.read_text()

    # Check for replay-related patterns
    replay_patterns = [
        (r'replay.*buffer', "Replay buffer class/instance"),
        (r'buffer.*sample', "Buffer sampling"),
        (r'reservoir', "Reservoir sampling"),
        (r'mixing.*ratio', "Batch mixing logic"),
        (r'replay.*batch', "Replay batch construction"),
    ]

    found_in_trainer = []
    found_in_runner = []

    for pattern, desc in replay_patterns:
        if re.search(pattern, trainer_code, re.IGNORECASE):
            found_in_trainer.append(desc)
        if re.search(pattern, run_exp_code, re.IGNORECASE):
            found_in_runner.append(desc)

    print(f"\n  Searching for Replay patterns in src/trainer.py:")
    print(f"    Found Replay-related code: {len(found_in_trainer) > 0}")
    if found_in_trainer:
        for f in found_in_trainer:
            print(f"      - {f}")
    else:
        print(f"      (none found)")

    print(f"\n  Searching for Replay patterns in src/run_experiment.py:")
    print(f"    Found Replay-related code: {len(found_in_runner) > 0}")
    if found_in_runner:
        for f in found_in_runner:
            print(f"      - {f}")
    else:
        print(f"      (none found)")

    # Result
    replay_implemented = len(found_in_trainer) >= 2
    return check("Replay implementation", replay_implemented,
                 "Replay requires buffer storage and mixing logic" if not replay_implemented else "")

# ============================================================
# TEST 3: Check if MER/Reptile code exists
# ============================================================
def test_mer_implementation():
    header("TEST 3: MER/Reptile Implementation Check")

    trainer_path = PROJECT_ROOT / "src" / "trainer.py"
    run_exp_path = PROJECT_ROOT / "src" / "run_experiment.py"

    trainer_code = trainer_path.read_text()
    run_exp_code = run_exp_path.read_text()

    # Check for MER/Reptile patterns
    mer_patterns = [
        (r'reptile', "Reptile meta-update"),
        (r'meta.*update', "Meta-learning update"),
        (r'interpolat', "Weight interpolation"),
        (r'theta.*old|theta.*prev', "Previous weights storage"),
        (r'reptile.*epsilon|epsilon.*reptile', "Reptile epsilon"),
    ]

    found_in_trainer = []
    found_in_runner = []

    for pattern, desc in mer_patterns:
        if re.search(pattern, trainer_code, re.IGNORECASE):
            found_in_trainer.append(desc)
        if re.search(pattern, run_exp_code, re.IGNORECASE):
            found_in_runner.append(desc)

    print(f"\n  Searching for MER/Reptile patterns in src/trainer.py:")
    print(f"    Found MER-related code: {len(found_in_trainer) > 0}")
    if found_in_trainer:
        for f in found_in_trainer:
            print(f"      - {f}")
    else:
        print(f"      (none found)")

    print(f"\n  Searching for MER/Reptile patterns in src/run_experiment.py:")
    print(f"    Found MER-related code: {len(found_in_runner) > 0}")
    if found_in_runner:
        for f in found_in_runner:
            print(f"      - {f}")
    else:
        print(f"      (none found)")

    # Result
    mer_implemented = len(found_in_trainer) >= 2
    return check("MER/Reptile implementation", mer_implemented,
                 "MER requires Reptile meta-update every k steps" if not mer_implemented else "")

# ============================================================
# TEST 4: Verify forgetting% calculation
# ============================================================
def test_forgetting_calculation():
    header("TEST 4: Forgetting% Calculation Verification")

    runs_dir = PROJECT_ROOT / "experiments" / "runs"

    results = []
    for run_dir in sorted(runs_dir.iterdir()):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        ppl_before = metrics.get("ppl_a_before")
        ppl_after = metrics.get("ppl_a_after")
        stored_forgetting = metrics.get("forgetting_pct")

        if ppl_before and ppl_after:
            # Recompute
            computed_forgetting = ((ppl_after - ppl_before) / ppl_before) * 100
            diff = abs(computed_forgetting - stored_forgetting) if stored_forgetting else float('inf')

            results.append({
                "run_id": run_dir.name,
                "ppl_before": ppl_before,
                "ppl_after": ppl_after,
                "stored": stored_forgetting,
                "computed": computed_forgetting,
                "match": diff < 0.01
            })

    print(f"\n  Recomputing forgetting% from PPL values:")
    all_match = True
    for r in results:
        status = "✓" if r["match"] else "✗"
        print(f"    [{status}] {r['run_id']}: stored={r['stored']:.4f}%, computed={r['computed']:.4f}%")
        if not r["match"]:
            all_match = False

    # Also check the formula interpretation
    print(f"\n  Formula used: (PPL_after - PPL_before) / PPL_before * 100")
    print(f"  Higher value = MORE forgetting (worse)")

    return check("Forgetting% calculation", all_match,
                 "All stored values match recomputed values" if all_match else "Mismatch detected")

# ============================================================
# TEST 5: Compare LAMBADA across runs
# ============================================================
def test_lambada_consistency():
    header("TEST 5: LAMBADA Evaluation Consistency")

    runs_dir = PROJECT_ROOT / "experiments" / "runs"

    lambada_data = []
    for run_dir in sorted(runs_dir.iterdir()):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            data = json.load(f)

        lambada_before = data.get("metrics", {}).get("lambada_before")
        lambada_after = data.get("metrics", {}).get("lambada_after")

        lambada_data.append({
            "run_id": run_dir.name,
            "before": lambada_before,
            "after": lambada_after,
        })

    print(f"\n  LAMBADA accuracy across runs:")
    for r in lambada_data:
        print(f"    {r['run_id']}: before={r['before']}, after={r['after']}")

    # Check for pilot_baseline_s0 discrepancy
    pilot = next((r for r in lambada_data if r["run_id"] == "pilot_baseline_s0"), None)
    others = [r for r in lambada_data if r["run_id"] != "pilot_baseline_s0"]

    if pilot and others:
        avg_before_others = sum(r["before"] for r in others if r["before"]) / len([r for r in others if r["before"]])
        avg_after_others = sum(r["after"] for r in others if r["after"]) / len([r for r in others if r["after"]])

        print(f"\n  pilot_baseline_s0:")
        print(f"    before={pilot['before']}, after={pilot['after']}")
        print(f"\n  Other runs average:")
        print(f"    before={avg_before_others:.3f}, after={avg_after_others:.3f}")

        pilot_anomaly = abs(pilot["before"] - avg_before_others) > 0.05 or abs(pilot["after"] - avg_after_others) > 0.05

        if pilot_anomaly:
            print(f"\n  ANOMALY DETECTED: pilot_baseline_s0 LAMBADA differs significantly!")
            print(f"    Possible causes:")
            print(f"      - Different code version (pilot ran before others)")
            print(f"      - Different evaluation subset")
            print(f"      - Bug fixed after pilot")

    consistent = not (pilot and others and abs(pilot["after"] - avg_after_others) > 0.05)
    return check("LAMBADA consistency", consistent,
                 "pilot_baseline_s0 shows different values" if not consistent else "")

# ============================================================
# TEST 6: Check config STUB comments
# ============================================================
def test_config_stubs():
    header("TEST 6: Config STUB Verification")

    configs_dir = PROJECT_ROOT / "configs" / "methods"

    stub_configs = []
    for config_path in sorted(configs_dir.glob("*.yaml")):
        content = config_path.read_text()
        if "STUB" in content or "not yet implemented" in content.lower():
            stub_configs.append(config_path.name)
            # Extract the STUB line
            for line in content.split("\n"):
                if "STUB" in line or "not yet implemented" in line.lower():
                    print(f"    {config_path.name}: {line.strip()}")
                    break

    print(f"\n  Configs marked as STUB/not implemented:")
    for cfg in stub_configs:
        print(f"    - {cfg}")

    has_stubs = len(stub_configs) > 0
    return check("STUB configs", has_stubs,
                 f"Found {len(stub_configs)} configs marked as stubs" if has_stubs else "")

# ============================================================
# TEST 7: Summary pack delta sign check
# ============================================================
def test_summary_pack_deltas():
    header("TEST 7: Summary Pack Delta Sign Verification")

    summary_path = PROJECT_ROOT / "experiments" / "summary_pack.md"
    if not summary_path.exists():
        return check("Summary pack exists", False)

    content = summary_path.read_text()

    # Extract executive summary claims
    exec_summary = content.split("## Executive Summary")[1].split("---")[0] if "## Executive Summary" in content else ""

    print(f"\n  Executive Summary claims:")
    for line in exec_summary.split("\n"):
        if "Best Forgetting" in line or "Δ=" in line:
            print(f"    {line.strip()}")

    # Check if "Best Forgetting" claims negative delta when methods are worse
    # From data: baseline mean = 87.00%, mer25 = 87.13% (WORSE, so delta should be POSITIVE)

    # Extract delta table
    delta_section = ""
    if "## Method Deltas vs Baseline Mean" in content:
        delta_section = content.split("## Method Deltas vs Baseline Mean")[1].split("---")[0]

    print(f"\n  Delta table shows:")
    for line in delta_section.split("\n"):
        if "mer25" in line.lower() or "replay25" in line.lower() or "ewc" in line.lower():
            print(f"    {line.strip()}")

    # Check for inconsistency
    exec_claims_negative = "Δ=-" in exec_summary and "Best Forgetting" in exec_summary
    delta_table_shows_positive = "+0.14%" in delta_section or "+0.13%" in delta_section

    if exec_claims_negative and delta_table_shows_positive:
        print(f"\n  INCONSISTENCY DETECTED:")
        print(f"    Executive summary claims negative delta (improvement)")
        print(f"    Delta table shows positive delta (worse than baseline)")
        print(f"    This is a BUG in generate_summary_pack.py")

    consistent = not (exec_claims_negative and delta_table_shows_positive)
    return check("Delta sign consistency", consistent,
                 "Executive summary delta sign inconsistent with delta table" if not consistent else "")

# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC TESTS FOR CONTINUAL LEARNING METHODS")
    print("=" * 60)

    results = []

    results.append(("EWC Implementation", test_ewc_implementation()))
    results.append(("Replay Implementation", test_replay_implementation()))
    results.append(("MER Implementation", test_mer_implementation()))
    results.append(("Forgetting Calculation", test_forgetting_calculation()))
    results.append(("LAMBADA Consistency", test_lambada_consistency()))
    results.append(("Config STUBs", test_config_stubs()))
    results.append(("Summary Pack Deltas", test_summary_pack_deltas()))

    # Summary
    header("SUMMARY")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\n  Tests passed: {passed}/{total}")
    print()

    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"  [{symbol}] {name}: {status}")

    # Key finding
    print("\n" + "-" * 60)
    print("  KEY FINDING:")
    print("-" * 60)

    ewc_impl = results[0][1]
    replay_impl = results[1][1]
    mer_impl = results[2][1]

    if not ewc_impl and not replay_impl and not mer_impl:
        print("""
  CONFIRMED: EWC, Replay, and MER are NOT IMPLEMENTED.

  All method runs (rq2_replay25_s42, rq2_mer25_s42, rq2_ewc_s42)
  executed the SAME baseline code as the baseline runs.

  The method configs contain STUB comments indicating the
  implementations were planned but never created.

  This explains why all forgetting% values are nearly identical
  (~87%) across all methods - they are all running baseline
  sequential fine-tuning.
        """)

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
