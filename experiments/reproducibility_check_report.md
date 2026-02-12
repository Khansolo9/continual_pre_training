# Reproducibility Check Report

**Generated**: 2026-02-07T20:30:00Z
**Auditor**: Claude Code (automated)

---

## Summary

| Check | Status | Notes |
|-------|--------|-------|
| Summary generation smoke exclusion | **FIXED** | Added `--include-smoke` flag, excluded by default |
| Summary statistics accuracy | **FIXED** | Baseline forgetting now 87.00% ± 0.34% (was 64.88% ± 44.23%) |
| Leanfix artifacts | **CLEAN** | No `_leanfix` directories found; only historical documentation references |
| Runpack RQ consistency | **FIXED** | 5 runpacks corrected to match registry |
| Orphaned directories | **NOTED** | `test_preflight` contains only config.yaml |

---

## Task A: Summary Generation Smoke Exclusion

### Problem
The `summary_pack.md` and `summary_table.csv` included smoke test runs in scientific aggregates, causing misleading statistics:
- Baseline forgetting showed 64.88% ± 44.23% (n=4)
- This was due to smoke runs having ~-1.5% forgetting (negative = improvement)

### Fix Applied
Modified [generate_summary_pack.py](src/adhoc/generate_summary_pack.py):

1. Added `is_smoke_run()` function to detect smoke runs by:
   - `research_question == "SMOKE"`
   - `run_id` starts with `smoke_`

2. Added `--include-smoke` CLI flag (smoke excluded by default)

3. Regenerated summaries with correct filtering

### Verification
```
$ python3 src/adhoc/generate_summary_pack.py --write

Loaded 6 completed runs with valid metrics
(Excluded 4 smoke test runs)
```

**Before**: Baseline Forgetting: 64.88% ± 44.23% (n=4)
**After**: Baseline Forgetting: 87.00% ± 0.34% (n=3)

---

## Task B: Leanfix Artifacts

### Search Results
- No `*leanfix*` or `*_lean*` files/directories found
- No `gpt2_pilot_baseline_s0_leanfix` directory exists in `experiments/runs/`

### Historical References Found (documentation only)
The following files contain historical references to `leanfix`:
- `docs/PROJECT_STATUS.md` - Documents that `gpt2_pilot_baseline_s0_leanfix` was an experimental run that was reverted
- `experiments/diagnostics/why_results_not_as_expected.md` - Historical analysis
- `src/adhoc/inspect_run.py` - Example in docstring
- `src/adhoc/recover_from_checkpoints.py` - Example in docstring
- `src/adhoc/recover_outputs.py` - Example in docstring

These are appropriate historical documentation and do not require cleanup.

### Orphaned Directory Noted
`experiments/runs/test_preflight/` exists with only `config.yaml` - documented as "orphaned" in PROJECT_STATUS.md

---

## Task C: Runpack Consistency Audit

### Inconsistencies Found and Fixed

| Run ID | Runpack Had | Registry Has | Status |
|--------|-------------|--------------|--------|
| gpt2_rq2_ewc_s42 | RQ1 | RQ2 | **FIXED** |
| gpt2_smoke_baseline_s1 | RQ1 | SMOKE | **FIXED** |
| gpt2_smoke_replay25_s1 | RQ1 | SMOKE | **FIXED** |
| gpt2_smoke_ewc_s1 | RQ1 | SMOKE | **FIXED** |
| gpt2_smoke_mer25_s1 | RQ1 | SMOKE | **FIXED** |

### Consistent Runpacks (no changes needed)
- `gpt2_pilot_baseline_s0` (RQ0)
- `gpt2_rq1_baseline_s42` (RQ1)
- `gpt2_rq1_baseline_s123` (RQ1)
- `gpt2_rq2_replay25_s42` (RQ2)
- `gpt2_rq2_mer25_s42` (RQ2)

---

## Files Modified

| File | Change |
|------|--------|
| `src/adhoc/generate_summary_pack.py` | Added smoke exclusion logic and `--include-smoke` flag |
| `experiments/summary_pack.md` | Regenerated with smoke runs excluded |
| `experiments/summary_table.csv` | Regenerated with smoke runs excluded |
| `experiments/runs/gpt2_rq2_ewc_s42/runpack_gpt2_rq2_ewc_s42.md` | Fixed RQ: RQ1 → RQ2 |
| `experiments/runs/gpt2_smoke_baseline_s1/runpack_gpt2_smoke_baseline_s1.md` | Fixed RQ: RQ1 → SMOKE |
| `experiments/runs/gpt2_smoke_replay25_s1/runpack_gpt2_smoke_replay25_s1.md` | Fixed RQ: RQ1 → SMOKE |
| `experiments/runs/gpt2_smoke_ewc_s1/runpack_gpt2_smoke_ewc_s1.md` | Fixed RQ: RQ1 → SMOKE |
| `experiments/runs/gpt2_smoke_mer25_s1/runpack_gpt2_smoke_mer25_s1.md` | Fixed RQ: RQ1 → SMOKE |

---

## EWC Fix Verification

### Code Changes (from previous session)
- `src/cl_methods.py`: Fixed Fisher computation to use sum reduction instead of mean reduction
- Added `get_fisher_stats()` helper method for diagnostics

### Test Suite
- `tests/test_ewc.py`: 10 EWC-specific tests covering Fisher magnitude, penalty behavior, scaling fix

### Diagnostic Script
- `src/adhoc/_diagnostics/ewc_smoke_check.py`: Quick verification script

### Results
EWC forgetting reduced from 68.88% (before fix) to 33.44% (after fix)

---

## Current State Summary

| Metric | Value |
|--------|-------|
| Total full runs | 6 |
| Total smoke runs | 4 |
| Baseline runs (full) | 3 |
| Baseline forgetting (mean) | 87.00% |
| Baseline forgetting (std) | 0.34% |
| Best method (forgetting) | mer25 (4.47%) |
| Second best method | replay25 (7.48%) |
| EWC forgetting | 33.44% |

---

## Recommendations

1. **Run pytest for full test coverage** when pytest is available in the environment
2. **Consider removing** `test_preflight` orphaned directory or documenting its purpose
3. **Add smoke exclusion** as default behavior to any future summary/analysis scripts

---

**Report complete.**
