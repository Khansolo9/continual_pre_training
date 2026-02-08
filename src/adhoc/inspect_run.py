#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def p(v, fmt=None):
    if v is None:
        return "N/A"
    if fmt:
        try:
            return format(v, fmt)
        except Exception:
            return str(v)
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="e.g., pilot_baseline_s0_leanfix")
    ap.add_argument("--project-root", default=".", help="repo root (default: .)")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    run_dir = root / "experiments" / "runs" / args.run_id
    metrics_path = run_dir / "metrics.json"
    runpack_path = run_dir / f"runpack_{args.run_id}.md"
    ckpt_dir = run_dir / "checkpoints"

    print("=" * 80)
    print(f"RUN DIR:      {run_dir}")
    print(f"metrics.json: {metrics_path.exists()} -> {metrics_path}")
    print(f"runpack.md:   {runpack_path.exists()} -> {runpack_path}")
    print(f"checkpoints:  {ckpt_dir.exists()} -> {ckpt_dir}")
    if ckpt_dir.exists():
        ckpts = sorted([p for p in ckpt_dir.glob("*.pt")])
        for c in ckpts:
            print(f"  - {c.name} ({c.stat().st_size/1024**2:.1f} MB)")
    print("=" * 80)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json at {metrics_path}")

    data = json.loads(metrics_path.read_text())
    run = data.get("run", {})
    inp = data.get("inputs", {})
    m = data.get("metrics", {})
    res = data.get("resources", {})
    anomalies = data.get("anomalies", [])

    print("\n[RUN METADATA]")
    print(f"run_id:         {run.get('run_id')}")
    print(f"status:         {run.get('status')}")
    print(f"method:         {run.get('method')}")
    print(f"seed:           {run.get('seed')}")
    print(f"start (utc):    {run.get('timestamp_start')}")
    print(f"end (utc):      {run.get('timestamp_end')}")
    print(f"config_file:    {run.get('config_file')}")
    print(f"runpack_file:   {run.get('runpack_file')}")

    print("\n[DATASET INPUTS]")
    print(f"Domain A: {inp.get('domain_a_name')} (tier={inp.get('domain_a_token_tier')}, tokens={inp.get('domain_a_tokens_used')})")
    print(f"Domain B: {inp.get('domain_b_name')} (tier={inp.get('domain_b_token_tier')}, tokens={inp.get('domain_b_tokens_used')})")

    print("\n[GEN SETTINGS]")
    print(f"quality: do_sample={inp.get('gen_quality_do_sample')} mode={inp.get('gen_quality_decoding_mode')} max_new={inp.get('gen_quality_max_new_tokens')}, temp={inp.get('gen_quality_temperature')}, top_p={inp.get('gen_quality_top_p')}")
    print(f"drift:   do_sample={inp.get('gen_drift_do_sample')} mode={inp.get('gen_drift_decoding_mode')} max_new={inp.get('gen_drift_max_new_tokens')}")

    print("\n[PRIMARY METRICS]")
    print(f"PPL_A init:     {p(m.get('ppl_a_init'), '.2f')}")
    print(f"PPL_A post_A:   {p(m.get('ppl_a_before'), '.2f')}  (median_batch={p(m.get('ppl_a_before_median_batch'), '.2f')})")
    print(f"PPL_A final:    {p(m.get('ppl_a_after'), '.2f')}   (median_batch={p(m.get('ppl_a_after_median_batch'), '.2f')})")
    print(f"PPL_B init:     {p(m.get('ppl_b_init'), '.2f')}")
    print(f"PPL_B final:    {p(m.get('ppl_b_after'), '.2f')}   (median_batch={p(m.get('ppl_b_after_median_batch'), '.2f')})")

    print("\n[FORGETTING / QUALITY]")
    print(f"forgetting_pct: {p(m.get('forgetting_pct'), '.2f')}%")
    print(f"rep4 post_A:    {p(m.get('rep4_before'), '.4f')}")
    print(f"rep4 final:     {p(m.get('rep4_after'), '.4f')}")
    print(f"rep8 post_A:    {p(m.get('rep8_before'), '.4f')}")
    print(f"rep8 final:     {p(m.get('rep8_after'), '.4f')}")

    print("\n[DRIFT]")
    print(f"drift_metric:   {m.get('drift_metric_name')}")
    print(f"drift before:   {p(m.get('drift_value_before'), '.4f')}")
    print(f"drift after:    {p(m.get('drift_value_after'), '.4f')}")
    print(f"vocab before:   {p(m.get('vocab_overlap_before'), '.4f')}")
    print(f"vocab after:    {p(m.get('vocab_overlap_after'), '.4f')}")

    print("\n[GENERAL ABILITY]")
    print(f"lambada before: {p(m.get('lambada_before'), '.4f')}")
    print(f"lambada after:  {p(m.get('lambada_after'), '.4f')}")

    print("\n[RESOURCES]")
    print(f"total_hours:    {p(res.get('total_hours'), '.2f')}")
    print(f"domain_a_hours: {p(res.get('domain_a_hours'), '.2f')}")
    print(f"domain_b_hours: {p(res.get('domain_b_hours'), '.2f')}")
    print(f"peak_vram_gb:   {p(res.get('peak_vram_gb'), '.2f')}")
    print(f"peak_ram_gb:    {p(res.get('peak_ram_gb'), '.2f')}")
    print(f"avg_tok/s:      {p(res.get('avg_tokens_per_sec'), '.0f')}")

    print("\n[ANOMALIES]")
    if anomalies:
        for a in anomalies:
            print(f"- {a}")
    else:
        print("- None")

    print("\n[NOTES]")
    print(data.get("notes", ""))

    print("\nDone.")


if __name__ == "__main__":
    main()
