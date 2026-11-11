#!/usr/bin/env python3
"""Run the five ICDE preset phases with durable checkpoints.

Phase membership manifests are materialized explicitly, and each phase calls
the live PostgreSQL campaign runner serially. Existing SQL timings are reused
through one master cache; phase status is updated before and after every
subprocess so an interrupted run can resume without repeating completed work.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "benchmarks/stage_a_30.jsonl"
FULL = ROOT / "benchmarks/manifest.jsonl"
STATUS = ROOT / "results/icde_master_campaign/status.json"
MASTER_CACHE = ROOT / "results/icde_sql_measurement_cache_master.jsonl"


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_manifest(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))


def update(status, **fields):
    status.update(fields, updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    report = ROOT / "reports/ICDE_PAPER_RESULTS.md"
    text = report.read_text() if report.exists() else "# ICDE paper results\n"
    marker = "\n## Five-phase master campaign checkpoint\n"
    body = marker + "\n| Phase | Status | Tasks | Output |\n|---|---|---:|---|\n"
    for name, phase in status.get("phases", {}).items():
        body += f"| {name} | {phase.get('status')} | {phase.get('tasks', 'n/a')} | {phase.get('output', phase.get('source', 'n/a'))} |\n"
    if marker in text:
        text = text.split(marker, 1)[0]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text.rstrip() + "\n" + body)


def run_phase(status, phase, records, config, output, *, suffix="10", extra_env=None, reuse_output=None):
    manifest = ROOT / "results/icde_master_campaign/manifests" / f"{phase}.jsonl"
    write_manifest(manifest, records)
    if reuse_output:
        update(status, phases={**status.get("phases", {}), phase: {"status": "completed", "tasks": len(records), "source": str(reuse_output)}})
        return
    phase_state = {"status": "running", "tasks": len(records), "manifest": str(manifest), "output": str(output), "config": str(config)}
    update(status, phases={**status.get("phases", {}), phase: phase_state})
    env = os.environ.copy()
    env["ICDE_DATABASE_SUFFIX"] = suffix
    if extra_env:
        env.update(extra_env)
    command = [sys.executable, "-u", "-m", "scripts.run_icde_paper_campaign",
               "--config", str(config), "--manifest", str(manifest),
               "--output", str(output), "--cache", str(MASTER_CACHE),
               "--budgets", "1,2,4,8,16",
               "--fixture-bindings", str(ROOT / "results/stage_a_sf10_v2_witnesses/postgresql_validated_bindings.json")]
    result = subprocess.run(command, cwd=ROOT, env=env)
    phase_state.update(status="completed" if result.returncode == 0 else "failed", returncode=result.returncode,
                       completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    update(status, phases={**status.get("phases", {}), phase: phase_state})
    if result.returncode:
        raise SystemExit(f"{phase} failed with exit code {result.returncode}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    stage = rows(STAGE)
    full = rows(FULL)
    tpch = [row for row in full if row.get("benchmark") == "tpch"]
    dsb = [row for row in full if row.get("benchmark") == "dsb"]
    calcite = [row for row in full if row.get("benchmark") == "calcite"]
    calcite_zipf_stage = calcite[:30]
    status = json.loads(STATUS.read_text()) if args.resume and STATUS.exists() else {
        "status": "running", "campaign": "icde_master_campaign", "master_cache": str(MASTER_CACHE),
        "phases": {}, "preset_quotas": {"phase1": 30, "phase2": len(tpch),
        "phase3_uniform": len(calcite), "phase3_zipf": len(calcite), "phase3_zipf_stage": len(calcite_zipf_stage), "phase4": len(dsb), "phase5": 30}}
    config10 = ROOT / "configs/stage-a-sf10.json"
    config_full = ROOT / "configs/published-hybrid.json"
    # Phase 1 is the completed live v4 Stage A study; preserve it as the
    # authoritative 30-query Phase 1 artifact and do not duplicate timings.
    run_phase(status, "phase1", stage, config10, ROOT / "results/icde_paper_campaign_v4",
              reuse_output=ROOT / "results/icde_paper_campaign_v4")
    run_phase(status, "phase5_sf50", stage, config_full, ROOT / "results/icde_sf50_cohort", suffix="50")
    run_phase(status, "phase3_calcite_zipf_stage", calcite_zipf_stage, config_full, ROOT / "results/icde_calcite10zipf_stage", suffix="zipf")
    run_phase(status, "phase2_tpch_sf10", tpch, config_full, ROOT / "results/icde_full_suite_tpch10", suffix="10")
    run_phase(status, "phase3_calcite_uniform", calcite, config_full, ROOT / "results/icde_full_suite_calcite10", suffix="10")
    run_phase(status, "phase3_calcite_zipf", calcite, config_full, ROOT / "results/icde_full_suite_calcite10zipf", suffix="zipf")
    run_phase(status, "phase4_dsb_sf10", dsb, config_full, ROOT / "results/icde_full_suite_dsb10", suffix="10")
    update(status, status="completed", completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


if __name__ == "__main__":
    main()
