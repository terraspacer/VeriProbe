#!/usr/bin/env python3
"""Print auditable local generation, service and benchmark progress."""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def main():
    probe = subprocess.run([str(ROOT / ".local/postgres/bin/pg_isready"), "-h", str(ROOT / ".runtime/socket"), "-p", "55432", "-d", "postgres"],
                           capture_output=True, text=True, timeout=5)
    print("PostgreSQL:", probe.stdout.strip() or probe.stderr.strip())
    print("Published datasets:")
    published = read(ROOT / "configs/published.json").get("databases", {})
    for name in published:
        receipt = read(ROOT / ".runtime/datasets" / f"{name}.json")
        generated = (ROOT / ".runtime/data/published" / name / "generation.json").exists()
        state = "ready" if receipt.get("ready") else "loading" if receipt else "generated" if generated else "generating or pending"
        rows = receipt.get("total_rows", sum(table.get("rows", 0) for table in receipt.get("tables", {}).values()))
        print(f"  {name}: {state}; {len(receipt.get('tables', {}))} committed tables; {rows:,} attested rows")
    maintenance = read(ROOT / "benchmarks/postload_maintenance.json")
    if maintenance:
        print("Post-load maintenance:", json.dumps({key: maintenance.get(key) for key in
              ("observed", "status", "ready_for_timing", "settings_restored", "pending_heap_bytes")}))
    print("Session readiness waits:")
    for path in sorted((ROOT / "results").glob("session_*/readiness.json")):
        state = read(path)
        print(f"  {path.parent.name}: {state.get('status', 'unknown')}; "
              f"command_started={state.get('command_started', False)}")
    full_suite = ROOT / "results/tests_latest.xml"
    if not full_suite.exists():
        full_suite = ROOT / "results/tests.xml"
    print("Test receipts (focused rechecks overlap the full suite):")
    for tests in [full_suite, ROOT / "results/sqlsolver_recheck.xml", ROOT / "results/runner_hardening_tests.xml",
                  *sorted((ROOT / "results/validation").glob("*.xml"))]:
        if not tests.exists():
            continue
        tree = ET.parse(tests).getroot()
        suites = [tree] if tree.tag == "testsuite" else list(tree.iter("testsuite"))
        sums = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
        print(f"  {tests.name}:", json.dumps(sums))
    print("Evaluation runs:")
    runs = set((ROOT / "results").glob("*/run.json"))
    runs.update((ROOT / "results").glob("session_*/hybrid/run.json"))
    runs.update((ROOT / "results/published").glob("*/run.json"))
    for path in sorted(runs):
        info = read(path)
        count = 0
        raw = path.parent / "raw.jsonl"
        if raw.exists():
            with raw.open() as file:
                count = sum(1 for line in file if line.strip())
        status = read(path.parent / "acceptance.json")
        incident = "; infrastructure incident recorded" if (path.parent / "incident.json").exists() else ""
        print(f"  {path.parent.relative_to(ROOT / 'results')}: {count}/{info.get('planned_tasks', '?')} tasks; "
              f"status={status.get('full_benchmark_status', status.get('acceptance_status', 'running'))}{incident}")
    print("Session model integration:")
    for path in sorted((ROOT / "results").glob("session_*/probe_status.json")):
        status = read(path)
        workers = read(path.parent / "worker_configuration.json")
        print(f"  {path.parent.name}: {status.get('status', 'running')}; "
              f"{workers.get('completed_model_responses', 0)} audited responses; "
              f"stages={{{', '.join(name + ': ' + stage.get('status', 'unknown') for name, stage in status.get('stages', {}).items())}}}")
    print("Persistent services:")
    try:
        result = subprocess.run(["systemctl", "--user", "list-units", "--all", "--no-pager", "--plain", "--no-legend", "mcts*"],
                                capture_output=True, text=True, timeout=5)
        print(result.stdout.strip() or "  no matching user services")
    except (OSError, subprocess.TimeoutExpired):
        print("  systemd user service information unavailable")


if __name__ == "__main__":
    main()
