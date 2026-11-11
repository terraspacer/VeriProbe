#!/usr/bin/env python3
"""Run reproducible search, verification, and independent benchmark evaluation.

Use ``run --config ... --output ...``; ``inspect`` checks coverage without a
database, ``doctor`` additionally probes configured databases, and ``report``
rebuilds reports from the immutable raw records in an existing output directory.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone

from mcts_rewrite.reporting import ARMS, build_report, finite_number, verified, write_reports
from mcts_rewrite.host_conditions import HostQuietTimeout, TaskHostMonitor

ROOT = Path(__file__).resolve().parent
ABLATIONS = ("full", "class_a_only", "class_b_only", "no_plan", "no_rag", "no_critic", "no_safety")
REQUIRED_MANIFEST = ("id", "source", "benchmark", "sql", "schema_path", "source_path", "sha256", "source_commit")


def digest(value):
    payload = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def public_provenance(value):
    if isinstance(value, dict):
        return {key: public_provenance(item) for key, item in value.items()
                if key.lower() not in {"dsn", "read_dsn", "admin_dsn", "password", "api_key", "token", "secret"}}
    if isinstance(value, list):
        return [public_provenance(item) for item in value]
    return value


def read_manifest(path):
    rows = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [field for field in REQUIRED_MANIFEST if field not in row]
        if missing:
            raise ValueError(f"Manifest line {line_number} lacks {missing}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate manifest id at line {line_number}: {row['id']}")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def parse_list(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def code_fingerprint():
    paths = [Path(__file__), *(ROOT / "mcts_rewrite").glob("*.py")]
    return digest({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)})


def snapshot_sources(output):
    """Keep the executed implementation reviewable if later tuning edits code."""
    paths = [Path(__file__), *(ROOT / "mcts_rewrite").glob("*.py"),
             ROOT / "requirements.lock", ROOT / "upstream.lock.json",
             ROOT / "prompts/action_patterns.jsonl", ROOT / "artifacts/model_manifest.json"]
    for path in paths:
        if path.is_file():
            destination = output / "source_snapshot" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)


def upstream_fingerprint():
    """Include saved rewrites, including any tracked local modifications."""
    versions = {}
    for path in sorted((ROOT / "vendor").glob("*")):
        if not (path / ".git").exists():
            continue
        values = []
        for command in (["git", "rev-parse", "HEAD"], ["git", "diff", "HEAD", "--"], ["git", "ls-files", "--others", "--exclude-standard"]):
            result = subprocess.run(command, cwd=path, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
            values.append(result.stdout)
        untracked = values.pop().decode().splitlines()
        values.extend((path / name).read_bytes() for name in untracked if (path / name).is_file())
        versions[path.name] = digest(b"\0".join(values))
    return versions


def postgres_service_profile(unit):
    """Record configured local service limits without fingerprinting cache usage."""
    if not unit:
        return {"status": "not_configured"}
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("postgres_service_unit must be a nonempty unit name")
    fields = ("ActiveState", "ControlGroup", "MemoryHigh", "MemoryMax", "MemorySwapMax",
              "ManagedOOMPreference", "CPUWeight", "IOWeight")
    try:
        result = subprocess.run(["systemctl", "--user", "show", "--property=" + ",".join(fields), "--", unit],
                                capture_output=True, text=True, timeout=5, check=False)
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        if result.returncode or set(fields) - values.keys() or values.get("ActiveState") != "active":
            raise RuntimeError("Configured PostgreSQL service is unavailable or its limits cannot be read")
        return {"status": "ok", "unit": unit, "properties": values,
                "scope": "Configured local PostgreSQL service; static cgroup limits, not instantaneous memory consumption"}
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
        return {"status": "unavailable", "unit": unit, "error": f"{type(exc).__name__}: {exc}"}


def runtime_fingerprint(config):
    versions = {}
    for package in ("psycopg", "sqlglot", "numpy", "JPype1"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    model = config.get("model")
    model_paths = [os.environ.get("MCTS_MODEL_FILE")]
    if config.get("baseline_catalog"):
        model_paths.append(config["baseline_catalog"])
    if isinstance(model, str):
        model_paths.append(model)
    elif isinstance(model, dict):
        model_paths.append(model.get("file", model.get("path")))
    files = {}
    patterns = ROOT / "prompts/action_patterns.jsonl"
    files[str(patterns)] = digest(patterns.read_bytes()) if patterns.exists() else None
    for name in model_paths:
        if name:
            path = resolve_path(name)
            files[str(path)] = digest(path.read_bytes()) if path.is_file() else None
    cpu_models = []
    if Path("/proc/cpuinfo").exists():
        cpu_models = sorted({line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")})
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "cpu_models": cpu_models, "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "packages": versions, "model_files": files,
            "postgres_service_profile": postgres_service_profile(config.get("postgres_service_unit")),
            "mcts_environment_sha256": digest({key: value for key, value in os.environ.items() if key.startswith("MCTS_")})}


def database_environment(database, timeout_ms=30000):
    """Fingerprint this database instance and its write position for safe resume.

    WAL insertion LSN changes on committed/aborted persistent writes, while server
    start identity also invalidates records after a restart. Unlogged tables are
    detected and make automatic resume unavailable unless an explicit data version
    is provided by the dataset configuration.
    """
    try:
        import psycopg
        with psycopg.connect(database["dsn"], connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT set_config('statement_timeout', %s, false)", (str(timeout_ms),))
                cur.execute("SELECT current_database(), version(), pg_postmaster_start_time()::text, pg_current_wal_insert_lsn()::text")
                name, version, start, wal = cur.fetchone()
                cur.execute("""SELECT n.nspname, c.relname, c.relpersistence, c.reltuples::bigint,
                                      pg_total_relation_size(c.oid)
                               FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                               WHERE c.relkind IN ('r','m') AND n.nspname NOT IN ('pg_catalog','information_schema')
                               AND n.nspname NOT LIKE 'pg_toast%' ORDER BY 1,2""")
                relations = cur.fetchall()
                cur.execute("""SELECT name, setting, unit FROM pg_settings
                               WHERE name IN ('shared_buffers','work_mem','effective_cache_size','jit',
                               'max_parallel_workers','max_parallel_workers_per_gather','max_worker_processes',
                               'random_page_cost','seq_page_cost','cpu_tuple_cost','cpu_index_tuple_cost','cpu_operator_cost',
                               'effective_io_concurrency','default_statistics_target','geqo','geqo_threshold',
                               'plan_cache_mode','track_io_timing','join_collapse_limit','from_collapse_limit',
                               'shared_preload_libraries') OR name LIKE 'enable_%' ORDER BY name""")
                settings = cur.fetchall()
        unsafe_unlogged = any(row[2] == "u" for row in relations) and not database.get("data_version")
        return {"status": "ok", "database": name, "server_version": version,
                "server_started_at": start, "wal_insert_lsn": wal,
                "relations": [list(row) for row in relations],
                "execution_settings": [list(row) for row in settings],
                "data_version": database.get("data_version"), "resume_safe": not unsafe_unlogged,
                "fingerprint_method": "server start + cluster WAL insert LSN + relation catalog + configured data_version"}
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}", "resume_safe": False}


def verification_database_for(config, dataset, database):
    """Resolve an optional correctness DB; an explicit per-dataset null disables it."""
    if "verification_database" in database:
        selected = database["verification_database"]
    else:
        configured = config.get("verification_databases", {})
        if not isinstance(configured, dict):
            raise ValueError("verification_databases must be a mapping")
        selected = configured.get(dataset, configured.get(database.get("benchmark", dataset)))
    if selected is not None and (not isinstance(selected, dict) or not isinstance(selected.get("dsn"), str) or not selected["dsn"].strip()):
        raise ValueError("Each verification_database must contain a nonempty dsn")
    return selected


def verification_database_environments(config):
    return {name: database_environment(selected, config.get("verification_timeout_ms", config.get("timeout_ms", 30000)))
            for name, database in config.get("databases", {}).items()
            if (selected := verification_database_for(config, name, database)) is not None}


def make_tasks(rows, databases, budgets, ablations):
    tasks = []
    for row in rows:
        matches = [(name, database) for name, database in databases.items() if database.get("benchmark", name) == row["benchmark"]]
        if not matches:
            # Preserve the entry even when the configuration lacks its dataset.
            matches = [(row["benchmark"], None)]
        for name, database in matches:
            for budget in budgets:
                for ablation in ablations:
                    tasks.append({"query": row, "dataset": name, "database": database, "budget": budget, "ablation": ablation})
    return tasks


def planned_groups(tasks):
    counts = Counter((task["query"]["source"], task["query"]["benchmark"], task["dataset"], task["ablation"], task["budget"]) for task in tasks)
    databases = {task["dataset"]: task["database"] for task in tasks}
    return [{"source": source, "benchmark": benchmark, "dataset": dataset,
             "scale": (databases[dataset] or {}).get("scale"), "ablation": ablation,
             "budget": budget, "expected_queries": count}
            for (source, benchmark, dataset, ablation, budget), count in sorted(counts.items())]


def task_key(task):
    return digest([task["query"]["id"], task["dataset"], task["budget"], task["ablation"]])


def safe_measure(db, sql):
    try:
        result = db.measure(sql, runs=1, warmups=0)
        value = result.get("median_ms")
        if result.get("status") == "ok" and (not finite_number(value) or value <= 0):
            return {**result, "status": "error", "median_ms": None, "error": "Nonpositive or nonfinite measured execution time"}
        return result
    except Exception as exc:
        return {"status": "error", "median_ms": None, "samples_ms": [],
                "error_type": type(exc).__name__, "sqlstate": getattr(exc, "sqlstate", None),
                "error": f"{type(exc).__name__}: {exc}"}


def evaluate_holdout(db, sql_by_arm, seed, runs=3, warmups=1, share_identical=True):
    """Interleave independent repetitions after selection has been frozen.

    Identical SQL sharing is explicit and optional. Every group receives fresh
    executions; no selection or upstream latency enters these measurements.
    """
    rng = random.Random(seed)
    groups = {}
    arm_to_group = {}
    for arm, sql in sql_by_arm.items():
        key = digest(sql.encode()) if share_identical else arm
        groups.setdefault(key, {"sql": sql, "arms": [], "warmups": [], "measurements": []})["arms"].append(arm)
        arm_to_group[arm] = key
    schedule = []
    failed = set()
    for phase, count in (("warmup", warmups), ("holdout", runs)):
        for repetition in range(count):
            order = list(groups)
            rng.shuffle(order)
            for key in order:
                if key in failed:
                    continue
                started = time.perf_counter()
                measurement = safe_measure(db, groups[key]["sql"])
                wall_ms = (time.perf_counter() - started) * 1000
                event = {"phase": phase, "repetition": repetition, "group": key, "arms": groups[key]["arms"],
                         "wall_ms": wall_ms, "measurement": measurement}
                schedule.append(event)
                groups[key]["warmups" if phase == "warmup" else "measurements"].append(measurement)
                if measurement.get("status") != "ok":
                    failed.add(key)
    results = {}
    for arm, key in arm_to_group.items():
        group = groups[key]
        measurements = group["measurements"]
        failures = [measurement for measurement in [*group["warmups"], *measurements] if measurement.get("status") != "ok"]
        samples = [measurement["median_ms"] for measurement in measurements if measurement.get("status") == "ok"]
        ok = not failures and len(samples) == runs
        latest = measurements[-1] if measurements else group["warmups"][-1] if group["warmups"] else {}
        results[arm] = {
            "status": "ok" if ok else failures[0].get("status", "error") if failures else "incomplete",
            "sql": group["sql"], "sql_sha256": digest(group["sql"].encode()),
            "median_ms": statistics.median(samples) if ok else None,
            "samples_ms": samples, "planned_runs": runs, "completed_runs": len(samples),
            "warmups": group["warmups"], "plan": latest.get("plan"),
            "estimated_cost": latest.get("estimated_cost"),
            "error": failures[0].get("error") if failures else None,
            "error_type": failures[0].get("error_type") if failures else None,
            "sqlstate": failures[0].get("sqlstate") if failures else None,
            "censored": bool(failures), "holdout_group": key,
            "common_execution_identity_pairing": len(group["arms"]) > 1,
            "paired_arms": group["arms"], "independent_of_search_measurements": True,
        }
    return results, schedule


def safe_verify(gate, original, candidate):
    try:
        return gate.verify(original, candidate)
    except Exception as exc:
        return {"accepted": False, "proof": "none", "formal": False,
                "execution": {"equivalent": False, "error_type": type(exc).__name__, "sqlstate": getattr(exc, "sqlstate", None)},
                "reason": f"{type(exc).__name__}: {exc}"}


def infrastructure_failure(value):
    """A lost server must stop a run rather than invalidate remaining queries."""
    if isinstance(value, dict):
        kind = value.get("error_type")
        state = value.get("sqlstate") or ""
        if kind in {"OperationalError", "InterfaceError", "AdminShutdown", "CrashShutdown", "CannotConnectNow", "ConnectionFailure"}:
            return True
        if isinstance(state, str) and (state.startswith("08") or state in {"57P01", "57P02", "57P03"}):
            return True
        return any(infrastructure_failure(item) for item in value.values())
    if isinstance(value, list):
        return any(infrastructure_failure(item) for item in value)
    return False


def run_task(task, config, run_hash, baseline_registry, components=None):
    if components is None:
        from mcts_rewrite.db import Postgres
        from mcts_rewrite.verify import EquivalenceGate
        from mcts_rewrite.adapters import HybridGenerator
        from mcts_rewrite.search import SearchConfig, SearchController
    else:
        Postgres, EquivalenceGate, HybridGenerator, SearchConfig, SearchController = components
    row = task["query"]
    mode = config.get("verification_mode", "strict")
    record = {"record_type": "query_result", "run_hash": run_hash, "task_id": task_key(task),
              "query_id": row["id"], "source": row["source"], "benchmark": row["benchmark"],
              "dataset": task["dataset"], "budget": task["budget"], "ablation": task["ablation"],
              "original_sql": row["sql"], "query_sha256": digest(row["sql"].encode()),
              "manifest_provenance": {key: row[key] for key in ("source_path", "sha256", "source_commit", "schema_path")},
              "started_at": utc_now(), "verification": {}, "baselines": {}, "evaluation": {},
              "status": "error", "verification_mode": mode}
    started = time.perf_counter()
    db = verification_db = None
    try:
        database = task["database"]
        if not database:
            raise ValueError(f"No configured database for benchmark {row['benchmark']}")
        record["dataset_provenance"] = public_provenance(database)
        schema_path = resolve_path(row["schema_path"])
        schema_sql = schema_path.read_text()
        record["schema_sha256"] = digest(schema_sql.encode())
        comparison_options = {key: config[key] for key in ("comparison_mode", "max_rows", "max_result_bytes") if key in config}
        db = Postgres(database["dsn"], timeout_ms=config.get("timeout_ms", 30000), search_path=database.get("search_path", "public"), **comparison_options)
        miniature_config = database.get("mini_dbs")
        if miniature_config is None:
            configured = config.get("mini_dbs", {})
            miniature_config = configured.get(task["dataset"], configured.get(row["benchmark"])) if isinstance(configured, dict) else configured
        record["miniature_configuration"] = public_provenance(miniature_config)
        verification_database = verification_database_for(config, task["dataset"], database)
        gate_options = dict(mode=mode, mini_dbs=miniature_config, solver_timeout=config.get("solver_timeout", 10))
        if verification_database is not None:
            from mcts_rewrite.verification_db import ScopedEquivalenceGate
            verification_db = Postgres(verification_database["dsn"],
                timeout_ms=config.get("verification_timeout_ms", config.get("timeout_ms", 30000)),
                search_path=verification_database.get("search_path", "public"), **comparison_options)
            gate = ScopedEquivalenceGate(db, verification_db, schema_sql, **gate_options,
                performance_provenance=public_provenance(database),
                verification_provenance=public_provenance(verification_database))
            record["verification_context"] = gate.metadata
        else:
            gate = EquivalenceGate(db, schema_sql, **gate_options)
            record["verification_context"] = {"scope": "performance_database", "performance_database_rows_hashed": True,
                "result_evidence": "Configured performance-database rows and any configured miniature cases; finite comparisons are not universal proofs"}
        original = row["sql"]
        record["verification"]["original"] = safe_verify(gate, original, original)
        if not verified(record["verification"]["original"], mode):
            raise ValueError("Original query could not be executed and identity-verified")
        ablation = task["ablation"]
        search_config = SearchConfig(budget=task["budget"], use_plan=ablation != "no_plan", use_rag=ablation != "no_rag",
                                     use_critic=ablation != "no_critic", use_safety=ablation != "no_safety", seed=config.get("seed", 0))
        generator = HybridGenerator(ROOT, model=config.get("model"), include_a=ablation != "class_b_only",
                                    include_b=ablation != "class_a_only", use_rag=ablation != "no_rag")
        search_started = time.perf_counter()
        try:
            selected = SearchController(db, gate, generator, config=search_config).run(original, schema_sql)
            record["search"] = selected
        except Exception as exc:
            selected = {"sql": original, "fallback": True, "trace": [], "expanded": None,
                        "error": f"{type(exc).__name__}: {exc}"}
            record["search"] = selected
            record["search_error"] = selected["error"]
        record["search_wall_ms"] = (time.perf_counter() - search_started) * 1000
        record["search_config"] = dataclasses.asdict(search_config) if dataclasses.is_dataclass(search_config) else vars(search_config)
        candidate = selected.get("sql", original)
        final_verification = safe_verify(gate, original, candidate)
        if not verified(final_verification, mode):
            record["rejected_search_selection"] = {"sql": candidate, "verification": final_verification}
            candidate = original
            final_verification = safe_verify(gate, original, original)
            record["forced_verified_fallback"] = True
        record["verification"]["mcts"] = final_verification
        record["final_sql"] = candidate if verified(final_verification, mode) else None
        record["final_sql_sha256"] = digest(candidate.encode()) if record["final_sql"] is not None else None
        sql_by_arm = {"original": original}
        if verified(final_verification, mode):
            sql_by_arm["mcts"] = candidate
        else:
            record["evaluation"]["mcts"] = {"status": "rejected_verification", "median_ms": None}
        for system in ("rbot", "quite"):
            if hasattr(baseline_registry, "lookup_for_dataset"):
                baseline = baseline_registry.lookup_for_dataset(system, original, row["benchmark"], task["dataset"], record["schema_sha256"])
            else:
                baseline = baseline_registry.lookup(system, original, row["benchmark"])
            if baseline is None:
                record["baselines"][system] = {"status": "unavailable", "mode": "unavailable", "reason": "No exact matching upstream rewrite"}
                record["evaluation"][system] = {"status": "unavailable", "median_ms": None}
                continue
            provenance = baseline.get("provenance", {})
            # The registry returns saved output, never a new optimizer invocation.
            record["baselines"][system] = {"status": "found", "mode": "fresh_standalone" if provenance.get("fresh_standalone") else "replay", "provenance": provenance,
                                            "proposed_sql_sha256": digest(baseline["sql"].encode())}
            evidence = safe_verify(gate, original, baseline["sql"])
            record["verification"][system] = evidence
            if verified(evidence, mode):
                sql_by_arm[system] = baseline["sql"]
            else:
                record["baselines"][system]["status"] = "rejected_verification"
                record["baselines"][system]["rejected_candidate_sql"] = baseline["sql"]
                record["evaluation"][system] = {"status": "rejected_verification", "median_ms": None}
        # No selection API is called after this point. These samples are holdout.
        evaluation_seed = int(digest([config.get("seed", 0), task_key(task), "independent-holdout"])[:16], 16)
        evaluation, schedule = evaluate_holdout(db, sql_by_arm, evaluation_seed, runs=3, warmups=1,
                                                share_identical=config.get("share_identical_sql", True))
        record["evaluation"].update(evaluation)
        record["evaluation_schedule"] = schedule
        record["evaluation_seed"] = evaluation_seed
        record["status"] = "search_error" if record.get("search_error") else "ok" if all(record["evaluation"].get(arm, {}).get("status") == "ok" for arm in ARMS) else "incomplete"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["error_type"] = type(exc).__name__
        record["sqlstate"] = getattr(exc, "sqlstate", None)
        if getattr(exc, "evidence", None):
            record["verification_context"] = {"scope": "separate_verification_database", "catalog_validation": exc.evidence}
        record["traceback"] = traceback.format_exc()
    finally:
        for name, connection in (("verification", verification_db), ("performance", db)):
            if connection is None:
                continue
            try:
                connection.close()
            except Exception as exc:
                record.setdefault("close_errors", {})[name] = f"{type(exc).__name__}: {exc}"
                record["status"] = "error"
    record["wall_ms"] = (time.perf_counter() - started) * 1000
    record["finished_at"] = utc_now()
    return record


def read_raw(path, expected_hash=None):
    records = []
    if not Path(path).exists():
        return records
    seen = set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Raw JSONL line {number} is truncated/corrupt; preserve and repair it before resuming") from exc
        if expected_hash and record.get("run_hash") != expected_hash:
            raise ValueError(f"Raw record {number} has a different run hash")
        if record.get("task_id") in seen:
            raise ValueError(f"Duplicate task in raw records on line {number}")
        seen.add(record.get("task_id"))
        records.append(record)
    return records


def run(args):
    config = read_json(args.config)
    if args.mode:
        config["verification_mode"] = args.mode
    rows = read_manifest(args.manifest)
    selected = rows
    if args.query_id:
        selected = [row for row in selected if row["id"] in set(args.query_id)]
        unknown = set(args.query_id) - {row["id"] for row in rows}
        if unknown:
            raise ValueError(f"Unknown query ids: {sorted(unknown)}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[:args.limit]
    budgets = parse_list(args.budgets, int) if args.budgets else config.get("budgets", [1, 2, 4, 8, 16])
    ablations = parse_list(args.ablations) if args.ablations else config.get("ablations", list(ABLATIONS))
    if not budgets or any(not isinstance(budget, int) or isinstance(budget, bool) or not 1 <= budget <= 16 for budget in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("Budgets must be distinct integers between 1 and 16 expansions")
    if not ablations or set(ablations) - set(ABLATIONS) or len(set(ablations)) != len(ablations):
        raise ValueError(f"Ablations must be distinct members of {ABLATIONS}")
    databases = config.get("databases", {})
    tasks = make_tasks(selected, databases, budgets, ablations)
    output = Path(args.output)
    if (output / "run.json").exists() and not args.resume:
        raise ValueError("Output already contains a run; use --resume or a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    environments = {name: database_environment(database, config.get("timeout_ms", 30000)) for name, database in databases.items()}
    verification_environments = verification_database_environments(config)
    # Credentials are included only as an irreversible digest, never emitted.
    source_fingerprints = {row["id"]: {"schema": digest(resolve_path(row["schema_path"]).read_bytes()) if resolve_path(row["schema_path"]).exists() else None,
                                       "query": digest(row["sql"].encode())} for row in rows}
    fingerprint = {"manifest_sha256": digest(Path(args.manifest).read_bytes()), "config_sha256": digest(config),
                   "code_sha256": code_fingerprint(), "upstreams": upstream_fingerprint(), "runtime": runtime_fingerprint(config),
                   "source_fingerprints": source_fingerprints, "environments": environments,
                   "verification_environments": verification_environments,
                   "selected_query_ids": [row["id"] for row in selected], "budgets": budgets, "ablations": ablations}
    if fingerprint["runtime"]["postgres_service_profile"]["status"] == "unavailable":
        raise RuntimeError("Configured PostgreSQL resource limits could not be recorded; no benchmark task started")
    run_hash = digest(fingerprint)
    metadata = {"created_at": utc_now(), "run_hash": run_hash, "fingerprint": fingerprint,
                "manifest": str(Path(args.manifest).resolve()), "config": str(Path(args.config).resolve()),
                "partial_run": len(selected) != len(rows), "manifest_queries": len(rows),
                "publication_scale_run": config.get("profile") == "published" and bool(databases)
                    and all(database.get("provenance", {}).get("publication_scale") is True for database in databases.values()),
                "selected_queries": len(selected), "planned_tasks": len(tasks),
                "planned_groups": planned_groups(tasks), "verification_mode": config.get("verification_mode", "strict"),
                "model_context": public_provenance(config.get("model_context", {})),
                "verification_databases": {name: public_provenance(selected) for name, database in databases.items()
                    if (selected := verification_database_for(config, name, database)) is not None},
                "dataset_provenance": public_provenance(databases)}
    raw_path = output / "raw.jsonl"
    if args.resume and (output / "run.json").exists():
        old = read_json(output / "run.json")
        if old["run_hash"] != run_hash:
            raise ValueError("Resume refused: manifest, configuration, code, upstream outputs, query selection, or live database environment changed; use a new output directory")
        if not all(env.get("resume_safe") for env in [*environments.values(), *verification_environments.values()]):
            raise ValueError("Resume refused: the live database fingerprint is unavailable or contains unversioned unlogged data")
        if fingerprint["runtime"]["postgres_service_profile"]["status"] == "unavailable":
            raise ValueError("Resume refused: configured PostgreSQL resource limits are unavailable")
        metadata = old
    else:
        (output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        snapshot_sources(output)
    records = read_raw(raw_path, run_hash)
    done = {record["task_id"] for record in records}
    if config.get("baseline_policy", "replay") == "fresh":
        from mcts_rewrite.fresh_baselines import FreshBaselineRegistry
        registry = FreshBaselineRegistry(ROOT, resolve_path(config["baseline_catalog"]), databases=databases)
    elif config.get("baseline_policy", "replay") == "replay":
        from mcts_rewrite.adapters import BaselineRegistry
        registry = BaselineRegistry(ROOT)
    else:
        raise ValueError("baseline_policy must be fresh or replay")
    random.Random(config.get("seed", 0)).shuffle(tasks)
    with raw_path.open("a") as stream:
        for position, task in enumerate(tasks, 1):
            if task_key(task) in done:
                continue
            row = task["query"]
            print(f"[{position}/{len(tasks)}] {row['id']} dataset={task['dataset']} budget={task['budget']} ablation={task['ablation']}", flush=True)
            monitor = TaskHostMonitor(config, task_key(task), run_hash, output / "host_conditions.jsonl", task["database"])
            try:
                with monitor:
                    record = run_task(task, config, run_hash, registry)
            except HostQuietTimeout as exc:
                (output / "incident.json").write_text(json.dumps({"kind": "host_quiet_wait_timeout",
                    "task_id": task_key(task), "time": utc_now(), "host_conditions": exc.summary,
                    "action": "Stopped before executing this task; it remains pending for resume."}, indent=2) + "\n")
                write_reports(output, build_report(records, metadata["planned_groups"], metadata["verification_mode"], metadata))
                raise
            record["host_conditions"] = monitor.summary()
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            records.append(record)
            done.add(record["task_id"])
            write_reports(output, build_report(records, metadata["planned_groups"], metadata["verification_mode"], metadata))
            if infrastructure_failure(record):
                (output / "incident.json").write_text(json.dumps({"kind": "infrastructure_failure", "task_id": record["task_id"],
                    "time": utc_now(), "action": "Stopped after preserving evidence; remaining query memberships were not executed.",
                    "note": "Repair the server and start a new run if its fingerprint changed."}, indent=2) + "\n")
                raise RuntimeError("Database connection failed; run stopped to preserve unexecuted queries")
    report = build_report(records, metadata["planned_groups"], metadata["verification_mode"], metadata)
    write_reports(output, report)
    print(json.dumps({"output": str(output.resolve()), "records": len(records), "acceptance_status": report["acceptance_status"],
                      "full_benchmark_status": "incomplete" if metadata["partial_run"] or not metadata["publication_scale_run"] or metadata["verification_mode"] not in {"strict", "hybrid", "differential"} or not report["publication_eligibility"]["eligible"] else report["acceptance_status"]}))
    return 0


def inspect(args, doctor=False):
    config = read_json(args.config)
    rows = read_manifest(args.manifest)
    from mcts_rewrite.adapters import BaselineRegistry
    from mcts_rewrite.baselines import formatting_fingerprint
    registry = BaselineRegistry(ROOT)
    coverage = Counter()
    unavailable = []
    for row in rows:
        for system in ("rbot", "quite"):
            key = (system, row["benchmark"], formatting_fingerprint(row["sql"]))
            # Static inspection must not start one JVM per saved rule sequence.
            status = "saved_sql_available" if key in registry.records else "reconstruction_required" if key in registry.rule_records else "missing"
            coverage[(row["source"], row["benchmark"], system, status)] += 1
            if status == "missing":
                unavailable.append({"query_id": row["id"], "system": system})
    result = {"queries": len(rows), "source_benchmark_counts": [{"source": source, "benchmark": benchmark, "queries": count} for (source, benchmark), count in sorted(Counter((row["source"], row["benchmark"]) for row in rows).items())],
              "baseline_coverage": [{"source": source, "benchmark": benchmark, "system": system, "status": status, "queries": count} for (source, benchmark, system, status), count in sorted(coverage.items())],
              "missing_baselines": unavailable,
              "missing_schema_paths": sorted({row["schema_path"] for row in rows if not resolve_path(row["schema_path"]).exists()}),
              "baseline_mode": "replay", "inspection_method": "Static exact input matching; saved rule reconstruction is deferred until execution",
              "configured_datasets": public_provenance(config.get("databases", {}))}
    if doctor:
        result["database_environments"] = {name: database_environment(database, config.get("timeout_ms", 30000)) for name, database in config.get("databases", {}).items()}
        result["verification_database_environments"] = verification_database_environments(config)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / ("doctor.json" if doctor else "inspection.json")).write_text(text + "\n")
    print(text)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "inspect", "doctor"):
        command = subparsers.add_parser(name)
        command.add_argument("--manifest", default=str(ROOT / "benchmarks/manifest.jsonl"))
        command.add_argument("--config", default=str(ROOT / "configs/development.json"))
        command.add_argument("--output", required=name == "run")
        if name == "run":
            command.add_argument("--budgets", help="Comma separated expansion budgets, default 1,2,4,8,16")
            command.add_argument("--ablations", help="Comma separated variants; defaults to all seven")
            command.add_argument("--limit", type=int)
            command.add_argument("--query-id", action="append")
            command.add_argument("--resume", action="store_true")
            command.add_argument("--mode", choices=("strict", "hybrid", "differential"),
                                 help="Strict formal evidence or hybrid formal/miniature differential plus real-DB comparison")
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run(args)
        if args.command in ("inspect", "doctor"):
            return inspect(args, args.command == "doctor")
        output = Path(args.output)
        metadata = read_json(output / "run.json")
        records = read_raw(output / "raw.jsonl", metadata["run_hash"])
        write_reports(output, build_report(records, metadata["planned_groups"], metadata.get("verification_mode", "strict"), metadata))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
