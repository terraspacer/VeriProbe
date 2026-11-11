#!/usr/bin/env python3
"""Generate, load, analyze, and attest local benchmark databases.

Generation is sequential to preserve DSB cross-table correlations and bounded
memory. Published means the released target scale; it never means an author
snapshot, because the repositories do not provide one.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid
import sys
import time

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.copy import LibpqWriter
from psycopg.generators import copy_to

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DSN = f"host={ROOT}/.runtime/socket port=55432 dbname=postgres user=steve"
TARGETS = {
    "development": {
        "tpch": {"dbname": "tpch_dev", "benchmark": "tpch", "scale": 0.01},
        "dsb": {"dbname": "dsb_dev", "benchmark": "dsb", "scale": 1},
        "calcite": {"dbname": "calcite_dev", "benchmark": "calcite", "scale": 10000, "distribution": "uniform"}},
    "published": {
        "tpch10": {"dbname": "tpch10", "benchmark": "tpch", "scale": 10},
        "tpch50": {"dbname": "tpch50", "benchmark": "tpch", "scale": 50},
        "dsb10": {"dbname": "dsb10", "benchmark": "dsb", "scale": 10},
        "dsb50": {"dbname": "dsb50", "benchmark": "dsb", "scale": 50},
        "calcite10": {"dbname": "calcite10", "benchmark": "calcite", "scale": 40000000, "distribution": "uniform"},
        "calcite10zipf": {"dbname": "calcite10zipf", "benchmark": "calcite", "scale": 40000000, "distribution": "zipf"}}}
GENERATORS = {
    "tpch": ("tpch-kit", "https://github.com/gregrahn/tpch-kit", "852ad0a5ee31ebefeed884cea4188781dd9613a3"),
    "dsb": ("dsb", "https://github.com/microsoft/dsb", "ec9a156ceee923db1114cbe388f6183b53d49787"),
    "calcite": ("SlabCity", "https://github.com/eidos06/SlabCity", "d57a516d7249869c0c61e44f88795e1ce952e31d")}


class BoundedCopyWriter(LibpqWriter):
    """Apply COPY socket backpressure instead of growing libpq's output queue."""
    def write(self, data):
        block_size = 128 * 1024
        for offset in range(0, len(data), block_size):
            self.connection.wait(copy_to(self.connection.pgconn, data[offset:offset + block_size], flush=True))


def emit(**data):
    print(json.dumps({"time": dt.datetime.now(dt.timezone.utc).isoformat(), **data}), flush=True)


def atomic_json(path, value):
    """Publish complete provenance/checkpoints without exposing partial JSON."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(cmd, cwd=None, env=None):
    emit(command=[str(x) for x in cmd], cwd=str(cwd or ROOT))
    subprocess.run([str(x) for x in cmd], cwd=cwd or ROOT, env=env, check=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_generator(benchmark):
    name, url, commit = GENERATORS[benchmark]
    path = ROOT / ".local/generators" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", url, path])
    current = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if current != commit:
        run(["git", "-C", path, "fetch", "origin", commit])
        run(["git", "-C", path, "checkout", "--detach", commit])
    if benchmark == "tpch":
        run(["make", "-j2", "MACHINE=LINUX", "DATABASE=POSTGRESQL"], path / "dbgen")
    elif benchmark == "dsb":
        run(["make", "-j2", "CC=gcc", "OS=LINUX"], path / "code/tools")
    return path


def generate(target, directory):
    benchmark, scale = target["benchmark"], target["scale"]
    source = prepare_generator(benchmark)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if benchmark == "tpch":
        env = dict(os.environ, DSS_PATH=str(directory))
        cmd = [source / "dbgen/dbgen", "-f", "-s", str(scale), "-b", source / "dbgen/dists.dss"]
        run(cmd, env=env)
    elif benchmark == "dsb":
        # A single all-table generation preserves the official generator's
        # cross-table correlations. -TERMINATE N avoids an extra empty column.
        cmd = [source / "code/tools/dsdgen", "-scale", str(scale), "-terminate", "N",
               "-force", "Y", "-dir", directory, "-rngseed", "0"]
        run(cmd, cwd=source / "code/tools")
    else:
        cmd = [sys.executable, ROOT / "scripts/generate_calcite.py", "--rows", str(scale),
               "--distribution", target["distribution"], "--output", directory]
        run(cmd)
    if benchmark != "calcite":
        executable = source / ("dbgen/dbgen" if benchmark == "tpch" else "code/tools/dsdgen")
        report = {"format_version": 1, "source_url": GENERATORS[benchmark][1],
                  "source_commit": GENERATORS[benchmark][2], "scale": scale,
                  "seed": "TPC-H built-in deterministic RNG streams" if benchmark == "tpch" else 0,
                  "command": [str(x) for x in cmd], "exact_author_snapshot": False,
                  "generator_directory": str(source.relative_to(ROOT)),
                  "executable_sha256": sha256(executable),
                  "elapsed_s": round(time.monotonic() - started, 3)}
        atomic_json(directory / "generation.json", report)


def schema_signature(conn, namespace):
    columns = conn.execute("""SELECT c.relname,a.attnum,a.attname,
        format_type(a.atttypid,a.atttypmod),a.attnotnull,
        pg_get_expr(d.adbin,d.adrelid),a.attidentity,a.attgenerated
        FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        JOIN pg_attribute a ON a.attrelid=c.oid
        LEFT JOIN pg_attrdef d ON d.adrelid=c.oid AND d.adnum=a.attnum
        WHERE n.nspname=%s AND c.relkind='r' AND a.attnum>0 AND NOT a.attisdropped
        AND c.relname NOT LIKE '_mcts_%%' ORDER BY c.relname,a.attnum""", (namespace,)).fetchall()
    constraints = conn.execute("""SELECT c.relname,k.conname,k.contype,
        pg_get_constraintdef(k.oid) FROM pg_constraint k
        JOIN pg_class c ON c.oid=k.conrelid JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname=%s AND c.relname NOT LIKE '_mcts_%%'
        ORDER BY c.relname,k.conname""", (namespace,)).fetchall()
    indexes = conn.execute("""SELECT c.relname,i.relname,x.indisunique,x.indisprimary,
        x.indkey::text,x.indclass::text,x.indcollation::text,x.indoption::text,
        pg_get_expr(x.indexprs,x.indrelid),pg_get_expr(x.indpred,x.indrelid)
        FROM pg_index x JOIN pg_class c ON c.oid=x.indrelid
        JOIN pg_class i ON i.oid=x.indexrelid JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname=%s AND c.relname NOT LIKE '_mcts_%%'
        ORDER BY c.relname,i.relname""", (namespace,)).fetchall()
    behavior = conn.execute("""SELECT c.relname,c.relkind,c.relhastriggers,
        c.relrowsecurity,c.relhasrules FROM pg_class c
        JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname=%s AND c.relname NOT LIKE '_mcts_%%'
        AND c.relkind NOT IN ('i','S') ORDER BY c.relname""", (namespace,)).fetchall()
    # Schema-qualified scratch references must compare with public references.
    constraints = [[value.replace(namespace + ".", "") if isinstance(value, str) else value
                    for value in row] for row in constraints]
    return {"columns": [list(row) for row in columns], "constraints": constraints,
            "indexes": [list(row) for row in indexes], "behavior": [list(row) for row in behavior]}


def validate_schema(conn, schema):
    """Compare real catalog columns with the pinned DDL without touching data."""
    actual = schema_signature(conn, "public")
    namespace = "_mcts_validate_" + uuid.uuid4().hex
    # Roll back the scratch namespace and tables after reading their signature.
    with conn.transaction(force_rollback=True):
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(namespace)))
        conn.execute(sql.SQL("SET LOCAL search_path={}").format(sql.Identifier(namespace)))
        conn.execute(schema.read_text())
        expected = schema_signature(conn, namespace)
    if actual != expected:
        raise RuntimeError("Existing table schema differs from pinned benchmark DDL")
    return expected


@contextmanager
def rebuild_primary_key_after_copy(conn, table, enabled=False):
    """Restore the original key before the surrounding COPY transaction commits."""
    constraints = conn.execute("""SELECT conname,pg_get_constraintdef(oid)
        FROM pg_constraint WHERE conrelid=%s::regclass AND contype='p'
        ORDER BY conname""", (table,)).fetchall() if enabled else []
    for name, definition in constraints:
        conn.execute(sql.SQL("ALTER TABLE {} DROP CONSTRAINT {}").format(sql.Identifier(table), sql.Identifier(name)))
    yield
    if constraints:
        conn.execute("SET LOCAL maintenance_work_mem='256MB'")
        for name, definition in constraints:
            emit(database=conn.info.dbname, table=table, status="primary_key_build_started", constraint=name)
            conn.execute(sql.SQL("ALTER TABLE {} ADD CONSTRAINT {} {}").format(sql.Identifier(table), sql.Identifier(name), sql.SQL(definition)))
            emit(database=conn.info.dbname, table=table, status="primary_key_build_complete", constraint=name)


def audit_ready_counts(target, directory, admin_dsn, report, schema, generation):
    """Honor an explicit count audit even when a ready database needs no load."""
    if report.get("generation") != generation or report.get("schema_sha256") != sha256(schema):
        raise RuntimeError("Ready dataset provenance differs from its source/schema")
    suffix = "csv" if target["benchmark"] == "calcite" else "tbl" if target["benchmark"] == "tpch" else "dat"
    paths = {path.stem.lower(): path for path in directory.glob(f"*.{suffix}")}
    audit = {"database": target["dbname"], "audit_passed": False, "tables": {},
             "method": "Explicit independent SELECT count(*) plus source SHA256 and pinned catalog validation",
             "started": dt.datetime.now(dt.timezone.utc).isoformat()}
    audit_path = ROOT / ".runtime/datasets" / f"{target['dbname']}.audit.json"
    atomic_json(audit_path, audit)
    with psycopg.connect(make_conninfo(admin_dsn, dbname=target["dbname"]), autocommit=True) as conn:
        conn.execute("SET statement_timeout=0")
        validate_schema(conn, schema)
        persisted = conn.execute("SELECT record FROM _mcts_dataset_provenance").fetchone()[0]
        if persisted["tables"] != report["tables"] or persisted["scale"] != target["scale"]:
            raise RuntimeError("Persisted ready-dataset provenance differs from the file report")
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            for table, receipt in sorted(report["tables"].items()):
                path = paths[table]
                if path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["source_sha256"]:
                    raise RuntimeError(f"Ready audit source hash mismatch: {table}")
                rows = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()[0]
                if rows != receipt["rows"]:
                    raise RuntimeError(f"Ready audit row count mismatch: {table}")
                audit["tables"][table] = {"rows": rows, "source_sha256": receipt["source_sha256"]}
                emit(database=target["dbname"], table=table, status="full_count_audit_passed", rows=rows)
    audit["audit_passed"] = True
    audit["completed"] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_json(audit_path, audit)


def load(target, directory, admin_dsn, profile, replace=False, full_count_verify=False, resume=False):
    dbname, benchmark = target["dbname"], target["benchmark"]
    report_path = ROOT / ".runtime/datasets" / f"{dbname}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    schema = ROOT / f"vendor/LLM4Rewrite/{benchmark}/create_tables.sql"
    generation_path = directory / "generation.json"
    if not generation_path.exists():
        raise RuntimeError(f"Missing completed generator provenance {generation_path}")
    generation = json.loads(generation_path.read_text())
    generated_scale = generation.get("size_parameter", generation.get("scale"))
    if generated_scale != target["scale"] or generation.get("source_commit") != GENERATORS[benchmark][2]:
        raise RuntimeError(f"Generator provenance does not match selected target: {dbname}")
    if benchmark == "calcite" and generation.get("distribution") != target["distribution"]:
        raise RuntimeError(f"Generator distribution does not match selected target: {dbname}")
    generator_directory = ROOT / generation.get("generator_directory", str(Path(".local/generators") / GENERATORS[benchmark][0]))
    actual_commit = subprocess.check_output(["git", "-C", str(generator_directory), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != GENERATORS[benchmark][2]:
        raise RuntimeError(f"Generator checkout revision mismatch: {dbname}")
    executable = generator_directory / ("dbgen/dbgen" if benchmark == "tpch" else "code/tools/dsdgen")
    if benchmark != "calcite" and generation.get("executable_sha256") and sha256(executable) != generation["executable_sha256"]:
        raise RuntimeError(f"Generator executable hash mismatch: {dbname}")
    if benchmark == "calcite" and sha256(generator_directory / generation["source_path"]) != generation["source_sha256"]:
        raise RuntimeError(f"Generator source hash mismatch: {dbname}")
    prior = json.loads(report_path.read_text()) if report_path.exists() else None
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        exists = bool(admin.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dbname,)).fetchone())
        if exists and not replace and prior and prior.get("ready"):
            if full_count_verify:
                audit_ready_counts(target, directory, admin_dsn, prior, schema, generation)
            emit(database=dbname, status="already_ready")
            return
        if exists and not replace:
            if not resume or not prior:
                raise RuntimeError(f"Database {dbname} is incomplete; use --resume with its checkpoint or --replace")
            for key in ("dbname", "benchmark", "scale"):
                if prior.get(key) != target[key]:
                    raise RuntimeError(f"Resume target mismatch in {key}: {dbname}")
            if prior.get("schema_sha256") != sha256(schema) or prior.get("generation") != generation:
                raise RuntimeError(f"Resume schema or generator provenance mismatch: {dbname}")
        if exists and replace:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(dbname)))
            exists = False
        if not exists:
            admin.execute(sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(dbname)))
    report = {"database": dbname, **target, "profile": profile, "ready": False,
              "exact_author_snapshot": False, "generation": generation,
              "schema_path": str(schema.relative_to(ROOT)), "schema_sha256": sha256(schema),
              "tables": {}, "started": dt.datetime.now(dt.timezone.utc).isoformat()}
    if exists:
        report = prior
        report.setdefault("resume_events", []).append({"time": dt.datetime.now(dt.timezone.utc).isoformat(),
            "generation_json_sha256": sha256(generation_path), "schema_sha256": sha256(schema),
            "completed_tables": sorted(report["tables"])})
    atomic_json(report_path, report)
    started = time.monotonic()
    with psycopg.connect(make_conninfo(admin_dsn, dbname=dbname), autocommit=True) as conn:
        conn.execute("SET statement_timeout=0")
        if not exists:
            conn.execute(schema.read_text())
        signature = validate_schema(conn, schema)
        report["catalog_schema_sha256"] = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_hint_plan")
        conn.execute("CREATE TABLE IF NOT EXISTS _mcts_dataset_load (table_name text PRIMARY KEY, record jsonb NOT NULL)")
        server_records = dict(conn.execute("SELECT table_name,record FROM _mcts_dataset_load"))
        # Server records are committed in the same transaction as COPY. They
        # recover the small commit-to-file-publication interruption window.
        for table, record in server_records.items():
            if table in report["tables"] and report["tables"][table] != record:
                raise RuntimeError(f"Server/file table checkpoint mismatch: {table}")
            report["tables"][table] = record
        suffix = "csv" if benchmark == "calcite" else "tbl" if benchmark == "tpch" else "dat"
        paths = sorted(directory.glob(f"*.{suffix}"))
        if not paths:
            raise RuntimeError(f"No generated {suffix} files in {directory}")
        for path in paths:
            table = path.stem.lower()
            checkpoint = report["tables"].get(table)
            if checkpoint:
                if checkpoint["bytes"] != path.stat().st_size or checkpoint["source_sha256"] != sha256(path):
                    raise RuntimeError(f"Resume source hash mismatch: {table}")
                if table in server_records and not full_count_verify:
                    counted_rows = checkpoint.get("server_copy_rows")
                    if counted_rows != checkpoint["rows"] or checkpoint.get("source_rows") != counted_rows:
                        raise RuntimeError(f"Inconsistent atomic COPY receipt: {table}")
                    method = "Atomic same-transaction server COPY/source row counts; dedicated read-only benchmark role; no repeat table scan"
                else:
                    counted_rows = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()[0]
                    if counted_rows != checkpoint["rows"]:
                        raise RuntimeError(f"Resume committed table count mismatch: {table}")
                    method = "Independent SELECT count(*) for legacy file-only checkpoint or explicit full-count audit"
                emit(database=dbname, table=table, status="resume_table_validated", rows=counted_rows,
                     evidence="source SHA256, catalog schema, pinned generator; " + method)
                continue
            if exists:
                # Only this task's incomplete tables may be reset. A completed
                # checkpoint is never truncated, even when validation fails.
                emit(database=dbname, table=table, status="resume_incomplete_table_reset", previous_rows="unknown_uncheckpointed",
                     reason="No committed completed-table checkpoint; restart this table only")
            columns = None
            if benchmark == "calcite":
                columns = generation["tables"][table]["columns"]
            copy = sql.SQL("COPY {} {}").format(sql.Identifier(table),
                sql.SQL("({})").format(sql.SQL(",").join(sql.Identifier(c.lower()) for c in columns)) if columns else sql.SQL(""))
            # Quote byte 0x01 does not occur in official plain-text generators.
            copy += sql.SQL(" FROM STDIN WITH (FORMAT csv, DELIMITER {}, QUOTE E'\\x01', NULL '')").format(
                sql.Literal("," if benchmark == "calcite" else "|")) if benchmark != "calcite" else sql.SQL(" FROM STDIN WITH (FORMAT csv, NULL '')")
            digest = hashlib.sha256()
            rows = 0
            emit(database=dbname, table=table, status="copy_started", bytes=path.stat().st_size)
            rebuild_key = benchmark == "dsb" and path.stat().st_size >= 100_000_000
            with conn.transaction(), rebuild_primary_key_after_copy(conn, table, rebuild_key), path.open("rb") as f, conn.cursor() as cursor:
                if exists:
                    conn.execute(sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(table)))
                with cursor.copy(copy, writer=BoundedCopyWriter(cursor)) as copier:
                    if benchmark == "calcite":
                        for _ in range(2):
                            digest.update(f.readline())
                    batch = []
                    for line in f:
                        digest.update(line)
                        rows += 1
                        if benchmark == "tpch" and line.endswith(b"|\n"):
                            line = line[:-2] + b"\n"
                        batch.append(line)
                        if len(batch) == 4096:
                            copier.write(b"".join(batch))
                            batch.clear()
                    if batch:
                        copier.write(b"".join(batch))
                committed_rows = cursor.rowcount
                table_record = {"rows": rows, "source_rows": rows, "server_copy_rows": committed_rows,
                    "row_count_verification": "PostgreSQL completed COPY command tag in checkpoint transaction",
                    "copy_transport": "libpq output flushed in bounded 128KiB blocks",
                    "source_sha256": digest.hexdigest(), "bytes": path.stat().st_size}
                if rebuild_key:
                    table_record["primary_key_load"] = "Original primary key rebuilt after COPY within the same transaction"
                if committed_rows != rows:
                    raise RuntimeError(f"Row mismatch {table}: source{rows} loaded{committed_rows}")
                conn.execute("INSERT INTO _mcts_dataset_load VALUES (%s,%s::jsonb)", (table, json.dumps(table_record)))
            # COPY's completed command tag is PostgreSQL's exact committed row
            # count. These are fresh empty tables, without row-changing triggers.
            method = "PostgreSQL completed COPY command tag on fresh empty table"
            if full_count_verify:
                counted_rows = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()[0]
                if counted_rows != committed_rows:
                    raise RuntimeError(f"COPY/count mismatch {table}: {committed_rows} versus {counted_rows}")
                method += " plus independent SELECT count(*)"
            if committed_rows != rows:
                raise RuntimeError(f"Row mismatch {table}: source{rows} loaded{committed_rows}")
            # The transaction record above is authoritative after a crash.
            report["tables"][table] = table_record
            atomic_json(report_path, report)
            emit(database=dbname, table=table, status="copy_complete", rows=rows)
        expected = {r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'")}
        missing = expected - report["tables"].keys() - {"dbgen_version", "_mcts_dataset_load", "_mcts_dataset_provenance"}
        if missing:
            raise RuntimeError(f"Missing generated tables: {sorted(missing)}")
        if benchmark == "calcite":
            for table in ("emp", "emp_b", "empnullables", "bonus"):
                if report["tables"][table]["rows"] != target["scale"]:
                    raise RuntimeError(f"Wrong published size for {table}")
        validate_schema(conn, schema)
        conn.execute("ANALYZE")
        report["ready"] = True
        report["total_rows"] = sum(t["rows"] for t in report["tables"].values())
        report["database_bytes"] = conn.execute("SELECT pg_database_size(current_database())").fetchone()[0]
        report["postgres_settings"] = dict(conn.execute("SELECT name,setting FROM pg_settings WHERE name IN ('server_version','shared_buffers','work_mem','maintenance_work_mem','autovacuum','max_wal_size','checkpoint_timeout','checkpoint_completion_target','fsync','synchronous_commit','wal_level','max_parallel_workers_per_gather','jit','session_preload_libraries','shared_preload_libraries')"))
        report["extensions"] = dict(conn.execute("SELECT extname,extversion FROM pg_extension"))
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        report["completed"] = dt.datetime.now(dt.timezone.utc).isoformat()
        with conn.transaction():
            conn.execute("CREATE TABLE IF NOT EXISTS _mcts_dataset_provenance (record jsonb NOT NULL)")
            conn.execute("DELETE FROM _mcts_dataset_provenance")
            conn.execute("INSERT INTO _mcts_dataset_provenance VALUES (%s::jsonb)", (json.dumps(report),))
        # If root already created the benchmark role, grant its required reads.
        if conn.execute("SELECT 1 FROM pg_roles WHERE rolname='mcts_read'").fetchone():
            conn.execute("GRANT USAGE ON SCHEMA public TO mcts_read")
            conn.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO mcts_read")
        atomic_json(report_path, report)
        emit(database=dbname, status="ready", total_rows=report["total_rows"], elapsed_s=report["elapsed_s"])


def write_configs():
    for profile, targets in TARGETS.items():
        databases = {}
        for name, target in targets.items():
            benchmark = target["benchmark"]
            databases[name] = {**target, "dsn": make_conninfo(DEFAULT_DSN, dbname=target["dbname"], user="mcts_read"),
                "provenance": {"profile": profile, "generator_url": GENERATORS[benchmark][1],
                               "generator_commit": GENERATORS[benchmark][2],
                               "readiness_report": f".runtime/datasets/{target['dbname']}.json",
                               "exact_author_snapshot": False,
                               "scale_unit": "rows per main table" if benchmark == "calcite" else "generator scale factor",
                               "publication_scale": profile == "published"}}
        config = {"format_version": 1, "profile": profile, "databases": databases,
                  "timeout_ms": 30000 if profile == "development" else 300000,
                  "budgets": [1, 2, 4, 8, 16],
                  "seed": 0, "strict": True, "measurement_runs": 3, "warmups": 1}
        (ROOT / f"configs/{profile}.json").write_text(json.dumps(config, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=TARGETS, default="development")
    parser.add_argument("--only", action="append", help="Target name; may be repeated")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--load-when-ready", action="store_true",
                        help="Load each completed generation as it becomes available; keep waiting for pending targets")
    parser.add_argument("--replace", action="store_true", help="Replace selected existing databases")
    parser.add_argument("--resume", action="store_true", help="Validate and preserve committed tables of an incomplete task-owned database")
    parser.add_argument("--full-count-verify", action="store_true", help="Independently scan table counts, including already-ready databases")
    parser.add_argument("--write-configs", action="store_true")
    parser.add_argument("--admin-dsn", default=DEFAULT_DSN)
    args = parser.parse_args()
    if args.replace and args.resume:
        parser.error("--replace and --resume are mutually exclusive")
    if args.write_configs:
        write_configs()
        return
    if args.load_when_ready:
        pending = list(args.only or TARGETS[args.profile])
        def resume_priority(name):
            path = ROOT / ".runtime/datasets" / f"{TARGETS[args.profile][name]['dbname']}.json"
            return 0 if path.exists() and not json.loads(path.read_text()).get("ready") else 1
        pending.sort(key=resume_priority)
        deadline = time.monotonic() + 24 * 3600
        while pending:
            for name in list(pending):
                directory = ROOT / ".runtime/data" / args.profile / name
                if (directory / "generation.json").exists():
                    try:
                        load(TARGETS[args.profile][name], directory, args.admin_dsn, args.profile, args.replace, args.full_count_verify, args.resume)
                    except psycopg.OperationalError as exc:
                        if not args.resume:
                            raise
                        emit(status="database_temporarily_unavailable", target=name,
                             error=str(exc), sqlstate=exc.sqlstate,
                             action="Wait for PostgreSQL and retry validated table checkpoints")
                        break
                    pending.remove(name)
            if pending:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Generation not complete after 24 hours: {pending}")
                emit(status="waiting_for_generation", pending=pending)
                time.sleep(10)
        return
    for name in args.only or TARGETS[args.profile]:
        target = TARGETS[args.profile][name]
        directory = ROOT / ".runtime/data" / args.profile / name
        if not args.load_only:
            if not (directory / "generation.json").exists():
                generate(target, directory)
            else:
                emit(target=name, status="generation_already_complete")
        if not args.generate_only:
            load(target, directory, args.admin_dsn, args.profile, args.replace, args.full_count_verify, args.resume)


if __name__ == "__main__":
    main()
