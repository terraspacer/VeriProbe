"""Dataset-bound catalogs of actual, independently generated upstream outputs.

Catalog SQL remains an untrusted proposal.  Consumers must run their own proof,
same-database comparison, and measurements.  Saved published SQL is never a
fallback for this registry.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlsplit, urlunsplit

from .adapters import UpstreamStandalone, _run_process, exact_sha256
from .baselines import _commit
from .upstream_runtime import model_context, SESSION_TRANSPORT, SESSION_PROVENANCE_FIELDS

SYSTEMS = ("rbot", "quite")
SOURCE_FIELDS = ("id", "source", "benchmark", "source_path", "source_commit",
                 "source_file_sha256", "schema_path", "schema_sha256", "sha256")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def fingerprint(value) -> str:
    return exact_sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(root: Path, path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_catalog(path: Path, rows: list[dict]) -> None:
    _atomic_text(path, "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path.name} line {number} is not an object")
        rows.append(row)
    return rows


def input_groups(root: Path, manifest: Path, query_ids=None, limit=None) -> list[dict]:
    """Preserve source memberships while sharing byte-identical input calls."""
    selected = set(query_ids or [])
    found, groups = set(), {}
    for row in _read_jsonl(manifest):
        if selected and row.get("id") not in selected:
            continue
        found.add(row["id"])
        if row.get("sha256") != exact_sha256(row["sql"]):
            raise ValueError(f"Manifest SQL hash mismatch: {row['id']}")
        schema = _path(root, row["schema_path"])
        if file_sha256(schema) != row.get("schema_sha256"):
            raise ValueError(f"Manifest schema hash mismatch: {row['id']}")
        if file_sha256(_path(root, row["source_path"])) != row.get("source_file_sha256"):
            raise ValueError(f"Manifest source file hash mismatch: {row['id']}")
        source = {field: row[field] for field in SOURCE_FIELDS}
        key = (row["benchmark"], row["sha256"], row["schema_sha256"])
        group = groups.setdefault(key, {"benchmark": row["benchmark"], "original_sql": row["sql"],
                                        "original_sha256": row["sha256"], "schema_sha256": row["schema_sha256"],
                                        "schema_path": row["schema_path"], "input_sources": []})
        group["input_sources"].append(source)
    if selected - found:
        raise ValueError("Unknown query id(s): " + ", ".join(sorted(selected - found)))
    result = list(groups.values())
    return result if limit is None else result[:limit]


def _public_url(value: str) -> str:
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if ":" in host:
        host = "[" + host + "]"
    if parsed.port:
        host += ":" + str(parsed.port)
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def model_configuration(system: str, environment: dict) -> dict:
    """Hash public model settings; never hash or expose API credential values."""
    if system == "rbot":
        kind = environment.get("RBOT_MODEL_TYPE", "")
        return {"model_type": kind, "model": "gpt-3.5-turbo-0125" if "gpt3" in kind else "gpt-4o",
                "embedding_model": model_context(system, environment)["embedding_source"]["model"],
                "model_context": model_context(system, environment),
                "endpoint": _public_url(environment.get("OPENAI_API_BASE") or environment.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")}
    return {"model_context": model_context(system, environment), "roles": {
            role: {"model": environment.get(role, ""), "endpoint": _public_url(environment.get(role + "_URL", ""))}
            for role in ("REASONING_MODEL", "DECISION_MODEL", "ASSISTANT_MODEL")}}


def bound_session_environment(environment: dict, run_id: str) -> dict:
    """Label HTTP requests with their own baseline run without changing prompts."""
    bound = dict(environment)
    if bound.get("MCTS_MODEL_TRANSPORT") != SESSION_TRANSPORT:
        return bound
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", run_id):
        raise ValueError("Invalid session run id")
    for name in ("OPENAI_API_BASE", "OPENAI_BASE_URL", "REASONING_MODEL_URL", "DECISION_MODEL_URL", "ASSISTANT_MODEL_URL"):
        if not bound.get(name):
            continue
        parsed = urlsplit(bound[name])
        if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "::1")
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path.rstrip("/") not in ("", "/v1")):
            raise ValueError("Session model transport requires a literal localhost /v1 endpoint")
        bound[name] = urlunsplit((parsed.scheme, parsed.netloc, "/runs/" + run_id + "/v1", "", ""))
    bound["MCTS_SESSION_BRIDGE_RUN_ID"] = run_id
    bound["NO_PROXY"] = ",".join(filter(None, [bound.get("NO_PROXY", ""), "127.0.0.1", "localhost", "::1"]))
    bound["no_proxy"] = bound["NO_PROXY"]
    return bound


def missing_model_configuration(system: str, environment: dict) -> list[str]:
    if system == "rbot":
        if "open" in environment.get("RBOT_MODEL_TYPE", ""):
            raise ValueError("Pinned R-Bot open mode has hardcoded local model paths/endpoints; configure the supported OpenAI mode for this catalog")
        names = ["OPENAI_API_KEY"]
        if environment.get("OPENAI_API_BASE"):
            names.append("OPENAI_API_BASE")
        elif environment.get("OPENAI_BASE_URL"):
            names.append("OPENAI_BASE_URL")
    else:
        names = [role + suffix for role in ("REASONING_MODEL", "DECISION_MODEL", "ASSISTANT_MODEL")
                 for suffix in ("", "_API_KEY", "_URL")]
    missing = []
    for name in names:
        value = environment.get(name, "").strip()
        if not value or value.startswith(("[", "YOUR_")):
            missing.append(name)
        elif name.endswith("_URL") or name == "OPENAI_API_BASE":
            try:
                parsed = urlsplit(value)
                if parsed.scheme not in ("http", "https") or not parsed.hostname:
                    missing.append(name)
                _ = parsed.port
            except ValueError:
                missing.append(name)
    return missing


def dataset_configuration_sha256(database: dict) -> str:
    # Connection secrets are excluded.  Database name, scale, distribution and
    # version/provenance still bind the choice to its intended dataset.
    from psycopg.conninfo import conninfo_to_dict
    connection = conninfo_to_dict(database.get("read_dsn") or database["dsn"])
    public_connection = {name: connection.get(name) for name in
                         ("host", "hostaddr", "port", "user", "dbname", "sslmode")}
    return fingerprint({"connection": public_connection, "dataset": {key: database.get(key) for key in
                        ("dbname", "benchmark", "scale", "distribution", "data_version", "provenance")}})


def bound_database_environment(database: dict, base: dict, *, timeout_ms=30000, threads=1) -> dict:
    from psycopg.conninfo import conninfo_to_dict
    values = conninfo_to_dict(database.get("read_dsn") or database["dsn"])
    environment = dict(base)
    bindings = {"PGHOST": values.get("host", base.get("PGHOST", "localhost")),
                "PGPORT": values.get("port", base.get("PGPORT", "5432")),
                "PGUSER": values.get("user", base.get("PGUSER", getpass.getuser())),
                "PGDATABASE": values.get("dbname", database.get("dbname", "")),
                "PGPASSWORD": values.get("password", base.get("PGPASSWORD", ""))}
    if not bindings["PGDATABASE"] or not str(bindings["PGPORT"]).isdigit():
        raise ValueError("A dataset needs an explicit database and a single numeric PostgreSQL port")
    environment.update(bindings)
    for upstream, libpq in (("DB_HOST", "PGHOST"), ("DB_PORT", "PGPORT"), ("DB_USER", "PGUSER"),
                            ("DB_NAME", "PGDATABASE"), ("DB_PASSWORD", "PGPASSWORD")):
        environment[upstream] = bindings[libpq]
    for key in ("PGSERVICE", "PGSERVICEFILE", "PGHOSTADDR", "PGOPTIONS"):
        environment.pop(key, None)
    if values.get("hostaddr"):
        environment["PGHOSTADDR"] = values["hostaddr"]
    for parameter, name in (("sslmode", "PGSSLMODE"), ("sslrootcert", "PGSSLROOTCERT"),
                            ("sslcert", "PGSSLCERT"), ("sslkey", "PGSSLKEY")):
        if parameter in values:
            environment[name] = values[parameter]
    environment["PGOPTIONS"] = (values.get("options", "") +
                                f" -c default_transaction_read_only=on -c statement_timeout={int(timeout_ms)}").strip()
    for name in THREAD_VARIABLES:
        environment[name] = str(int(threads))
    environment.setdefault("JAVA_TOOL_OPTIONS", "-Xmx1024m")
    environment.setdefault("PYTHONHASHSEED", "0")
    return environment


def probe_read_only_database(environment: dict) -> dict:
    """Reject privileged/write-capable roles before launching an upstream agent."""
    import psycopg
    connection_options = {parameter: environment[name] for parameter, name in
                          (("hostaddr", "PGHOSTADDR"), ("sslmode", "PGSSLMODE"), ("sslrootcert", "PGSSLROOTCERT"),
                           ("sslcert", "PGSSLCERT"), ("sslkey", "PGSSLKEY")) if name in environment}
    with psycopg.connect(host=environment["PGHOST"], port=environment["PGPORT"],
                         user=environment["PGUSER"], password=environment["PGPASSWORD"],
                         dbname=environment["PGDATABASE"], options=environment["PGOPTIONS"],
                         connect_timeout=5, **connection_options) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""SELECT current_database(), current_user, current_setting('server_version_num'),
                                     current_setting('transaction_read_only'), rolsuper, rolcreatedb, rolcreaterole,
                                     has_database_privilege(current_database(), 'CREATE')
                              FROM pg_roles WHERE rolname=current_user""")
            name, role, version, readonly, *privileges = cursor.fetchone()
            cursor.execute("""SELECT EXISTS(SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                              WHERE c.relkind IN ('r','p','v','m','f') AND n.nspname NOT IN ('pg_catalog','information_schema')
                              AND n.nspname NOT LIKE 'pg_toast%'
                              AND has_table_privilege(c.oid, 'INSERT,UPDATE,DELETE,TRUNCATE'))""")
            writes = cursor.fetchone()[0]
    if readonly != "on" or any(privileges) or writes:
        raise ValueError("Fresh upstream execution requires a read-only session and a role without administrative/database creation or table write privileges")
    return {"database": name, "role": role, "server_version_num": version,
            "read_only": True, "write_privileges": False}


def _record_key(row: dict) -> tuple:
    return tuple(row.get(name) for name in ("system", "dataset", "benchmark", "original_sha256", "schema_sha256"))


def _source_identity(source: dict) -> str:
    return fingerprint({field: source.get(field) for field in SOURCE_FIELDS})


def _safe_model_context(context: dict) -> dict:
    result = {field: context[field] for field in ("transport", "actual_serving_model", "reasoning_effort", "model_substitution",
              "published_model_configuration_reproduced", "sampling_controls_reproduced", "usage_known")
              if field in context and (context[field] is None or isinstance(context[field], (str, bool)))}
    for field in ("requested_role_models", "actual_role_models"):
        values = context.get(field, {})
        result[field] = {role: value for role, value in values.items() if role in {"rewrite", "reasoning", "decision", "assistant"}
                         and isinstance(value, str)} if isinstance(values, dict) else {}
    embedding = context.get("embedding_source", {})
    result["embedding_source"] = {field: embedding[field] for field in ("provider", "model", "text_embedding_dimension",
        "index_combined_dimension", "requires_live_query_embeddings", "available_via_session_bridge", "substituted",
        "revision", "feature_dimension", "max_length_wordpieces", "weights_sha256", "model_manifest_sha256",
        "index_build_sha256", "index_validation_sha256", "available_locally", "original_model", "native_retrieval_code_unchanged")
        if field in embedding and (embedding[field] is None or isinstance(embedding[field], (str, int, bool)))} if isinstance(embedding, dict) else {}
    result["requested_sampling_options_not_enforced"] = [name for name in context.get("requested_sampling_options_not_enforced", [])
                                                        if isinstance(name, str)]
    if result.get("transport") == SESSION_TRANSPORT:
        if (result.get("actual_serving_model") != "gpt-6.1-sol" or result.get("reasoning_effort") != "xhigh"
                or any(value != "gpt-6.1-sol" for value in result["actual_role_models"].values())):
            raise ValueError("Session model attribution mismatch")
        result.update({"published_model_configuration_reproduced": False, "sampling_controls_reproduced": False, "usage_known": False})
    return result


class FreshBaselineRegistry:
    """Read only exact-input successful fresh outputs for their own dataset."""

    def __init__(self, root: Path, catalog_path: Path | str, *, databases: dict | None = None):
        self.root = Path(root).resolve()
        self.path = _path(self.root, catalog_path)
        self.catalog_sha256 = file_sha256(self.path)
        self.diagnostics, self.records = [], {}
        groups = input_groups(self.root, self.root / "benchmarks/manifest.jsonl")
        self.inputs = {(group["benchmark"], group["original_sha256"], group["schema_sha256"]): group for group in groups}
        commits = {system: _commit(self.root / "vendor" / directory) for system, directory in
                   (("rbot", "LLM4Rewrite"), ("quite", "QUITE"))}
        ambiguous = set()
        for number, line in enumerate(self.path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("status") != "ok" or row.get("upstream_identity_fallback"):
                    continue
                if row.get("system") not in SYSTEMS or not isinstance(row.get("dataset"), str) or not row["dataset"]:
                    raise ValueError("Invalid system/dataset")
                original, sql = row["original_sql"], row["sql"]
                if not isinstance(original, str) or not isinstance(sql, str) or not sql.strip():
                    raise ValueError("Original and output SQL must be strings")
                if row.get("original_sha256") != exact_sha256(original):
                    raise ValueError("Original SQL hash mismatch")
                if row.get("sql_sha256") != exact_sha256(sql):
                    raise ValueError("Output SQL hash mismatch")
                expected = self.inputs.get((row["benchmark"], row["original_sha256"], row["schema_sha256"]))
                if expected is None or expected["original_sql"] != original:
                    raise ValueError("No exact manifest SQL/schema match")
                valid_sources = {_source_identity(source) for source in expected["input_sources"]}
                sources = row.get("input_sources", [])
                if not sources or any(_source_identity(source) not in valid_sources for source in sources):
                    raise ValueError("Input source provenance mismatch")
                provenance = row.get("provenance", {})
                if (provenance.get("mode") != "fresh_standalone" or provenance.get("system") != row["system"]
                        or not commits[row["system"]] or provenance.get("source_commit") != commits[row["system"]]):
                    raise ValueError("Missing/mismatched actual upstream source provenance")
                if isinstance(provenance.get("model_context"), dict):
                    provenance["model_context"] = _safe_model_context(provenance["model_context"])
                    if provenance["model_context"].get("transport") == SESSION_TRANSPORT:
                        if (type(provenance.get("session_completion_count")) is not int or provenance["session_completion_count"] < 1
                                or not provenance.get("session_completed_request_ids") or not provenance.get("session_workers")
                                or not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("session_completion_audit_sha256", "")))):
                            raise ValueError("Session completion audit evidence is missing")
                if databases is not None:
                    database = databases.get(row["dataset"])
                    if (database is None or database.get("benchmark", row["dataset"]) != row["benchmark"]
                            or dataset_configuration_sha256(database) != row.get("dataset_configuration_sha256")):
                        raise ValueError("Dataset configuration mismatch")
                key = _record_key(row)
                if key in self.records and self.records[key]["sql"] != sql:
                    ambiguous.add(key)
                self.records.setdefault(key, row)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                self.diagnostics.append({"line": number, "error": str(exc)})
        for key in ambiguous:
            self.records.pop(key, None)
            self.diagnostics.append({"status": "conflicting_fresh_outputs", "dataset": key[1], "original_sha256": key[3]})

    def lookup_for_dataset(self, system: str, original_sql: str, benchmark: str, dataset: str,
                           schema_sha256: str | None = None) -> dict | None:
        candidates = [row for key, row in self.records.items()
                      if key[:4] == (system, dataset, benchmark, exact_sha256(original_sql))
                      and row["original_sql"] == original_sql and (schema_sha256 is None or key[4] == schema_sha256)]
        if len(candidates) != 1:
            return None
        row = candidates[0]
        # No catalog assertion can become proof, even if an external provider
        # inserted formal/verified/equivalent fields into its provenance.
        safe = {key: row["provenance"][key] for key in
                ("system", "source_commit", "source_repository", "entry_point", "output_dir")
                if isinstance(row["provenance"].get(key), str)}
        for field in SESSION_PROVENANCE_FIELDS:
            if field in row["provenance"]:
                safe[field] = row["provenance"][field]
        if isinstance(safe.get("model_context"), dict) and safe["model_context"].get("transport") == SESSION_TRANSPORT:
            safe["model_context"] = {**safe["model_context"], "published_model_configuration_reproduced": False,
                                     "sampling_controls_reproduced": False, "usage_known": False}
        if "runtime_compatibility" in row["provenance"]:
            safe["runtime_compatibility"] = row["provenance"]["runtime_compatibility"]
        return {"sql": row["sql"], "provenance": {**safe, "mode": "fresh_standalone_catalog",
                "fresh_standalone": True, "catalog_path": str(self.path), "catalog_replay": True,
                "catalog_sha256": self.catalog_sha256,
                "dataset": dataset, "benchmark": benchmark, "original_sha256": row["original_sha256"],
                "schema_sha256": row["schema_sha256"],
                "input_sources": [{field: source[field] for field in SOURCE_FIELDS} for source in row["input_sources"]],
                "generation_context_sha256": row.get("generation_context_sha256"),
                "dataset_configuration_sha256": row.get("dataset_configuration_sha256"),
                "model_configuration_sha256": row.get("model_configuration_sha256"),
                "environment_sha256": row.get("environment_sha256"),
                "published_timings_used": False, "verified": False, "formal": False, "claim_used_as_proof": False}}


def run_bound_standalone(root: Path, request: dict, environment: dict) -> dict:
    """Give a gateway process its own dataset environment and real wrapper."""
    result_file = Path(request["output_dir"]) / "gateway_result.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.unlink(missing_ok=True)
    process = _run_process([sys.executable, "-m", "mcts_rewrite.fresh_baselines", "--worker"],
                           cwd=root, request=request, env=environment, timeout=float(request["timeout"]) + 60)
    # Gateway diagnostics contain no serialized environment/connection object.
    # Replace credential values if a provider nevertheless echoed one.
    credentials = {str(value) for name, value in environment.items() if value and
                   any(marker in name.upper() for marker in ("API_KEY", "PASSWORD", "SECRET", "TOKEN"))}
    for field in ("stdout", "stderr"):
        diagnostic = process.get(field, "")
        for credential in sorted(credentials, key=len, reverse=True):
            diagnostic = diagnostic.replace(credential, "[REDACTED]")
        _atomic_text(result_file.parent / f"gateway_{field}.log", diagnostic)
    if process["status"] == "ok" and result_file.exists():
        return json.loads(result_file.read_text())
    return {"status": "timeout" if process["status"] == "timeout" else "error",
            "error": "FreshGatewayError: inspect the gateway diagnostics and bounded upstream logs",
            "elapsed_s": process.get("elapsed_s")}


def _configuration_failure(row: dict) -> bool:
    if row.get("upstream_identity_fallback") is True:
        return True
    error = str(row.get("error", ""))
    return any(marker.lower() in error.lower() for marker in
               ("Missing upstream model configuration", "APIConnectionError", "AuthenticationError",
                "PermissionDeniedError", "RateLimitError", "invalid_api_key", "insufficient_quota", "Connection error",
                "ModuleNotFoundError", "ImportError", "FileNotFoundError", "FreshGatewayError"))


def generate_catalog(root: Path, manifest: Path, config: dict, catalog: Path, *, systems=SYSTEMS,
                     datasets=None, query_ids=None, limit=None, timeout=600, threads=1,
                     retry_failed=False, environment=None) -> dict:
    root, catalog = Path(root).resolve(), Path(catalog).resolve()
    environment = dict(os.environ if environment is None else environment)
    if (limit is not None and limit < 1) or threads < 1 or timeout <= 0:
        raise ValueError("Limit, threads, and timeout must be positive")
    groups = input_groups(root, _path(root, manifest), query_ids, limit)
    databases = config["databases"]
    selected_datasets = list(datasets or databases)
    if any(name not in databases for name in selected_datasets):
        raise ValueError("Unknown dataset selection")
    if any(system not in SYSTEMS for system in systems):
        raise ValueError("Unknown upstream system")
    planned_keys = {(system, dataset, group["benchmark"], group["original_sha256"], group["schema_sha256"])
                    for system in systems for dataset in selected_datasets for group in groups
                    if databases[dataset].get("benchmark", dataset) == group["benchmark"]}
    if not planned_keys:
        raise ValueError("No manifest inputs match the selected systems/datasets")
    existing = _read_jsonl(catalog) if catalog.exists() else []
    records = {_record_key(row): row for row in existing}
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "catalog": str(catalog),
              "manifest_sha256": file_sha256(_path(root, manifest)), "executed": 0, "resumed": 0,
              "blocked": [], "serial_execution": True, "threads_per_child": threads,
              "no_saved_output_fallback": True, "planned_executions": len(planned_keys)}
    probes = {}
    for system in systems:
        missing = missing_model_configuration(system, environment)
        # Invalid endpoint values should produce one clear preflight block,
        # rather than an exception while preparing a public model fingerprint.
        settings = model_configuration(system, environment) if not missing else {"unconfigured_system": system}
        system_stopped = False
        for dataset in selected_datasets:
            database = databases[dataset]
            benchmark = database.get("benchmark", dataset)
            matching = [group for group in groups if group["benchmark"] == benchmark]
            if not matching:
                continue
            bound = bound_database_environment(database, environment, timeout_ms=config.get("timeout_ms", 30000), threads=threads)
            public_environment = {key: bound.get(key) for key in (*THREAD_VARIABLES, "JAVA_TOOL_OPTIONS", "PYTHONHASHSEED", "OPENAI_TIMEOUT")}
            lock = root / f"configs/upstreams/{system}.lock"
            public_environment["dependency_lock_sha256"] = file_sha256(lock) if lock.exists() else None
            public_environment["python"] = sys.version.split()[0]
            public_environment["bridge_sources_sha256"] = fingerprint({name: file_sha256(root / "mcts_rewrite" / name)
                    if (root / "mcts_rewrite" / name).is_file() else None
                    for name in ("adapters.py", "upstream_worker.py", "upstream_runtime.py", "calcite_worker.py", "fresh_baselines.py")})
            environment_hash = fingerprint(public_environment)
            dataset_hash = dataset_configuration_sha256(database)
            upstream_commit = _commit(root / "vendor" / ("LLM4Rewrite" if system == "rbot" else "QUITE"))
            context = fingerprint({"dataset": dataset, "dataset_configuration_sha256": dataset_hash,
                                   "environment_sha256": environment_hash, "model": settings,
                                   "system": system, "upstream_commit": upstream_commit})
            for group in matching:
                query_context = fingerprint({"context": context, "input_sources": group["input_sources"]})
                row = {key: value for key, value in group.items() if key != "schema_path"}
                row.update({"format_version": 1, "system": system, "dataset": dataset, "generation_context_sha256": query_context,
                            "dataset_configuration_sha256": dataset_hash, "environment_sha256": environment_hash,
                            "model_configuration_sha256": fingerprint(settings)})
                key = _record_key(row)
                previous = records.get(key)
                if previous and previous.get("generation_context_sha256") == query_context and (previous.get("status") == "ok" or not retry_failed):
                    report["resumed"] += 1
                    continue
                if missing or system_stopped:
                    if not any(item.get("system") == system for item in report["blocked"]):
                        report["blocked"].append({"system": system, "reason": "missing_model_configuration", "variables": missing})
                    continue
                if dataset not in probes:
                    try:
                        probes[dataset] = probe_read_only_database(bound)
                    except Exception as exc:
                        # Never serialize a connection string or exception body
                        # that could contain a provider-supplied secret value.
                        probes[dataset] = None
                        detail = str(exc) if isinstance(exc, ValueError) else "Database connection/catalog probe failed"
                        report["blocked"].append({"dataset": dataset, "reason": "read_only_database_preflight_failed",
                                                  "exception": type(exc).__name__, "detail": detail})
                if probes[dataset] is None:
                    continue
                output = catalog.parent / "runs" / system / dataset / fingerprint([key, query_context])
                execution_environment = bound_session_environment(bound, "run_" + fingerprint([key, query_context]))
                request = {"root": str(root), "system": system, "sql": group["original_sql"],
                           "schema_sql": _path(root, group["schema_path"]).read_text(), "output_dir": str(output),
                           "database": bound["PGDATABASE"], "timeout": timeout,
                           "hint_capable": database.get("pg_hint_plan") is True}
                report["current_execution"] = {"system": system, "dataset": dataset,
                        "original_sha256": group["original_sha256"],
                        "query_ids": [source["id"] for source in group["input_sources"]],
                        "started_at": datetime.now(timezone.utc).isoformat()}
                _atomic_text(catalog.with_suffix(".report.json"), json.dumps(report, indent=2))
                result = run_bound_standalone(root, request, execution_environment)
                report.pop("current_execution", None)
                report["executed"] += 1
                provenance = result.get("provenance", {})
                successful = (result.get("status") == "ok" and isinstance(result.get("sql"), str)
                              and bool(result["sql"].strip()) and not result.get("upstream_identity_fallback")
                              and provenance.get("system") == system and provenance.get("mode") == "fresh_standalone"
                              and upstream_commit and provenance.get("source_commit") == upstream_commit)
                safe_provenance = {field: provenance[field] for field in
                                   ("system", "mode", "source_commit", "source_repository", "entry_point", "output_dir")
                                   if field in provenance}
                for field in (*SESSION_PROVENANCE_FIELDS, "runtime_compatibility"):
                    if field in provenance:
                        safe_provenance[field] = provenance[field]
                safe_provenance.setdefault("model_context", model_context(system, environment))
                if environment.get("MCTS_MODEL_TRANSPORT") == SESSION_TRANSPORT:
                    successful = successful and provenance.get("session_completion_count", 0) > 0
                row.update({"status": "ok" if successful else result.get("status", "error"),
                            "sql": result["sql"] if successful else None, "elapsed_s": result.get("elapsed_s"),
                            "sql_sha256": exact_sha256(result["sql"]) if successful else None,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "provenance": {**safe_provenance, "output_dir": str(output), "verified": False,
                                           "formal": False, "published_timings_used": False},
                            "database_environment_sha256": fingerprint(probes[dataset])})
                if not successful:
                    row["status"] = "error" if row["status"] == "ok" else row["status"]
                    error_class = re.match(r"([A-Za-z_][A-Za-z_0-9]*(?:Error|Exception))\b", str(result.get("error", "")))
                    row["error_type"] = error_class.group(1) if error_class else "upstream_error"
                    if _configuration_failure(result):
                        system_stopped = True
                        report["blocked"].append({"system": system, "reason": "upstream_model_configuration_or_service_failed",
                                                  "first_failure_key": list(key)})
                records[key] = row
                write_catalog(catalog, list(records.values()))
                _atomic_text(catalog.with_suffix(".report.json"), json.dumps(report, indent=2))
                if system_stopped:
                    break
    write_catalog(catalog, list(records.values()))
    selected_records = [row for key, row in records.items() if key in planned_keys]
    report["statuses"] = dict(Counter(row.get("status", "unknown") for row in selected_records))
    report["catalog_statuses"] = dict(Counter(row.get("status", "unknown") for row in records.values()))
    report["not_executed"] = report["planned_executions"] - report["executed"] - report["resumed"]
    report["status"] = "blocked" if report["blocked"] else ("error" if any(row.get("status") != "ok" for row in selected_records) else "ok")
    _atomic_text(catalog.with_suffix(".report.json"), json.dumps(report, indent=2))
    return report


def _worker() -> None:
    request = json.load(sys.stdin)
    result = UpstreamStandalone(Path(request["root"]), request["system"], timeout=request["timeout"]).rewrite(
        request["sql"], request["schema_sql"], output_dir=Path(request["output_dir"]),
        database=request["database"], hint_capable=request.get("hint_capable", False))
    _atomic_text(Path(request["output_dir"]) / "gateway_result.json", json.dumps(result, indent=2))


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Use scripts/generate_fresh_baselines.py")
    _worker()
