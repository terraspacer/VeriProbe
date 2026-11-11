"""Fresh, configured upstream entry points, invoked by UpstreamStandalone.

This process is disposable and bounded by the parent process-group timeout.
No stand-in rewriting algorithm is used if upstream imports/configuration fail.
"""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile


def _require_configuration(names: list[str]) -> None:
    missing = []
    for name in names:
        value = os.environ.get(name, "").strip()
        if not value or value.startswith("[") or value.startswith("YOUR_"):
            missing.append(name)
        elif name.endswith("_URL") and not value.startswith(("http://", "https://")):
            missing.append(name)
    if missing:
        raise ValueError("Missing upstream model configuration: " + ", ".join(missing))


def _rbot(request: dict) -> dict:
    from .adapters import CalciteBridge
    from .upstream_runtime import prepare_rbot_runtime, load_rbot_rewrite_wrapper
    root = Path(request["root"])
    upstream = root / "vendor/LLM4Rewrite"
    output = Path(request["output_dir"])
    jar = CalciteBridge(root)._prepare_jar()
    runtime = prepare_rbot_runtime(root)
    os.chdir(runtime / "my_rewriter")
    sys.path.insert(0, str(runtime))
    # The released wrapper assumes a jar path inconsistent with its own driver
    # working directory.  Change only that runtime path in the upstream AST;
    # every method body and all retrieval/model/arrangement code remain upstream.
    load_rbot_rewrite_wrapper(upstream, jar)
    from my_rewriter import config
    if "open" not in os.environ.get("RBOT_MODEL_TYPE", ""):
        _require_configuration(["OPENAI_API_KEY"])
    cache = output / "rbot_cache"
    cache.mkdir(exist_ok=True)
    config.CACHE_PATH = str(cache)
    database = request.get("database") or os.environ.get("DB_NAME") or os.environ.get("PGDATABASE")
    if not database:
        raise ValueError("R-Bot standalone requires database or DB_NAME/PGDATABASE")
    (cache / f"{database}.jsonl").touch()
    model_args = config.init_llms(os.environ.get("RBOT_MODEL_TYPE", ""))
    if os.environ.get("MCTS_MODEL_TRANSPORT") == "codex_session_file_spool":
        from llama_index.core import Settings
        Settings.llm.timeout = float(os.environ.get("OPENAI_TIMEOUT", "600"))
    from my_rewriter.database import DBArgs, Database
    from my_rewriter.rag_retrieve import init_docstore
    from my_rewriter.test_utils import test
    db_args = DBArgs({"host": os.environ.get("DB_HOST", os.environ.get("PGHOST", "localhost")),
                      "port": int(os.environ.get("DB_PORT", os.environ.get("PGPORT", "5432"))),
                      "user": os.environ.get("DB_USER", os.environ.get("PGUSER", "")),
                      "password": os.environ.get("DB_PASSWORD", os.environ.get("PGPASSWORD", "")),
                      "dbname": database, "db": "postgresql"})
    logdir = output / "rbot_logs"
    logdir.mkdir(exist_ok=True)
    log_path = logdir / "exact_input.log"
    log_path.unlink(missing_ok=True)
    test("exact_input", request["sql"], request["schema_sql"], db_args, model_args,
         init_docstore(), str(logdir), RETRIEVER_TOP_K=10, CASE_BATCH=5, RULE_BATCH=10,
         REWRITE_ROUNDS=1, index="hybrid")
    rewrites = []
    for line in log_path.read_text().splitlines():
        if "root INFO Rewrite Execution Results" in line:
            rewrites.append(ast.literal_eval(line[line.index("{"):]))
    valid = [row for row in rewrites if row.get("output_sql") not in (None, "None") and row.get("output_cost", -1) >= 0]
    if not valid:
        return {"status": "error", "error": "Upstream R-Bot emitted no executable rewrite", "rewrites": rewrites}
    best = min(valid, key=lambda row: row["output_cost"])
    # Match the upstream standalone analyzer's cost/subquery selection rule.
    db = Database(db_args)
    try:
        input_cost = db.cost_estimation(request["sql"])
    finally:
        if getattr(db, "conn", None):
            db.conn.close()
    use_output = input_cost < 0 or best["output_cost"] <= input_cost or "SUB_QUERY_TO_CORRELATE" in str(best.get("used_rules", []))
    return {"status": "ok", "sql": best["output_sql"] if use_output else request["sql"],
            "rewrites": rewrites, "selection": "upstream_cost_and_subquery_rule",
            "provenance": {"system": "rbot", "mode": "fresh_standalone", "entry_point": "my_rewriter.test_utils.test",
                           "original_jar_path_runtime_adaptation": str(jar), "published_timings_used": False,
                           "verified": False}}


def _quite(request: dict) -> dict:
    from .upstream_runtime import prepare_quite_runtime
    root, output = Path(request["root"]), Path(request["output_dir"])
    run_output = Path(tempfile.mkdtemp(prefix="quite-", dir=output))
    upstream = prepare_quite_runtime(root, run_output / "runtime")
    runtime_compatibility = json.loads((upstream / "runtime_compatibility.json").read_text())
    # Match libpq's conventional environment names when the caller did not
    # supply QUITE's DB_* names.  Explicit upstream settings still win.
    for name, pg_name in (("DB_HOST", "PGHOST"), ("DB_PORT", "PGPORT"),
                          ("DB_USER", "PGUSER"), ("DB_PASSWORD", "PGPASSWORD"),
                          ("DB_NAME", "PGDATABASE")):
        if name not in os.environ and pg_name in os.environ:
            os.environ[name] = os.environ[pg_name]
    from dotenv import load_dotenv
    load_dotenv(upstream / "config_file/.env")
    _require_configuration([f"{role}_{suffix}" for role in ("REASONING_MODEL", "DECISION_MODEL", "ASSISTANT_MODEL")
                            for suffix in ("API_KEY", "URL")] + ["REASONING_MODEL", "DECISION_MODEL", "ASSISTANT_MODEL"])
    inputs, schema = output / "input.json", output / "schema.sql"
    inputs.write_text(json.dumps([{"id": "exact_input", "query": request["sql"]}]))
    schema.write_text(request["schema_sql"])
    os.environ["PROJECT_ROOT"] = str(upstream)
    if request.get("database"):
        os.environ["DB_NAME"] = request["database"]
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    # Execute the original CLI, including its middleware and finite-state machine.
    sys.argv = [str(upstream / "run.py"), "--input_path", str(inputs), "--schema_file", str(schema),
                "--output_dir", str(run_output), "--enable_rewriter", "--save_rewriter_logs"]
    if request.get("hint_capable"):
        sys.argv.append("--enable_recommender")
    # QUITE catches model exceptions and writes identity SQL after all workers
    # fail.  Preserve its console evidence, but do not count that failure as a
    # successful fresh rewrite merely because an output JSON file exists.
    console = io.StringIO()
    try:
        with redirect_stdout(console):
            runpy.run_path(str(upstream / "run.py"), run_name="__main__")
    finally:
        sys.stdout.write(console.getvalue())
    failure_markers = ("All the parallel reasoning workers failed",
                       "All the parallel verification workers failed")
    failed_stages = [marker for marker in failure_markers if marker in console.getvalue()]
    if failed_stages:
        return {"status": "error", "error": "QUITE exhausted all upstream workers: " + "; ".join(failed_stages),
                "upstream_identity_fallback": True,
                "provenance": {"system": "quite", "mode": "fresh_standalone",
                               "entry_point": "run.py", "upstream_output_dir": str(run_output),
                               "runtime_compatibility": runtime_compatibility,
                               "published_timings_used": False, "verified": False}}
    found = []
    paths = [run_output / ("recommended_hints.json" if request.get("hint_capable") else "rewritten_queries.json")]
    for path in paths:
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text())
        except ValueError:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("original_query") == request["sql"] and isinstance(row.get("rewritten_query"), str):
                found.append((path, row))
    if not found:
        return {"status": "error", "error": "Upstream QUITE emitted no exact-input rewritten SQL"}
    path, row = found[-1]
    return {"status": "ok", "sql": row["rewritten_query"],
            "provenance": {"system": "quite", "mode": "fresh_standalone", "entry_point": "run.py",
                           "output_path": str(path), "hint_recommender_enabled": request.get("hint_capable", False),
                           "runtime_compatibility_patch": "Honor original --enable_recommender flag around the otherwise unconditional hint stage",
                           "runtime_compatibility": runtime_compatibility,
                           "published_timings_used": False, "verified": False}}


def main():
    request = json.load(sys.stdin)
    try:
        result = _rbot(request) if request["system"] == "rbot" else _quite(request)
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                  "provenance": {"system": request["system"], "mode": "fresh_standalone"}}
    try:
        from .upstream_runtime import session_transport_provenance, SESSION_TRANSPORT
        transport = session_transport_provenance(request["system"])
        result.setdefault("provenance", {}).update(transport)
        if (result.get("status") == "ok" and os.environ.get("MCTS_MODEL_TRANSPORT") == SESSION_TRANSPORT
                and not transport.get("session_completion_count")):
            result.update({"status": "error", "error": "No actual session completion evidence exists for this upstream run"})
    except Exception as exc:
        result.update({"status": "error", "error": "Session provenance audit failed: " + type(exc).__name__})
    from .baselines import _commit
    source_dir = "LLM4Rewrite" if request["system"] == "rbot" else "QUITE"
    result.setdefault("provenance", {}).update({"source_commit": _commit(Path(request["root"]) / "vendor" / source_dir),
                                                "source_repository": source_dir})
    (Path(request["output_dir"]) / "standalone_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
