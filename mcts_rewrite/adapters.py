"""Real upstream operators and auditable model exchange for hybrid search.

Candidates are proposals, never equivalence certificates.  The search controller
must verify each proposal against its original root before measuring it.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.request
import zipfile


def exact_sha256(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def plan_sha256(plan: dict) -> str:
    """Fingerprint the complete plan supplied for a finite critic estimate."""
    return exact_sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")))


def _read_json(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip())
        try:
            result = json.loads(clean)
            return result if isinstance(result, dict) else {}
        except (TypeError, ValueError):
            return {}
    return {}


class FileModel:
    """Read model-produced JSONL only for byte-identical original input SQL.

    Rows require ``original_sql`` and may supply ``original_sha256`` (checked),
    ``task``, ``rule_sequences``, ``candidates``, and ``provenance``.  A row
    without task is available for both proposal tasks.  Critic rows additionally
    require ``task=cost_critic`` and ``plan_sha256`` of the complete input plan.
    Nothing generated for
    one root or intermediate node is silently applied to a different input.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path).resolve()
        self._rows: dict[str, list[tuple[int, dict]]] = {}
        self.errors: list[dict] = []
        for line_no, line in enumerate(self.path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                original = row["original_sql"]
                if not isinstance(original, str):
                    raise ValueError("original_sql must be a string")
                digest = exact_sha256(original)
                if row.get("original_sha256", digest) != digest:
                    raise ValueError("original_sha256 does not match original_sql")
                if row.get("task") == "cost_critic" and not re.fullmatch(r"[0-9a-f]{64}", str(row.get("plan_sha256", ""))):
                    raise ValueError("cost_critic requires a canonical plan_sha256")
                self._rows.setdefault(digest, []).append((line_no, row))
            except (ValueError, KeyError, TypeError) as exc:
                self.errors.append({"line": line_no, "error": str(exc)})

    def supports_task(self, task: str, sql: str | None = None, plan: dict | None = None) -> bool:
        """Declare artifact task coverage without implying a live model call."""
        groups = self._rows.values() if sql is None else [self._rows.get(exact_sha256(sql), [])]
        for rows in groups:
            for _, row in rows:
                if sql is not None and row["original_sql"] != sql:
                    continue
                if task == "cost_critic" and plan is not None and row.get("plan_sha256") != plan_sha256(plan):
                    continue
                if row.get("task") == task or (row.get("task") is None and task in {"select_rule_sequences", "macro_rewrite"}):
                    return True
        return False

    def __call__(self, payload: dict) -> dict:
        digest = exact_sha256(payload["sql"])
        result: dict = {"candidates": [], "rule_sequences": [],
                        "provenance": {"provider": "file_model", "path": str(self.path),
                                       "original_sha256": digest, "lines": []}}
        for line_no, row in self._rows.get(digest, []):
            if row["original_sql"] != payload["sql"]:
                continue
            task = payload.get("task")
            if row.get("task") not in (None, task):
                continue
            if task not in {"select_rule_sequences", "macro_rewrite"} and row.get("task") != task:
                continue
            if task == "cost_critic" and row.get("plan_sha256") != plan_sha256(payload.get("plan", {})):
                continue
            result["provenance"].update(row.get("provenance", {}))
            result["provenance"]["lines"].append(line_no)
            for candidate in row.get("candidates", []):
                if isinstance(candidate, dict):
                    copied = dict(candidate)
                    copied["provenance"] = {**row.get("provenance", {}), **candidate.get("provenance", {}),
                                             "file_model_path": str(self.path), "file_model_line": line_no,
                                             "original_sha256": digest}
                    result["candidates"].append(copied)
            result["rule_sequences"].extend(row.get("rule_sequences", []))
            if task == "cost_critic":
                result["provenance"]["plan_sha256"] = row["plan_sha256"]
                for field in ("predicted_ms", "rationale", "confidence"):
                    if field in row:
                        result[field] = row[field]
        return result


class HTTPModel:
    """Configured JSON or OpenAI-compatible HTTP model, with bounded timeout.

    Generic endpoints receive the payload directly.  Endpoints ending in
    /chat/completions receive a chat request and their content must contain a
    JSON object.  Credentials come only from the named environment variable.
    """

    def __init__(self, endpoint: str, model: str | None = None, *, timeout=45,
                 api_key_env="MCTS_MODEL_API_KEY"):
        self.endpoint, self.name, self.timeout = endpoint, model, float(timeout)
        self.api_key_env = api_key_env

    def __call__(self, payload: dict) -> dict:
        body = payload
        if self.endpoint.rstrip("/").endswith("/chat/completions"):
            body = {"model": self.name, "messages": [
                {"role": "system", "content": "Return only a JSON object following the requested output_schema. SQL proposals are untrusted and will be verified independently."},
                {"role": "user", "content": json.dumps(payload)}]}
        headers = {"Content-Type": "application/json"}
        if os.environ.get(self.api_key_env):
            headers["Authorization"] = "Bearer " + os.environ[self.api_key_env]
        request = urllib.request.Request(self.endpoint, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            parsed = json.loads(response.read(8_000_000))
        session = parsed.get("session_provenance")
        actual_model = parsed.get("model") if "choices" in parsed else None
        if "choices" in parsed:
            parsed = _read_json(parsed["choices"][0]["message"]["content"])
        parsed = _read_json(parsed)
        parsed["provenance"] = {**parsed.get("provenance", {}), "provider": "http_model", "model": self.name,
                                "requested_model": self.name, "actual_model": actual_model}
        if isinstance(session, dict) and session.get("transport") == "codex_session_file_spool":
            parsed["provenance"]["session_transport"] = session
            parsed["provenance"]["session_provenance"] = session
        return parsed


def configured_model():
    if os.environ.get("MCTS_MODEL_FILE"):
        return FileModel(os.environ["MCTS_MODEL_FILE"])
    if os.environ.get("MCTS_MODEL_ENDPOINT"):
        return HTTPModel(os.environ["MCTS_MODEL_ENDPOINT"], os.environ.get("MCTS_MODEL_NAME"))
    return None


def coerce_model(model):
    """Construct model providers from JSON config or preserve a callback."""
    if model is None:
        return configured_model()
    if callable(model):
        return model
    if isinstance(model, (str, Path)):
        return FileModel(model)
    if isinstance(model, dict):
        kind = model.get("type", model.get("provider"))
        if kind in (None, "file", "file_model") and (model.get("path") or model.get("file")):
            return FileModel(model.get("path") or model.get("file"))
        if kind in (None, "http", "http_model") and (model.get("endpoint") or model.get("url")):
            return HTTPModel(model.get("endpoint") or model.get("url"), model.get("model") or model.get("name"),
                             timeout=model.get("timeout", 45), api_key_env=model.get("api_key_env", "MCTS_MODEL_API_KEY"))
    raise ValueError("model must be a callable, file path, or file/http provider configuration")


def _run_process(command: list[str], *, cwd: Path, request: dict | None = None,
                 timeout: float = 30, env: dict | None = None) -> dict:
    """Bound a complete subprocess tree, including a JVM or upstream tools."""
    start = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    try:
        stdout, stderr = process.communicate(None if request is None else json.dumps(request), timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return {"status": "timeout", "error": f"Subprocess exceeded {timeout:g}s", "elapsed_s": time.monotonic() - start}
    return {"status": "ok" if process.returncode == 0 else "error", "returncode": process.returncode,
            "stdout": stdout, "stderr": stderr[-8000:], "elapsed_s": time.monotonic() - start}


class CalciteBridge:
    """Invoke the actual pinned R-Bot Rewriter through an isolated JVM."""

    def __init__(self, root: Path, timeout: float = 30):
        self.root, self.timeout = Path(root).resolve(), float(timeout)
        self.source = self.root / "vendor/LLM4Rewrite"
        self.source_jar = self.source / "CalciteRewrite/out/artifacts/LearnedRewrite_jar/LearnedRewrite.jar"
        self._jar: Path | None = None
        self.jar_sha256: str | None = None

    def _prepare_jar(self) -> Path:
        if self._jar is not None:
            return self._jar
        digest = hashlib.sha256(self.source_jar.read_bytes()).hexdigest()
        target = self.root / ".cache/mcts_rewrite/calcite" / f"LearnedRewrite-{digest}.jar"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            # The distributed fat jar combines signed dependencies.  Their
            # stale signatures fail the JVM's manifest verification.  Preserve
            # the pinned source and strip only signatures in a derived copy.
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".jar", delete=False) as temporary:
                temp_path = Path(temporary.name)
            try:
                with zipfile.ZipFile(self.source_jar) as original, zipfile.ZipFile(temp_path, "w") as copy:
                    for item in original.infolist():
                        upper = item.filename.upper()
                        if upper.startswith("META-INF/") and upper.endswith((".SF", ".DSA", ".RSA", ".EC")):
                            continue
                        copy.writestr(item, original.read(item.filename))
                temp_path.replace(target)
            finally:
                temp_path.unlink(missing_ok=True)
        self._jar, self.jar_sha256 = target, digest
        return target

    def _call(self, operation: str, sql: str, schema_sql: str, **extra) -> dict:
        try:
            jar = self._prepare_jar()
            # This is the same CREATE TABLE segmentation used upstream.
            request = {"jar": str(jar), "operation": operation, "sql": sql,
                       "create_tables": [part for part in schema_sql.split(";") if part.strip()], **extra}
            execution = _run_process([sys.executable, "-m", "mcts_rewrite.calcite_worker"],
                                     cwd=self.root, request=request, timeout=self.timeout)
            if execution["status"] != "ok":
                return {key: value for key, value in execution.items() if key != "stdout"}
            for line in reversed(execution["stdout"].splitlines()):
                if line.startswith("MCTS_CALCITE_JSON="):
                    result = json.loads(line.split("=", 1)[1])
                    result["elapsed_s"] = execution["elapsed_s"]
                    result["provenance"] = {"engine": "rbot_calcite_java", "entry_point": "rewriter.Rewriter." + ("matchAllRules" if operation == "match" else "rewrite"),
                                            "source_jar_sha256": self.jar_sha256, "derived_jar_signatures_removed": True}
                    return result
            return {"status": "error", "error": "Calcite worker returned no protocol response", "stderr": execution["stderr"]}
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            return {"status": "unavailable", "error": str(exc)}

    def match(self, sql: str, schema_sql: str) -> dict:
        return self._call("match", sql, schema_sql)

    def rewrite(self, sql: str, schema_sql: str, rule_sequences: list[list[str]], rounds=1) -> dict:
        return self._call("rewrite", sql, schema_sql, rule_sequences=rule_sequences, rounds=rounds)


class HybridGenerator:
    def __init__(self, root: Path, model=None, include_a=True, include_b=True, use_rag=True):
        self.root, self.model = Path(root).resolve(), coerce_model(model)
        self.include_a, self.include_b, self.use_rag = include_a, include_b, use_rag
        self.calcite = CalciteBridge(self.root, timeout=float(os.environ.get("MCTS_CALCITE_TIMEOUT", "30")))
        self.last_diagnostics: list[dict] = []
        self._match_cache: dict[tuple[str, str], dict] = {}
        self._memory: dict[str, list[dict]] = {}

    def begin_query(self, original_sql: str, schema_sql: str) -> None:
        """Reset volatile proposal memory between independent query searches."""
        self._memory.clear()
        self.last_diagnostics = []

    def _model_call(self, payload: dict) -> dict:
        if self.model is None:
            return {}
        try:
            return _read_json(self.model(payload))
        except Exception as exc:
            self.last_diagnostics.append({"operator": payload["task"], "status": "model_error", "error": str(exc)})
            return {}

    def _prompt_sources(self, kind: str) -> list[dict]:
        paths = (["vendor/LLM4Rewrite/my_rewriter/prompts.py"] if kind == "A" else [
            "vendor/QUITE/src/Query_Rewriter/agent_definition.py",
            "vendor/QUITE/src/Rewrite_Middleware/Agent_Memory_Buffer/memory_buffer.py"])
        results = []
        for relative in paths:
            path = self.root / relative
            if path.exists():
                content = path.read_text()
                if kind == "A":
                    module = ast.parse(content)
                    selected = []
                    for node in module.body:
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"RAG_SELECT_ARRANGE_RULES_SYS_PROMPT", "SELECT_RULES_SYS_PROMPT", "ARRANGE_RULES_SYS_PROMPT"} for t in node.targets):
                            selected.append(ast.literal_eval(node.value))
                    content = "\n\n".join(selected)
                elif relative.endswith("agent_definition.py"):
                    # Extract actual upstream prompt templates without importing
                    # its model clients or executing f-string expressions.
                    templates = []
                    for node in ast.walk(ast.parse(content)):
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "prompt" for t in node.targets):
                            templates.append(ast.get_source_segment(content, node.value) or "")
                    content = "\n\n".join(templates[:5])
                results.append({"source": relative, "sha256": exact_sha256(path.read_text()), "text": content[:18000]})
        return results

    @staticmethod
    def _hint_capable(evidence: dict) -> bool:
        capabilities = evidence.get("capabilities", {})
        return (capabilities.get("pg_hint_plan") is True or capabilities.get("pg_hint_plan_installed") is True
                or os.environ.get("MCTS_PG_HINT_PLAN") == "1")

    def candidates(self, sql, schema_sql, plan, bottleneck, evidence, limit=4) -> list[dict]:
        self.last_diagnostics = []
        if limit <= 0:
            return []
        evidence = evidence or {}
        used_evidence = evidence if self.use_rag else {}
        common = {"sql": sql, "original_sha256": exact_sha256(sql), "schema_sql": schema_sql,
                  "plan": plan, "bottleneck": bottleneck, "evidence": used_evidence}
        proposed: list[dict] = []
        # Reserve room for both operator classes whenever both are enabled.
        a_limit = max(1, limit // 2) if self.include_b else limit
        if self.include_a:
            key = (exact_sha256(sql), exact_sha256(schema_sql))
            if key not in self._match_cache:
                self._match_cache[key] = self.calcite.match(sql, schema_sql)
            matched = self._match_cache[key]
            matches = matched.get("matches", [])
            self.last_diagnostics.append({"operator": "A", "status": matched.get("status"),
                                          "matched_rules": len(matches), "error": matched.get("error")})
            payload = {**common, "task": "select_rule_sequences", "upstream_prompts": self._prompt_sources("A"),
                       "rules": [{"name": row["name"], "type": row.get("type"), "plan_after": row.get("plan_after")} for row in matches],
                       "output_schema": {"rule_sequences": [["MATCHED_RULE_NAME", "MATCHED_RULE_NAME"]]}}
            selected = self._model_call(payload) if matches else {}
            allowed = {row["name"] for row in matches}
            sequences = [sequence for sequence in selected.get("rule_sequences", [])
                         if isinstance(sequence, list) and sequence and all(isinstance(rule, str) and rule in allowed for rule in sequence)]
            if sequences:
                rewritten = self.calcite.rewrite(sql, schema_sql, sequences[:a_limit])
                for sequence, row in zip(sequences, rewritten.get("rewrites", [])):
                    if row.get("sql"):
                        proposed.append({"sql": row["sql"], "action_class": "A", "action": "calcite_rule_sequence",
                                         "provenance": {**rewritten.get("provenance", {}), "rule_sequence": sequence,
                                                        "used_rules": row["rules"], "selection": "model_rag" if self.use_rag else "model_no_rag",
                                                        "model": selected.get("provenance", {}), "verified": False}})
            if not any(row["action_class"] == "A" for row in proposed):
                # The fallback orders actual matched Calcite rules.  It is not
                # called a fresh standalone R-Bot execution.
                context = json.dumps({"bottleneck": bottleneck, "evidence": used_evidence}).upper()
                def rank(row):
                    terms = row["name"].split("_")
                    return (-sum(term in context for term in terms if len(term) > 3), row["name"])
                for row in sorted(matches, key=rank)[:a_limit]:
                    if row.get("sql_after"):
                        proposed.append({"sql": row["sql_after"], "action_class": "A", "action": "calcite_matched_rule",
                                         "provenance": {**matched.get("provenance", {}), "rule_sequence": [row["name"]],
                                                        "selection": "deterministic_matched_rule_fallback", "standalone_rbot": False,
                                                        "fallback_reason": "no applicable model rule sequence", "verified": False}})
        if self.include_b:
            hint_capable = self._hint_capable(evidence)
            families = ["cte_folding", "cte_materialization", "decorrelation", "redundancy_elimination", "join_filter_pushdown"]
            if hint_capable:
                families.append("pg_hint_plan")
            payload = {**common, "task": "macro_rewrite", "upstream_prompts": self._prompt_sources("B"),
                       "memory": self._memory.get(exact_sha256(sql), []), "allowed_transformations": families,
                       "capabilities": {"pg_hint_plan": hint_capable},
                       "instruction": "Propose semantically equivalent PostgreSQL SELECT rewrites. Preserve duplicates, NULL semantics, types, order and LIMIT. Consider CTE folding/materialization, decorrelation, and eliminating redundant work. Only emit pg_hint_plan hints if capability is true. Permitted hint forms are Set(enable_hashjoin|enable_mergejoin|enable_nestloop on|off), HashJoin/MergeJoin/NestLoop(alias ...), SeqScan/IndexScan/IndexOnlyScan/BitmapScan(alias ...), and Leading(...). A planner hint should preserve every SQL token outside its hint comment. Return JSON; no CREATE/ALTER/DROP/SET statements. Proposals require independent verification.",
                       "output_schema": {"candidates": [{"sql": "SELECT ...", "action": "transformation_name"}]}}
            response = self._model_call(payload)
            for row in response.get("candidates", []):
                if not isinstance(row, dict) or not isinstance(row.get("sql"), str) or not row["sql"].strip():
                    continue
                if not hint_capable and ("/*+" in row["sql"] or row.get("action") == "pg_hint_plan"):
                    self.last_diagnostics.append({"operator": "B", "status": "unsupported_hint_capability"})
                    continue
                proposed.append({"sql": row["sql"], "action_class": "B", "action": row.get("action", "quite_macro_rewrite"),
                                 "provenance": {**response.get("provenance", {}), **row.get("provenance", {}),
                                                "operator": "quite_prompt_memory_adapter", "standalone_quite": False,
                                                "prompt_sources": [entry["source"] for entry in payload["upstream_prompts"]],
                                                "verified": False}})
            if not response.get("candidates"):
                self.last_diagnostics.append({"operator": "B", "status": "unavailable" if self.model is None else "no_model_candidates",
                                              "reason": "Class B requires a configured model or exact-input file artifact"})
        seen, result = {exact_sha256(sql)}, []
        for row in proposed:
            digest = exact_sha256(row["sql"])
            if digest not in seen:
                seen.add(digest)
                result.append(row)
        result = result[:limit]
        if self.include_b:
            self._memory[exact_sha256(sql)] = [{"sql_sha256": exact_sha256(row["sql"]), "action": row["action"], "verified": False} for row in result]
        return result


class UpstreamStandalone:
    """Fresh standalone upstream execution; never substitutes local heuristics.

    R-Bot delegates to its original test_utils.test (retrieval, model selection,
    arrangement and reflection).  QUITE delegates to run.py.  Both need their
    documented Python dependencies, DB and model configuration.  Missing setup
    is returned as an explicit failure and cannot become a baseline success.
    """

    def __init__(self, root: Path, system: str, timeout=600, python: str | None = None):
        if system not in {"rbot", "quite"}:
            raise ValueError("system must be rbot or quite")
        self.root, self.system, self.timeout = Path(root).resolve(), system, timeout
        isolated = self.root / f".venv-{system}/bin/python"
        self.python = python or (str(isolated) if isolated.exists() else sys.executable)

    def rewrite(self, sql: str, schema_sql: str, *, output_dir: Path, database: str | None = None,
                hint_capable=False) -> dict:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        # A previous successful output must never survive a failed fresh run.
        (output_dir / "standalone_result.json").unlink(missing_ok=True)
        request = {"system": self.system, "sql": sql, "schema_sql": schema_sql,
                   "output_dir": str(output_dir), "database": database,
                   "hint_capable": bool(hint_capable), "root": str(self.root)}
        # The wrapper itself only imports upstream modules in its child.
        environment = os.environ.copy()
        local_lib = self.root / ".local/postgres/lib"
        if local_lib.exists():
            environment["LD_LIBRARY_PATH"] = str(local_lib) + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
        process = _run_process([self.python, "-m", "mcts_rewrite.upstream_worker"], cwd=self.root,
                               request=request, timeout=self.timeout, env=environment)
        (output_dir / "standalone_request.json").write_text(json.dumps(request, indent=2))
        (output_dir / "standalone_stdout.log").write_text(process.pop("stdout", ""))
        (output_dir / "standalone_stderr.log").write_text(process.pop("stderr", ""))
        result_path = output_dir / "standalone_result.json"
        if process["status"] == "ok" and result_path.exists():
            result = json.loads(result_path.read_text())
            result["elapsed_s"] = process["elapsed_s"]
            result.setdefault("provenance", {}).update({"original_sha256": exact_sha256(sql),
                                                        "output_dir": str(output_dir)})
            result_path.write_text(json.dumps(result, indent=2))
            return result
        return {**process, "provenance": {"mode": "fresh_standalone", "system": self.system,
                                          "output_dir": str(output_dir)}}


# Preserve the agreed public import path.
from .baselines import BaselineRegistry  # noqa: E402
