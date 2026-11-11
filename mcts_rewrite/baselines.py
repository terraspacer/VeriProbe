"""Provenance-aware replay and partial R-Bot rule-sequence reconstruction.

Published timing fields never enter this registry.  The runner must independently
verify and time every returned SQL statement on its own database.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


_TOKEN = re.compile(
    r"--[^\n]*(?:\n|$)|/\*[\s\S]*?\*/|"
    r"\$(?P<tag>[A-Za-z_][A-Za-z_0-9]*|)\$[\s\S]*?\$(?P=tag)\$|"
    r"(?:[eEbBxXnN]|[uU]&)?'(?:''|\\[\s\S]|[^'])*'|"
    r'(?:[uU]&)?"(?:""|[^"])*"|'
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|"
    r"[A-Za-z_\u0080-\uffff][\w$]*|"
    r"[~!@#%^&|`?+*/<>=:-]+|[^\s]",
    re.UNICODE,
)


def formatting_fingerprint(sql: str) -> str:
    """Ignore only spaces between lexical tokens and an optional final ';'.

    Identifier/keyword case, quotes, literal bytes, operator adjacency and all
    comments (including hints) remain significant.  This is deliberately less
    permissive than parsing and re-serializing an AST.
    """
    tokens = [match.group(0) for match in _TOKEN.finditer(sql)]
    if tokens and tokens[-1] == ";":
        tokens.pop()
    return hashlib.sha256(json.dumps(tokens, ensure_ascii=False).encode()).hexdigest()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _commit(directory: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def rbot_source_inputs(root: Path, benchmark: str) -> dict[str, dict]:
    """Mirror the exact source-input construction in upstream test.py."""
    directory = root / "vendor/LLM4Rewrite" / benchmark
    result = {}
    if benchmark == "calcite":
        path = directory / "calcite.jsonl"
        if path.exists():
            for line_no, line in enumerate(path.read_text().splitlines(), 1):
                obj = json.loads(line)
                names = sorted(row["name"] for row in obj.get("rewrites", []))
                if names:
                    result[names[0]] = {"sql": obj["input_sql"], "source_path": str(path), "line": line_no}
    elif directory.exists():
        for path in sorted(directory.glob("query*/*.sql")):
            # The 10-scale summaries contain variants 0 and 1, not 50-scale.
            if not re.search(r"_[01]\.sql$", path.name):
                continue
            text = re.sub(r"--.*\n", "", path.read_text())
            queries = [part.strip() + ";" for part in text.split(";") if part.strip()]
            for index, sql in enumerate(queries):
                name = path.stem if len(queries) == 1 else f"{path.stem}_{index}"
                result[name] = {"sql": sql, "source_path": str(path), "statement_index": index}
    return result


class BaselineRegistry:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.records: dict[tuple[str, str, str], dict] = {}
        self.rule_records: dict[tuple[str, str, str], dict] = {}
        self.diagnostics: list[dict] = []
        self._cache: dict[tuple[str, str, str], dict | None] = {}
        self._calcite = None
        self.source_commits = {"rbot": _commit(self.root / "vendor/LLM4Rewrite"),
                               "quite": _commit(self.root / "vendor/QUITE")}
        self._load()

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    def _load(self):
        for benchmark in ("tpch", "dsb", "calcite"):
            inputs = rbot_source_inputs(self.root, benchmark)
            supported = {formatting_fingerprint(row["sql"]) for row in inputs.values()}
            quite_input = self.root / f"vendor/QUITE/dataset/queries/{benchmark}_queries.json"
            if quite_input.exists():
                try:
                    supported.update(formatting_fingerprint(row["query"]) for row in json.loads(quite_input.read_text()) if isinstance(row, dict) and isinstance(row.get("query"), str))
                except (ValueError, TypeError) as exc:
                    self.diagnostics.append({"source_path": self._relative(quite_input), "error": str(exc)})
            for system, pattern in (("quite", f"QUITE_{benchmark}_*queries.json"),
                                    ("rbot", f"[Rr][Bb]ot_GPT4o_{benchmark}_*queries.json")):
                directory = self.root / f"vendor/QUITE/experiments_results/{benchmark}"
                for path in sorted(directory.glob(pattern)):
                    try:
                        rows = json.loads(path.read_text())
                    except (ValueError, TypeError) as exc:
                        self.diagnostics.append({"source_path": self._relative(path), "error": str(exc)})
                        continue
                    for index, row in enumerate(rows):
                        if not isinstance(row, dict):
                            continue
                        original, candidate = row.get("original_query"), row.get("rewritten_query")
                        if not isinstance(original, str) or not isinstance(candidate, str) or not candidate.strip():
                            continue
                        fingerprint = formatting_fingerprint(original)
                        if fingerprint not in supported:
                            self.diagnostics.append({"source_path": self._relative(path), "index": index,
                                                     "status": "unsupported_original_source"})
                            continue
                        key = (system, benchmark, fingerprint)
                        record = {"sql": candidate, "provenance": {
                            "system": system, "mode": "upstream_saved_sql_replay", "fresh_standalone": False,
                            "source_repository": "QUITE", "source_commit": self.source_commits["quite"],
                            "source_path": self._relative(path), "source_index": index, "source_id": row.get("id"),
                            "source_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "original_sha256": _sha(original), "original_sql": original,
                            "matching": "lexical_formatting_only_case_literals_identifiers_hints_preserved",
                            "upstream_equivalence_claim": row.get("equivalence"), "claim_used_as_proof": False,
                            "published_timings_used": False,
                            "upstream_model": "GPT4o" if system == "rbot" else "QUITE published configuration"}}
                        # Ambiguous saved results never get resolved by their
                        # published speedup or latency.  Stable path order wins.
                        if key in self.records and self.records[key]["sql"] != candidate:
                            self.diagnostics.append({"status": "duplicate_output_stable_first", "source_path": self._relative(path), "index": index})
                        else:
                            self.records.setdefault(key, record)
            summary = self.root / f"vendor/LLM4Rewrite/my_rewriter/logs/{benchmark}10.log"
            if summary.exists():
                for line_no, line in enumerate(summary.read_text().splitlines(), 1):
                    if not line.startswith("|"):
                        continue
                    cells = [value.strip() for value in line.strip("|").split("|")]
                    if len(cells) != 4 or not cells[3].startswith("["):
                        continue
                    try:
                        sequence = ast.literal_eval(cells[3])
                    except (ValueError, SyntaxError):
                        continue
                    if not isinstance(sequence, list) or not all(isinstance(rule, str) for rule in sequence):
                        continue
                    original = inputs.get(cells[0])
                    if original is None:
                        self.diagnostics.append({"source_path": self._relative(summary), "line": line_no, "status": "unresolved_template", "template": cells[0]})
                        continue
                    fingerprint = formatting_fingerprint(original["sql"])
                    self.rule_records[("rbot", benchmark, fingerprint)] = {
                        "original": original["sql"], "rule_sequence": sequence,
                        "provenance": {"system": "rbot", "mode": "reconstructed_saved_rule_sequence",
                                       "fresh_standalone": False, "published_output_sql": False,
                                       "source_repository": "LLM4Rewrite", "source_commit": self.source_commits["rbot"],
                                       "source_path": self._relative(summary), "source_line": line_no,
                                       "template": cells[0], "input_source_path": self._relative(Path(original["source_path"])),
                                       "original_sha256": _sha(original["sql"]),
                                       "rule_sequence": sequence, "published_timings_used": False,
                                       "coverage_limitation": "Only rows present in the upstream improvements table can be reconstructed; omitted rows remain unavailable"}}

    def lookup(self, system: str, original_sql: str, benchmark: str) -> dict | None:
        if system not in {"rbot", "quite"} or benchmark not in {"tpch", "dsb", "calcite"}:
            return None
        key = (system, benchmark, formatting_fingerprint(original_sql))
        if key in self.records:
            record = self.records[key]
            return {"sql": record["sql"], "provenance": {**record["provenance"], "lookup_original_sha256": _sha(original_sql)}}
        if key not in self.rule_records:
            return None
        if key not in self._cache:
            from .adapters import CalciteBridge
            if self._calcite is None:
                self._calcite = CalciteBridge(self.root)
            record = self.rule_records[key]
            schema_path = self.root / f"vendor/LLM4Rewrite/{benchmark}/create_tables.sql"
            result = self._calcite.rewrite(record["original"], schema_path.read_text(), [record["rule_sequence"]])
            rewrites = result.get("rewrites", [])
            if result.get("status") == "ok" and rewrites and rewrites[0].get("sql"):
                self._cache[key] = {"sql": rewrites[0]["sql"], "provenance": {
                    **record["provenance"], "calcite": result.get("provenance", {}),
                    "reconstruction_used_rules": rewrites[0].get("rules"), "reconstruction_verified": False}}
            else:
                self._cache[key] = None
                self.diagnostics.append({"system": system, "benchmark": benchmark, "status": "reconstruction_failed",
                                         "error": result.get("error"), "template": record["provenance"]["template"]})
        cached = self._cache[key]
        return None if cached is None else {"sql": cached["sql"], "provenance": {**cached["provenance"], "lookup_original_sha256": _sha(original_sql)}}

    def coverage(self) -> dict:
        return {system: {benchmark: {
            "saved_sql": sum(k[:2] == (system, benchmark) for k in self.records),
            "saved_rule_sequences": sum(k[:2] == (system, benchmark) for k in self.rule_records),
            "available_input_fingerprints": len({k for k in (*self.records, *self.rule_records) if k[:2] == (system, benchmark)})}
            for benchmark in ("tpch", "dsb", "calcite")} for system in ("rbot", "quite")}
