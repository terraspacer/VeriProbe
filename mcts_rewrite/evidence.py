"""Auditable retrieval from the actual upstream knowledge, isolated per query.

Only named upstream knowledge files and reviewed action patterns are read.
Raw benchmark queries/results, timing receipts, vector-store pickles, and model
answers to evaluation queries are not retrieval sources. This is local BM25,
not a claim to execute the upstream hosted embedding services.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any


_KNOWLEDGE_FILES = (
    ("rbot", "LLM4Rewrite/explain_rule/calcite_rewrite_rules_structured.jsonl"),
    ("rbot", "LLM4Rewrite/knowledge-base/rule_cluster_summaries_structured.jsonl"),
    ("quite", "QUITE/src/Rewrite_Middleware/Structured_Knowledge_Base/storage/knowledge_base.json"),
)
_STOP = {"a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "in", "is", "it",
         "of", "on", "or", "query", "select", "sql", "that", "the", "this", "to", "with"}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _template_sha256(sql: str) -> str | None:
    """Use the same literal-insensitive identity as reviewed feedback receipts."""
    try:
        import sqlglot
        from sqlglot import exp
        tree = sqlglot.parse_one(sql, read="postgres")
        for node in list(tree.find_all(exp.Literal)):
            node.replace(exp.Placeholder())
        for node in tree.find_all(exp.Identifier):
            if not node.args.get("quoted"):
                node.set("this", node.this.lower())
        return _sha256(tree.sql(dialect="postgres"))
    except Exception:
        return None


def _hashes(value: Any, *, required: bool = False) -> bool:
    return (isinstance(value, list) and (bool(value) or not required)
            and all(isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item) for item in value))


def _tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z][a-z0-9_]*", text.lower()) if word not in _STOP]


def _fingerprint(sql: str) -> str:
    try:
        import sqlglot
        tokens = sqlglot.Dialect.get_or_raise("postgres").tokenize(sql)
        while tokens and tokens[-1].token_type.name == "SEMICOLON":
            tokens.pop()
        preserved = {"STRING", "NUMBER", "IDENTIFIER", "BYTE_STRING", "RAW_STRING", "NATIONAL_STRING", "HEX_STRING", "BIT_STRING"}
        parts = [(token.token_type.name,
                  token.text if token.token_type.name in preserved else token.text.lower(),
                  token.comments) for token in tokens]
        return json.dumps(parts, ensure_ascii=False)
    except Exception:
        return re.sub(r"\s+", " ", sql).strip().rstrip(";")


class EvidenceStore:
    def __init__(self, root: Path | str | None = None):
        root = Path(root or Path(__file__).resolve().parents[1])
        self.root = root
        self.vendor = root / "vendor" if (root / "vendor").is_dir() else root
        self.documents: list[dict] = []
        self.load_errors: list[dict] = []
        self.action_patterns_rejected = 0
        self._load()
        self._frequencies = [Counter(_tokens(doc["text"])) for doc in self.documents]
        self._df = Counter(term for frequency in self._frequencies for term in frequency)
        self._average_length = sum(sum(f.values()) for f in self._frequencies) / max(1, len(self.documents))
        self.begin_query("", "")

    def _load(self) -> None:
        commits: dict[str, str | None] = {}
        for source, relative in _KNOWLEDGE_FILES:
            path = self.vendor / relative
            if not path.is_file():
                self.load_errors.append({"source": source, "path": str(path), "error": "knowledge file missing"})
                continue
            try:
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                entries = [json.loads(line) for line in raw.decode().splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(raw)
                if isinstance(entries, dict):
                    entries = [entries]
                if source not in commits:
                    try:
                        commits[source] = subprocess.check_output(
                            ["git", "-C", str(self.vendor / relative.split("/")[0]), "rev-parse", "HEAD"],
                            stderr=subprocess.DEVNULL, text=True, timeout=2).strip()
                    except (OSError, subprocess.SubprocessError):
                        commits[source] = None
                for position, entry in enumerate(entries, 1):
                    if not isinstance(entry, dict):
                        continue
                    # Retain the upstream conditions and caveats, not just a synthetic rule label.
                    if source == "rbot":
                        content = {key: entry[key] for key in ("name", "conditions", "transformations", "rewrite_rules_structured") if key in entry}
                    else:
                        content = {key: entry[key] for key in ("group", "question_description", "question_query_example", "answer_summary", "answer_query_example") if key in entry}
                    text = json.dumps(content, ensure_ascii=False)
                    entry_hash = hashlib.sha256(text.encode()).hexdigest()
                    self.documents.append({
                        "id": str(entry.get("id", entry.get("name", entry.get("index", position)))),
                        "source": source, "text": text,
                        "provenance": {"path": str(path), "entry": position,
                                       "sha256": digest, "entry_sha256": entry_hash,
                                       "source_commit": commits[source], "corpus_role": "upstream_knowledge"},
                        "_examples": [entry[key] for key in ("question_query_example", "answer_query_example") if isinstance(entry.get(key), str)],
                    })
            except (OSError, ValueError, TypeError) as exc:
                self.load_errors.append({"source": source, "path": str(path), "error": str(exc)})
        self._load_action_patterns()

    def _load_action_patterns(self) -> None:
        """Index reviewed guidance, never the timing or evaluation receipts."""
        path = self.root / "prompts/action_patterns.jsonl"
        if not path.is_file():
            return
        try:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            lines = raw.decode().splitlines()
        except (OSError, UnicodeError) as exc:
            self.load_errors.append({"source": "action_patterns", "path": str(path), "error": str(exc)})
            return
        for position, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                validation, leakage = entry.get("validation", {}), entry.get("leakage", {})
                receipts = validation.get("receipts", [])
                scope = leakage.get("scope")
                if (entry.get("status") != "validated" or validation.get("status") != "validated"
                        or entry.get("provenance", {}).get("corpus_role") != "validated_action_pattern"
                        or validation.get("requires_root_equivalence_gate") is not True
                        or validation.get("timing_scope") != "search_training_only"
                        or not isinstance(receipts, list) or not receipts
                        or any(not isinstance(receipt, dict) or receipt.get("phase") != "search"
                               or not all(_hashes([receipt.get(key)], required=True) for key in
                                          ("receipt_sha256", "query_sha256", "template_sha256", "schema_sha256"))
                               for receipt in receipts)
                        or scope not in ("template_disjoint", "transductive_query_only")
                        or not all(_hashes(leakage.get(key)) for key in
                                   ("exclude_query_sha256", "exclude_template_sha256", "allow_query_sha256"))
                        or not _hashes(leakage.get("allow_schema_sha256"), required=True)
                        or (scope == "transductive_query_only" and not leakage["allow_query_sha256"])):
                    raise ValueError("action pattern lacks reviewed search-only validation or safe retrieval scope")
                if any(receipt["query_sha256"] not in leakage["exclude_query_sha256"]
                       or receipt["template_sha256"] not in leakage["exclude_template_sha256"]
                       or receipt["schema_sha256"] not in leakage["allow_schema_sha256"]
                       for receipt in receipts):
                    raise ValueError("action pattern retrieval scope does not cover its validation sources")
                if not entry.get("id") or entry.get("action_class") not in ("A", "B"):
                    raise ValueError("action pattern identity or action class missing")
                content = {key: entry[key] for key in
                           ("conditions", "transformations", "safety_checks", "trigger") if key in entry}
                if not content.get("conditions") or not content.get("transformations"):
                    raise ValueError("action pattern guidance missing")
                text = json.dumps(content, ensure_ascii=False)
                # Store just immutable pointers and scope. Raw receipts may carry
                # observations and must never become retrieval or critic input.
                pointer_fields = ("receipt_sha256", "raw_path", "raw_sha256", "run_hash", "task_id",
                                  "candidate_trace_index", "candidate_sql_sha256", "query_sha256",
                                  "template_sha256", "schema_sha256", "source_commit")
                self.documents.append({
                    "id": str(entry["id"]), "source": "action_patterns", "text": text,
                    "provenance": {"path": str(path), "entry": position, "sha256": digest,
                                   "entry_sha256": _sha256(json.dumps(entry, sort_keys=True, ensure_ascii=False)),
                                   "corpus_role": "validated_action_pattern", "feedback_scope": scope,
                                   "validation_receipts": [{key: receipt[key] for key in pointer_fields if key in receipt}
                                                           for receipt in receipts],
                                   "validation_semantics_scope": validation.get("semantics_scope"),
                                   "metrics_loaded": False, "requires_root_equivalence_gate": True},
                    "_examples": [], "_leakage": leakage,
                })
            except (ValueError, TypeError, AttributeError) as exc:
                self.action_patterns_rejected += 1
                self.load_errors.append({"source": "action_patterns", "path": str(path),
                                         "entry": position, "error": str(exc)})

    def begin_query(self, original_sql: str, schema_sql: str = "") -> None:
        """QUITE-style volatile memory is never shared between benchmark entries."""
        self._example_fingerprints: dict[int, list[str]] = {}
        self.original_sql = original_sql
        self.original_fingerprint = _fingerprint(original_sql) if original_sql else ""
        self.original_sha256 = _sha256(original_sql)
        self.template_sha256 = _template_sha256(original_sql) if original_sql else None
        self.schema_sha256 = _sha256(schema_sql)
        self.memory = {"initial_sql": original_sql,
                       "schema_sha256": self.schema_sha256,
                       "optimization_advice": [], "attempts": [],
                       "scope": "current_query_only", "evaluation_observations": False,
                       "upstream_memory_reference": "QUITE/src/Rewrite_Middleware/Agent_Memory_Buffer/memory_buffer.py"}

    def record_attempt(self, sql: str, status: str, verification: dict | None = None,
                       bottleneck: dict | None = None, **_: Any) -> None:
        self.memory["attempts"].append({"sql": sql, "status": status,
            "verification": {key: (verification or {}).get(key) for key in ("accepted", "proof", "reason")},
            "bottleneck": {key: (bottleneck or {}).get(key) for key in ("kind", "node_type", "relation")}})
        self.memory["attempts"] = self.memory["attempts"][-24:]

    def retrieve(self, sql: str, plan: dict | None = None, bottleneck: dict | None = None,
                 limit: int = 6) -> dict:
        del plan  # Do not stringify full plans or measured latency into retrieval queries.
        detail = bottleneck or {}
        kind = detail.get("kind", "")
        expansion = {"nested_loop": "join nested loop correlated subquery exists decorrelation",
                     "sequential_scan": "predicate filter index sargable selection pushdown",
                     "hashed_subplan": "hashed subquery not in null aware anti join filter",
                     "correlated_subplan": "correlated subquery exists decorrelation aggregate join"}.get(kind, "")
        terms = set(_tokens(sql + " " + expansion + " " + str(detail.get("node_type", ""))))
        ranked: list[tuple[float, int]] = []
        excluded = 0
        scope_excluded = 0
        count = len(self.documents)
        for index, (doc, frequency) in enumerate(zip(self.documents, self._frequencies)):
            leakage = doc.get("_leakage")
            if leakage:
                allowed = self.schema_sha256 in leakage["allow_schema_sha256"]
                if leakage["scope"] == "transductive_query_only":
                    allowed = allowed and self.original_sha256 in leakage["allow_query_sha256"]
                else:
                    allowed = (allowed and self.template_sha256 is not None
                               and self.original_sha256 not in leakage["exclude_query_sha256"]
                               and self.template_sha256 not in leakage["exclude_template_sha256"])
                if not allowed:
                    scope_excluded += 1
                    continue
            # Knowledge examples that reproduce the evaluation input are excluded.
            if index not in self._example_fingerprints:
                self._example_fingerprints[index] = [_fingerprint(example) for example in doc["_examples"]]
            if self.original_fingerprint and self.original_fingerprint in self._example_fingerprints[index]:
                excluded += 1
                continue
            length = sum(frequency.values())
            score = 0.0
            for term in terms & frequency.keys():
                inverse = math.log(1 + (count - self._df[term] + 0.5) / (self._df[term] + 0.5))
                hits = frequency[term]
                score += inverse * hits * 2.5 / (hits + 1.5 * (0.25 + 0.75 * length / max(1, self._average_length)))
            if score > 0:
                ranked.append((score, index))
        ranked.sort(key=lambda item: (-item[0], self.documents[item[1]]["source"], item[1]))
        # Preserve complementary sources when each has relevant evidence.
        selected: list[tuple[float, int]] = []
        sources = [source for source in ("rbot", "quite", "action_patterns")
                   if any(self.documents[index]["source"] == source for _, index in ranked)]
        quota = max(1, limit // max(1, len(sources)))
        for source in sources:
            selected.extend([item for item in ranked if self.documents[item[1]]["source"] == source][:quota])
        chosen = {index for _, index in selected}
        selected.extend(item for item in ranked if item[1] not in chosen)
        selected = selected[:max(0, limit)]
        documents = [{key: value for key, value in self.documents[index].items() if not key.startswith("_")} | {"score": round(score, 8)}
                     for score, index in selected]
        scopes = sorted({doc["provenance"]["feedback_scope"] for doc in documents
                         if doc["source"] == "action_patterns"})
        return {"documents": documents, "memory": json.loads(json.dumps(self.memory)),
                "retrieval": {"method": "local_bm25_source_balanced", "corpus_documents": count,
                              "sources": sorted({doc["source"] for doc in documents}),
                              "benchmark_results_loaded": False, "exact_input_examples_excluded": excluded,
                              "action_patterns_loaded": sum(doc["source"] == "action_patterns" for doc in self.documents),
                              "action_patterns_rejected": self.action_patterns_rejected,
                              "action_pattern_scope_excluded": scope_excluded,
                              "action_pattern_metrics_loaded": False, "feedback_scopes": scopes,
                              "transductive_patterns_used": "transductive_query_only" in scopes,
                              "load_errors": self.load_errors}}
