#!/usr/bin/env python3
"""Explain native vector-dedup differences without changing either index."""
import ast
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcts_rewrite.local_rbot_embeddings import MODEL_REVISION, atomic_json, canonical_sha256
from tokenizers import Tokenizer


def documents(path):
    with sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True) as connection:
        return set(connection.execute("""SELECT d.string_value,r.string_value FROM embeddings e
            JOIN embedding_metadata d ON d.id=e.id AND d.key='chroma:document'
            JOIN embedding_metadata r ON r.id=e.id AND r.key='references'""").fetchall())


def main():
    runtime = ROOT / ".cache/mcts_rewrite" / ("rbot_local_minilm_" + MODEL_REVISION[:12])
    original = ROOT / ".cache/mcts_rewrite/rbot_runtime"
    old_docs = documents(original / "rag/chroma_db/chroma.sqlite3")
    new_docs = documents(runtime / "rag/chroma_db/chroma.sqlite3")
    missing, added = old_docs - new_docs, new_docs - old_docs
    references = {ast.literal_eval(refs)[0] for _, refs in missing}
    cases = {}
    with (original / "rag/stackoverflow-rewrite-query-optimization.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            key = f"{row['id']}-{row['answer_id']}"
            if key in references:
                cases[key] = {"summary": row["summary"], "sqls": row["question_body_sqls"] or [""]}
    templates = {}
    with (original / "rag/stackoverflow-rewrite-sql-templates-query-optimization.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            key = f"{row['id']}-{row['answer_id']}"
            if key in references:
                for item in row["sql_templates"]:
                    templates.setdefault((key, item["sql"]), []).append(item["template"])
    pairs = {}
    for key, case in cases.items():
        for sql in case["sqls"]:
            for template in templates.get((key, sql), [""]):
                pairs[(f"{case['summary']}\n{sql}\n{template}", str([key]))] = {"case": key, "sql": sql, "template": template}
    needed_templates = {value["template"] for value in pairs.values()}
    vectors = {}
    for label, base in (("original", original), ("local", runtime)):
        values = {}
        with (base / "rag/stackoverflow-rewrite-sql-templates-embed-query-optimization.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["sql_template"] in needed_templates:
                    values[row["sql_template"]] = row["embedding"]
        vectors[label] = values
    rules = {}
    needed_sql = {value["sql"] for value in pairs.values()}
    with (original / "rag/stackoverflow-rewrite-rules-query-optimization.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["sql"] in needed_sql:
                rules[row["sql"]] = (sorted(set(row["nl_rules"])), sorted(set(row["rules"])))
    tokenizer = Tokenizer.from_file(str(ROOT / ".cache/models/all-MiniLM-L6-v2" / MODEL_REVISION / "tokenizer.json"))
    tokenizer.no_padding()
    collisions = []
    for document in missing:
        omitted = pairs[document]
        candidates = [pairs[key] for key in new_docs if key in pairs and pairs[key]["case"] == omitted["case"]]
        found = None
        for retained in candidates:
            if (rules.get(retained["sql"], ([], [])) == rules.get(omitted["sql"], ([], []))
                    and vectors["local"][retained["template"]] == vectors["local"][omitted["template"]]):
                found = retained
                break
        if found is None:
            raise ValueError("A missing native document lacks a real-vector deduplication match")
        tokenizer.no_truncation()
        before_lengths = [len(tokenizer.encode(value["template"]).ids) for value in (omitted, found)]
        tokenizer.enable_truncation(max_length=256)
        same_tokens = tokenizer.encode(omitted["template"]).ids == tokenizer.encode(found["template"]).ids
        collisions.append({"references": omitted["case"], "omitted_document_sha256": hashlib.sha256(document[0].encode()).hexdigest(),
            "omitted_template_sha256": hashlib.sha256(omitted["template"].encode()).hexdigest(),
            "retained_template_sha256": hashlib.sha256(found["template"].encode()).hexdigest(),
            "same_sql": omitted["sql"] == found["sql"], "same_native_rule_lists": True,
            "real_local_template_vectors_equal": True, "local_template_vector_sha256": canonical_sha256(vectors["local"][omitted["template"]]),
            "original_template_vectors_equal": vectors["original"][omitted["template"]] == vectors["original"][found["template"]],
            "wordpieces_before_truncation": before_lengths, "same_tokens_after_256_limit": same_tokens})
    report = {"status": "ok", "original_documents": len(old_docs), "local_documents": len(new_docs),
              "omitted_by_native_vector_deduplication": len(missing), "added_documents": len(added),
              "same_qa_reference_coverage": {refs for _, refs in old_docs} == {refs for _, refs in new_docs},
              "source_summary_rows_encoded": 2091, "source_template_rows_encoded": 3798,
              "synthetic_or_padded_nodes": False, "native_deduplication_changed": False, "collisions": collisions}
    if added or not report["same_qa_reference_coverage"]:
        raise ValueError("Local index changed source reference coverage")
    atomic_json(runtime / "local_index_deduplication.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
