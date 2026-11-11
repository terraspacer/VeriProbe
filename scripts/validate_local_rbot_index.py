#!/usr/bin/env python3
"""Independently validate real local vectors and native retrieval compatibility."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcts_rewrite.local_rbot_embeddings import DIMENSION, MAX_LENGTH, atomic_json, make_embedding, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    build = json.loads((runtime / "local_index_build.json").read_text())
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      ANONYMIZED_TELEMETRY="False", OPENAI_API_KEY="offline-parity-test-no-model-api")
    import torch
    import numpy as np
    import httpx
    import chromadb
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Settings, VectorStoreIndex, StorageContext
    from llama_index.vector_stores.chroma import ChromaVectorStore
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    def no_network(*args, **kwargs):
        raise RuntimeError("Network calls are forbidden during local retrieval parity validation")
    httpx.Client.send = no_network
    httpx.AsyncClient.send = no_network
    embedding = make_embedding(build["model_directory"], build["cache_directory"], batch_size=16)
    Settings.embed_model = embedding
    examples, long_text = [], None
    for name, field in (("stackoverflow-rewrite-sql-templates-embed-query-optimization.jsonl", "sql_template"),
                        ("stackoverflow-rewrite-query-optimization.jsonl", "summary")):
        selected = 0
        with (runtime / "rag" / name).open() as handle:
            for line in handle:
                row = json.loads(line)
                if selected < 5:
                    examples.append(row[field])
                    selected += 1
                if long_text is None and len(embedding._model.tokenizer.encode(row[field], truncation=False)) > MAX_LENGTH:
                    long_text = row[field]
                if selected == 5 and long_text is not None:
                    break
    cached = embedding._embed(examples, prompt_name="query")
    recomputed = HuggingFaceEmbedding._embed(embedding, examples, prompt_name="query")
    differences = [float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(cached, recomputed)]
    if max(differences) > 1e-5:
        raise ValueError("Pinned CPU recomputation disagrees with cached corpus vectors")
    shapes = []
    model = embedding._model[0].model
    original_forward = model.forward
    def capture_forward(*args, **kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is not None:
            shapes.append(list(ids.shape))
        return original_forward(*args, **kwargs)
    # Sentence Transformers v6 calls the bound forward method directly, so
    # nn.Module pre-hooks do not observe this path.
    model.forward = capture_forward
    try:
        HuggingFaceEmbedding._embed(embedding, [long_text], prompt_name="query")
    finally:
        model.forward = original_forward
    if not shapes or any(shape[-1] != MAX_LENGTH for shape in shapes):
        raise ValueError("Actual encoder input shape disagrees with the declared 256-wordpiece limit: " + repr(shapes))
    collection = chromadb.PersistentClient(path=str(runtime / "rag/chroma_db")).get_collection("stackoverflow")
    if collection.count() != build["index_count"]:
        raise ValueError("Persisted native index count changed")
    stored = collection.get(limit=10, include=["embeddings", "documents", "metadatas"])
    self_matches = []
    for node_id, vector in zip(stored["ids"], stored["embeddings"]):
        if len(vector) != build["combined_dimension"] or not np.isfinite(vector).all():
            raise ValueError("Persisted combined vector is invalid")
        result = collection.query(query_embeddings=[vector.tolist()], n_results=10)
        distance = float(result["distances"][0][0])
        if abs(distance) > 1e-5:
            raise ValueError("Persisted real-vector self-match failed")
        self_matches.append({"node_id": node_id, "first_distance": distance, "self_in_top10": node_id in result["ids"][0]})
    os.chdir(runtime / "my_rewriter")
    sys.path.insert(0, str(runtime))
    from mcts_rewrite.adapters import CalciteBridge
    from mcts_rewrite.upstream_runtime import load_rbot_rewrite_wrapper
    load_rbot_rewrite_wrapper(ROOT / "vendor/LLM4Rewrite", CalciteBridge(ROOT)._prepare_jar())
    from rag import my_query_fusion_retriver as native
    original = native.gen_rewrites_from_rules
    sql = "SELECT empno FROM emp WHERE empno = 1;"
    # This is a retrieval unit fixture, not a model-generated rewrite/baseline.
    def fixture_rewrites(**kwargs):
        return {"nl": [{"rewrite": sql}], "calcite": []}, {"nl": [native.NL_RULES[0]], "calcite_normal": []}
    native.gen_rewrites_from_rules = fixture_rewrites
    try:
        dummy = SimpleNamespace(schema="CREATE TABLE emp (empno integer);", _queries={}, _verbose=False,
                                embed_dim=DIMENSION, _achat=lambda messages: None)
        bundles = native.MyQueryFusionRetriever._get_queries(dummy, sql)
    finally:
        native.gen_rewrites_from_rules = original
    if not bundles or any(len(bundle.embedding) != build["combined_dimension"] for bundle in bundles):
        raise ValueError("Native query feature/embedding concatenation failed")
    feature_count = len(native.NL_RULES) + len(native.NORMAL_RULES)
    query_vector = np.asarray(embedding.get_query_embedding(sql))
    template_vectors = [np.asarray(embedding.get_query_embedding(text)) for text in dummy._queries["sql_templates"]]
    if not template_vectors:
        template_vectors = [np.zeros(DIMENSION)]
    rules = [0.0] * feature_count
    rules[0] = 1.0
    for bundle, template in zip(bundles, template_vectors):
        expected = np.concatenate([query_vector, rules, template]) / math.sqrt(3)
        if not np.allclose(expected, bundle.embedding, atol=1e-7):
            raise ValueError("Native query normalization/concatenation differs")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=StorageContext.from_defaults(vector_store=vector_store), embed_model=embedding)
    retrieved = index.as_retriever(similarity_top_k=10).retrieve(bundles[0])
    if len(retrieved) != 10 or not all(node.node.metadata.get("references") for node in retrieved):
        raise ValueError("Native real-vector retrieval returned incomplete references")
    isolation = runtime / "isolation_before.json"
    before = json.loads((isolation if isolation.exists() else ROOT / ".runtime/local_embeddings/isolation_before.json").read_text())
    original_index_unchanged = sha256(ROOT / ".cache/mcts_rewrite/rbot_runtime/rag/chroma_db/chroma.sqlite3") == before["original_index_sha256"]
    vendor_case_rules_unchanged = sha256(ROOT / "vendor/LLM4Rewrite/my_rewriter/stackoverflow-rewrite-rules-query-optimization.jsonl") == before["vendor_case_rules_sha256"]
    if not original_index_unchanged or not vendor_case_rules_unchanged:
        raise ValueError("A protected original artifact changed")
    report = {"status": "ok", "real_cpu_recomputations": len(examples), "cached_recomputed_max_abs_difference": max(differences),
              "raw_long_input_wordpieces": len(embedding._model.tokenizer.encode(long_text, truncation=False)),
              "actual_encoder_input_shapes": shapes, "actual_truncation_verified": True,
              "persisted_index_count": collection.count(), "combined_dimension": build["combined_dimension"],
              "persisted_self_matches": self_matches, "native_query_bundles": len(bundles),
              "native_feature_concatenation_verified": True, "native_retrieved_references": len(retrieved),
              "retrieval_fixture_not_model_baseline": True, "native_baseline_workflow_executed": False,
              "original_index_unchanged": original_index_unchanged, "vendor_case_rules_unchanged": vendor_case_rules_unchanged,
              "model_api_requests": 0, "synthetic_vectors": False, "source_snapshots_sha256": before["source_snapshots_sha256"]}
    atomic_json(runtime / "local_index_validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
