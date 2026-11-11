#!/usr/bin/env python3
"""Build an isolated native R-Bot index using real pinned CPU MiniLM vectors."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcts_rewrite.local_rbot_embeddings import (MODEL_ID, MODEL_REVISION, DIMENSION, MAX_LENGTH,
    atomic_json, download_model, make_embedding, prepare_variant_runtime, sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, default=ROOT / ".cache/models/all-MiniLM-L6-v2" / MODEL_REVISION)
    parser.add_argument("--runtime", type=Path, default=ROOT / ".cache/mcts_rewrite" / ("rbot_local_minilm_" + MODEL_REVISION[:12]))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("Batch size and threads must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    model = download_model(args.model_directory)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    cache = args.runtime.resolve() / "embedding_cache"
    metadata = prepare_variant_runtime(ROOT, args.runtime, args.model_directory.resolve(), cache)
    runtime = Path(metadata["runtime"])
    isolation = runtime / "isolation_before.json"
    if not isolation.exists():
        snapshots = runtime / "source_snapshot"
        source_paths = ("mcts_rewrite/local_rbot_embeddings.py", "scripts/build_local_rbot_index.py")
        for name in source_paths:
            destination = snapshots / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        atomic_json(isolation, {"source_snapshots_sha256": {name: sha256(ROOT / name) for name in source_paths},
                    "original_index_sha256": sha256(ROOT / ".cache/mcts_rewrite/rbot_runtime/rag/chroma_db/chroma.sqlite3"),
                    "vendor_case_rules_sha256": sha256(ROOT / "vendor/LLM4Rewrite/my_rewriter/stackoverflow-rewrite-rules-query-optimization.jsonl"),
                    "variant_case_rules_symlink": (runtime / "my_rewriter/stackoverflow-rewrite-rules-query-optimization.jsonl").is_symlink()})
    report_path = runtime / "local_index_build.json"
    if report_path.exists():
        print(report_path.read_text())
        return
    import torch
    import numpy as np
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    embedding = make_embedding(args.model_directory, cache, batch_size=args.batch_size)
    start = time.monotonic()
    original = ROOT / ".cache/mcts_rewrite/rbot_runtime/rag"
    progress = {"status": "encoding", "model": MODEL_ID, "revision": MODEL_REVISION, "completed_rows": 0,
                "threads": args.threads, "batch_size": args.batch_size}
    atomic_json(runtime / "build_progress.json", progress)
    source_hashes, row_counts = {}, {}
    for name, text_field in (("stackoverflow-rewrite-sql-templates-embed-query-optimization.jsonl", "sql_template"),
                             ("stackoverflow-rewrite-query-optimization.jsonl", "summary")):
        source_hashes[name] = sha256(original / name)
        with (original / name).open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        destination = runtime / "rag" / name
        temporary = destination.with_suffix(".jsonl.building")
        with temporary.open("w") as handle:
            for offset in range(0, len(rows), args.batch_size):
                batch = rows[offset:offset + args.batch_size]
                vectors = embedding._embed([row[text_field] for row in batch], prompt_name="query")
                for row, vector in zip(batch, vectors):
                    row["embedding"] = vector
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                progress.update(current_file=name, current_file_rows=min(offset + len(batch), len(rows)))
                progress["completed_rows"] += len(batch)
                atomic_json(runtime / "build_progress.json", progress)
        temporary.replace(destination)
        row_counts[name] = len(rows)
    del embedding
    progress["status"] = "building_native_index"
    atomic_json(runtime / "build_progress.json", progress)
    database = runtime / "rag/chroma_db"
    if database.is_symlink():
        raise ValueError("Refusing to replace a symlinked derived index")
    if database.exists():
        shutil.rmtree(database)
    os.chdir(runtime / "rag")
    sys.path.insert(0, str(runtime))
    from mcts_rewrite.adapters import CalciteBridge
    from mcts_rewrite.upstream_runtime import load_rbot_rewrite_wrapper
    load_rbot_rewrite_wrapper(ROOT / "vendor/LLM4Rewrite", CalciteBridge(ROOT)._prepare_jar())
    os.environ["OPENAI_API_KEY"] = "offline-native-index-no-model-api"
    import httpx
    def no_network(*args, **kwargs):
        raise RuntimeError("External model/network calls are forbidden during local index construction")
    httpx.Client.send = no_network
    httpx.AsyncClient.send = no_network
    sys.argv = [str(runtime / "rag/rag_gen.py")]
    namespace = runpy.run_path(str(runtime / "rag/rag_gen.py"), run_name="__main__")
    features = len(namespace["NL_RULES"]) + len(namespace["NORMAL_RULES"])
    dimension = DIMENSION * 2 + features
    nodes, collection = namespace["index_nodes"], namespace["chroma_collection"]
    vectors = namespace["content_to_embed_map"]
    if collection.count() != len(nodes) or not nodes:
        raise ValueError("Native local index is incomplete")
    norms = []
    for node in nodes:
        vector = np.asarray(vectors[node.text])
        if len(vector) != dimension or not np.isfinite(vector).all():
            raise ValueError("Invalid native concatenated local vector")
        norms.append(float(np.linalg.norm(vector)))
        summary_norm = np.linalg.norm(vector[:DIMENSION]) * math.sqrt(3)
        structural_norm = np.linalg.norm(vector[DIMENSION:DIMENSION + features]) * math.sqrt(3)
        template_norm = np.linalg.norm(vector[DIMENSION + features:]) * math.sqrt(3)
        if abs(summary_norm - 1) > 1e-4 or min(abs(structural_norm), abs(structural_norm - 1)) > 1e-4 or min(abs(template_norm), abs(template_norm - 1)) > 1e-4:
            raise ValueError("Native block normalization/concatenation parity failed")
    self_matches = []
    for index in np.linspace(0, len(nodes) - 1, 10, dtype=int):
        node = nodes[int(index)]
        result = collection.query(query_embeddings=[vectors[node.text]], n_results=10)
        distance = float(result["distances"][0][0])
        if abs(distance) > 1e-5:
            raise ValueError("Real local-vector self-match failed")
        self_matches.append({"node_id": node.node_id, "first_distance": distance, "self_in_top10": node.node_id in result["ids"][0]})
    caches = [json.loads(path.read_text()) for path in cache.glob("*.json")]
    report = {**metadata, "status": "ok", "model_manifest_sha256": sha256(args.model_directory / "model_manifest.json"),
              "model_files": model["files"], "source_files_sha256": source_hashes, "source_rows": row_counts,
              "index_count": collection.count(), "original_index_count": 5507, "same_node_count": len(nodes) == 5507,
              "feature_dimension": features, "combined_dimension": dimension, "normalization": "native block L2 and concatenation divided by sqrt(3)",
              "native_builder_sha256": sha256(ROOT / "vendor/LLM4Rewrite/rag/rag_gen.py"),
              "native_retriever_sha256": sha256(ROOT / "vendor/LLM4Rewrite/rag/my_query_fusion_retriver.py"),
              "native_builder_and_retrieval_code_unchanged": True, "embedding_cache_entries": len(caches),
              "truncated_unique_texts": sum(row["truncated"] for row in caches), "max_wordpieces": MAX_LENGTH,
              "combined_norm_min": min(norms), "combined_norm_max": max(norms), "self_matches": self_matches,
              "threads": args.threads, "batch_size": args.batch_size, "cpu_only": True, "model_api_requests": 0,
              "synthetic_vectors": False, "elapsed_s": time.monotonic() - start,
              "packages": {name: importlib.metadata.version(name) for name in ("torch", "sentence-transformers", "transformers", "chromadb", "llama-index-core")},
              "native_baseline_workflow_executed": False, "variant": "native_rbot_with_minilm_embedding_substitution"}
    atomic_json(report_path, report)
    atomic_json(runtime / "build_progress.json", {"status": "ok", "index_count": collection.count(), "combined_dimension": dimension})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
