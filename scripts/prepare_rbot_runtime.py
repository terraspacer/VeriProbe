#!/usr/bin/env python3
"""Extract provided embeddings and optionally run the real upstream index builder.

The offline index build uses only saved embeddings.  A clearly fake API key
initializes the upstream clients, and HTTP requests are blocked during building.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcts_rewrite.adapters import CalciteBridge
from mcts_rewrite.upstream_runtime import prepare_rbot_runtime, load_rbot_rewrite_wrapper

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build-index", action="store_true")
parser.add_argument("--rebuild-index", action="store_true", help="Replace the derived index using saved embeddings")
args = parser.parse_args()
runtime = prepare_rbot_runtime(ROOT)
index_metadata = runtime / "index_build.json"
index_directory = runtime / "rag/chroma_db"
index_complete = index_metadata.exists() and (index_directory / "chroma.sqlite3").exists()
if (args.build_index and not index_complete) or args.rebuild_index:
    # The upstream builder assigns fresh random node IDs.  Re-running it on an
    # existing collection appends duplicate retrieval cases; restart an
    # interrupted derived build from empty and skip already completed builds.
    index_metadata.unlink(missing_ok=True)
    if index_directory.is_symlink():
        raise RuntimeError("Refusing to replace a symlinked index directory")
    if index_directory.exists():
        shutil.rmtree(index_directory)
    os.chdir(runtime / "rag")
    sys.path.insert(0, str(runtime))
    load_rbot_rewrite_wrapper(ROOT / "vendor/LLM4Rewrite", CalciteBridge(ROOT)._prepare_jar())
    os.environ["OPENAI_API_KEY"] = "offline-saved-embeddings-no-network"
    import httpx
    original_send = httpx.Client.send
    async_original_send = httpx.AsyncClient.send
    def no_network(*args, **kwargs):
        raise RuntimeError("Network model requests are forbidden during offline saved-embedding index construction")
    httpx.Client.send = no_network
    httpx.AsyncClient.send = no_network
    start = time.monotonic()
    sys.argv = [str(runtime / "rag/rag_gen.py")]
    namespace = runpy.run_path(str(runtime / "rag/rag_gen.py"), run_name="__main__")
    node_count = len(namespace["index_nodes"])
    collection_count = namespace["chroma_collection"].count()
    if not node_count or collection_count != node_count:
        raise RuntimeError(f"Incomplete saved-embedding index: {collection_count} records for {node_count} upstream nodes")
    index_metadata.write_text(json.dumps({"entry_point": "vendor/LLM4Rewrite/rag/rag_gen.py",
                                                         "source_embeddings_only": True, "live_model_requests": 0,
                                                         "node_count": node_count, "collection_count": collection_count,
                                                         "elapsed_s": time.monotonic() - start}, indent=2))
print(json.dumps({"runtime": str(runtime), "index_built": (runtime / "index_build.json").exists()}))
