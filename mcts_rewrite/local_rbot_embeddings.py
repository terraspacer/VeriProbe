"""Isolated, explicitly substituted real CPU embeddings for native R-Bot.

No existing runtime/index or vendor file is modified. The original native index
builder and query-fusion code keep their feature normalization and concatenation.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
WEIGHTS_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
DIMENSION = 384
MAX_LENGTH = 256
MODEL_FILES = ("model.safetensors", "config.json", "config_sentence_transformers.json", "modules.json",
               "sentence_bert_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "vocab.txt", "1_Pooling/config.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def download_model(directory: Path) -> dict:
    """Download only pinned public safetensors/config files, without HF auth."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in MODEL_FILES:
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
            request = urllib.request.Request(url, headers={"User-Agent": "ICDE1-pinned-public-model-download"})
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", delete=False) as handle:
                temporary = Path(handle.name)
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        while chunk := response.read(1024 * 1024):
                            handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            if name == "model.safetensors" and sha256(temporary) != WEIGHTS_SHA256:
                temporary.unlink(missing_ok=True)
                raise ValueError("Pinned safetensors checksum mismatch")
            temporary.replace(destination)
        files[name] = {"sha256": sha256(destination), "bytes": destination.stat().st_size}
    if files["model.safetensors"]["sha256"] != WEIGHTS_SHA256:
        raise ValueError("Local safetensors checksum mismatch")
    metadata = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "files": files,
                "dimension": DIMENSION, "max_length_wordpieces": MAX_LENGTH, "pooling": "attention_mask_mean",
                "normalize": True, "device": "cpu", "trust_remote_code": False, "local_files_only": True,
                "credential_used": False, "model_source": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
                "substitutes": "text-embedding-3-small", "substituted": True}
    atomic_json(directory / "model_manifest.json", metadata)
    return metadata


def make_embedding(model_directory: str | Path, cache_directory: str | Path, *, batch_size=16):
    """Actual HuggingFace CPU inference with exact-text/revision cache entries."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from pydantic import PrivateAttr
    import numpy as np

    model_directory, cache_directory = Path(model_directory).resolve(), Path(cache_directory).resolve()
    manifest = json.loads((model_directory / "model_manifest.json").read_text())
    if (manifest.get("revision") != MODEL_REVISION or manifest.get("model_id") != MODEL_ID
            or sha256(model_directory / "model.safetensors") != WEIGHTS_SHA256):
        raise ValueError("Pinned local embedding model identity mismatch")
    for name, record in manifest["files"].items():
        if sha256(model_directory / name) != record["sha256"]:
            raise ValueError("Pinned model/config file changed: " + name)
    identity = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "model_manifest_sha256": sha256(model_directory / "model_manifest.json"),
                "max_length": MAX_LENGTH, "normalize": True, "dimension": DIMENSION}
    cache_directory.mkdir(parents=True, exist_ok=True)

    class CachedEmbedding(HuggingFaceEmbedding):
        _cache_directory: Path = PrivateAttr()
        _identity: dict = PrivateAttr()
        _cache_statistics: dict = PrivateAttr()

        def __init__(self):
            super().__init__(model_name=str(model_directory), device="cpu", max_length=MAX_LENGTH,
                             normalize=True, embed_batch_size=batch_size, trust_remote_code=False,
                             local_files_only=True, show_progress_bar=False)
            self._cache_directory, self._identity = cache_directory, identity
            self._cache_statistics = {"hits": 0, "misses": 0, "truncated": 0}

        def _embed(self, inputs, prompt_name=None):
            if not all(isinstance(text, str) for text in inputs):
                raise ValueError("This local embedding variant supports text only")
            keys = [canonical_sha256({**self._identity, "prompt_name": prompt_name, "text": text}) for text in inputs]
            vectors, missing = {}, {}
            for index, (key, text) in enumerate(zip(keys, inputs)):
                path = self._cache_directory / (key + ".json")
                if path.exists():
                    record = json.loads(path.read_text())
                    if record.get("cache_key") != key or record.get("text_sha256") != hashlib.sha256(text.encode()).hexdigest():
                        raise ValueError("Local embedding cache identity mismatch")
                    vector = record["embedding"]
                    if len(vector) != DIMENSION or not np.isfinite(vector).all() or abs(np.linalg.norm(vector) - 1) > 1e-4:
                        raise ValueError("Invalid cached real embedding")
                    vectors[index] = vector
                    self._cache_statistics["hits"] += 1
                else:
                    missing.setdefault(key, {"text": text, "indices": []})["indices"].append(index)
            if missing:
                unique = list(missing.items())
                encoded = super()._embed([record["text"] for _, record in unique], prompt_name=prompt_name)
                for (key, record), vector in zip(unique, encoded):
                    vector = [float(value) for value in vector]
                    if len(vector) != DIMENSION or not np.isfinite(vector).all() or abs(np.linalg.norm(vector) - 1) > 1e-4:
                        raise ValueError("Actual local model emitted an invalid embedding")
                    tokens = len(self._model.tokenizer.encode(record["text"], add_special_tokens=True, truncation=False))
                    truncated = tokens > MAX_LENGTH
                    self._cache_statistics["misses"] += 1
                    self._cache_statistics["truncated"] += int(truncated)
                    atomic_json(self._cache_directory / (key + ".json"), {
                        "cache_key": key, **self._identity, "prompt_name": prompt_name,
                        "text_sha256": hashlib.sha256(record["text"].encode()).hexdigest(),
                        "embedding": vector, "input_tokens_before_truncation": tokens, "truncated": truncated,
                        "generated_by": "real_local_cpu_inference", "synthetic": False})
                    for index in record["indices"]:
                        vectors[index] = vector
            return [vectors[index] for index in range(len(inputs))]

    return CachedEmbedding()


def prepare_variant_runtime(root: Path, runtime: Path, model_directory: Path, cache_directory: Path) -> dict:
    """Create a separate runtime with mutable files copied and native code linked."""
    root, runtime = Path(root).resolve(), Path(runtime).resolve()
    original = root / ".cache/mcts_rewrite/rbot_runtime"
    if runtime == original or runtime.is_relative_to(root / "vendor") or runtime.is_symlink():
        raise ValueError("A local variant cannot overwrite the original runtime or vendor")
    source = root / "vendor/LLM4Rewrite"
    runtime.mkdir(parents=True, exist_ok=True)
    ownership = runtime / "variant_ownership.json"
    identity = {"variant": "rbot_local_minilm", "model_id": MODEL_ID, "revision": MODEL_REVISION,
                "source_archive_sha256": json.loads((original / "extraction.json").read_text())["archive_sha256"]}
    if ownership.exists() and json.loads(ownership.read_text()) != identity:
        raise ValueError("Existing derived runtime belongs to another variant")
    atomic_json(ownership, identity)
    for item in source.iterdir():
        if item.name in {"rag", "my_rewriter", ".git"}:
            continue
        destination = runtime / item.name
        if not destination.exists():
            destination.symlink_to(item, target_is_directory=item.is_dir())
    for directory in ("rag", "my_rewriter"):
        destination = runtime / directory
        destination.mkdir(exist_ok=True)
        for item in (source / directory).iterdir():
            if not item.is_file() or item.suffix == ".zip" or directory == "my_rewriter" and item.name == "config.py":
                continue
            target = destination / item.name
            if not target.exists():
                if directory == "my_rewriter" and item.suffix in {".json", ".jsonl"}:
                    shutil.copy2(item, target)
                else:
                    target.symlink_to(item)
    for name in ("stackoverflow-rewrite-rules-query-optimization.jsonl", "stackoverflow-rewrite-sql-templates-query-optimization.jsonl"):
        target = runtime / "rag" / name
        if not target.exists():
            target.symlink_to(original / "rag" / name)
    tree = ast.parse((source / "my_rewriter/config.py").read_text())
    tree.body.insert(0, ast.ImportFrom(module="mcts_rewrite.local_rbot_embeddings", names=[ast.alias(name="make_embedding")], level=0))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CASE_RULES_PATH":
                    node.value = ast.Constant(str(runtime / "my_rewriter/stackoverflow-rewrite-rules-query-optimization.jsonl"))
        if isinstance(node, ast.FunctionDef) and node.name == "init_llms":
            if not isinstance(node.body[0], ast.If):
                raise ValueError("Pinned upstream model initializer shape changed")
            local = ast.parse(f"if load_model:\n    Settings.embed_model = make_embedding({str(model_directory)!r}, {str(cache_directory)!r})\nembed_dim = {DIMENSION}").body
            node.body = local + node.body[1:]
    ast.fix_missing_locations(tree)
    configuration = runtime / "my_rewriter/config.py"
    configuration.write_text(ast.unparse(tree) + "\n")
    return {**identity, "runtime": str(runtime), "model_directory": str(model_directory), "cache_directory": str(cache_directory),
            "text_embedding_dimension": DIMENSION, "original_text_embedding_dimension": 1536,
            "config_source_sha256": sha256(source / "my_rewriter/config.py"), "config_runtime_sha256": sha256(configuration),
            "mutable_case_rules_copied": True, "source_runtime_unchanged": True}


def validated_variant_runtime(root: Path, runtime: Path) -> Path:
    """Fail closed if the explicitly selected local variant lacks real evidence."""
    root, runtime = Path(root).resolve(), Path(runtime).resolve()
    if runtime == root / ".cache/mcts_rewrite/rbot_runtime" or runtime.is_relative_to(root / "vendor"):
        raise ValueError("Local embedding activation cannot select the original/vendor runtime")
    build = json.loads((runtime / "local_index_build.json").read_text())
    validation = json.loads((runtime / "local_index_validation.json").read_text())
    if (build.get("status") != "ok" or validation.get("status") != "ok"
            or build.get("revision") != MODEL_REVISION or build.get("text_embedding_dimension") != DIMENSION
            or build.get("combined_dimension") != DIMENSION * 2 + build.get("feature_dimension", -1)
            or validation.get("persisted_index_count") != build.get("index_count")
            or not validation.get("actual_truncation_verified") or not validation.get("native_feature_concatenation_verified")
            or build.get("model_manifest_sha256") != sha256(Path(build["model_directory"]) / "model_manifest.json")
            or build.get("config_runtime_sha256") != sha256(runtime / "my_rewriter/config.py")
            or not (runtime / "rag/chroma_db/chroma.sqlite3").is_file()):
        raise ValueError("Local embedding/index validation does not match the selected runtime")
    return runtime


def local_embedding_context(root: Path, runtime: Path) -> dict:
    runtime = validated_variant_runtime(root, runtime)
    build = json.loads((runtime / "local_index_build.json").read_text())
    return {"provider": "local_huggingface_cpu", "model": MODEL_ID, "revision": MODEL_REVISION,
            "text_embedding_dimension": DIMENSION, "index_combined_dimension": build["combined_dimension"],
            "feature_dimension": build["feature_dimension"], "max_length_wordpieces": MAX_LENGTH,
            "weights_sha256": WEIGHTS_SHA256, "model_manifest_sha256": build["model_manifest_sha256"],
            "index_build_sha256": sha256(runtime / "local_index_build.json"),
            "index_validation_sha256": sha256(runtime / "local_index_validation.json"),
            "requires_live_query_embeddings": False, "available_via_session_bridge": False,
            "available_locally": True, "substituted": True, "original_model": "text-embedding-3-small",
            "native_retrieval_code_unchanged": True}
