import ast
import json
from pathlib import Path

import pytest

from mcts_rewrite.local_rbot_embeddings import DIMENSION, MODEL_REVISION, prepare_variant_runtime


def fixture_source(tmp_path):
    source = tmp_path / "vendor/LLM4Rewrite"
    (source / "rag").mkdir(parents=True)
    (source / "my_rewriter").mkdir()
    (source / "knowledge-base").mkdir()
    (source / "my_rewriter/config.py").write_text('''CACHE_PATH = "cache"
CASE_RULES_PATH = "stackoverflow-rewrite-rules-query-optimization.jsonl"
def init_llms(model_type="", load_model=True):
    if "open" in model_type:
        embed_dim = 1536
    else:
        embed_dim = 1536
    llm_choice = "unchanged"
    return {"EMBED_DIM": embed_dim, "choice": llm_choice}
''')
    (source / "my_rewriter/stackoverflow-rewrite-rules-query-optimization.jsonl").write_text('{"id":1}\n')
    (source / "rag/rag_gen.py").write_text("# unchanged native builder\n")
    original = tmp_path / ".cache/mcts_rewrite/rbot_runtime"
    (original / "rag").mkdir(parents=True)
    (original / "extraction.json").write_text(json.dumps({"archive_sha256": "pinned-archive"}))
    for name in ("stackoverflow-rewrite-rules-query-optimization.jsonl", "stackoverflow-rewrite-sql-templates-query-optimization.jsonl"):
        (original / "rag" / name).write_text("{}\n")
    return source, original


def test_local_runtime_copies_mutable_case_rules_and_preserves_native_llm_branch(tmp_path):
    source, _ = fixture_source(tmp_path)
    runtime = tmp_path / ".cache/mcts_rewrite/rbot_local_minilm"
    metadata = prepare_variant_runtime(tmp_path, runtime, tmp_path / "model", runtime / "embedding_cache")
    mutable = runtime / "my_rewriter/stackoverflow-rewrite-rules-query-optimization.jsonl"
    assert not mutable.is_symlink()
    mutable.write_text("updated only in derived runtime\n")
    assert (source / "my_rewriter/stackoverflow-rewrite-rules-query-optimization.jsonl").read_text() == '{"id":1}\n'
    module = ast.parse((runtime / "my_rewriter/config.py").read_text())
    initializer = next(node for node in module.body if isinstance(node, ast.FunctionDef))
    assert ast.unparse(initializer.body[-2]) == "llm_choice = 'unchanged'"
    assert f"embed_dim = {DIMENSION}" in ast.unparse(initializer)
    assert metadata["revision"] == MODEL_REVISION and metadata["mutable_case_rules_copied"] is True
    assert (runtime / "rag/rag_gen.py").read_text() == "# unchanged native builder\n"


def test_local_variant_cannot_overwrite_original_runtime_or_vendor(tmp_path):
    source, original = fixture_source(tmp_path)
    for target in (original, source / "new_index"):
        with pytest.raises(ValueError, match="cannot overwrite"):
            prepare_variant_runtime(tmp_path, target, tmp_path / "model", tmp_path / "cache")


def test_local_context_keeps_actual_chat_model_and_embedding_substitution_distinct(monkeypatch):
    from mcts_rewrite import local_rbot_embeddings
    from mcts_rewrite.upstream_runtime import model_context
    from mcts_rewrite.fresh_baselines import model_configuration, _safe_model_context
    source = {"provider": "local_huggingface_cpu", "model": local_rbot_embeddings.MODEL_ID,
              "revision": MODEL_REVISION, "text_embedding_dimension": 384, "index_combined_dimension": 835,
              "feature_dimension": 67, "max_length_wordpieces": 256, "weights_sha256": "a" * 64,
              "model_manifest_sha256": "b" * 64, "index_build_sha256": "c" * 64,
              "index_validation_sha256": "d" * 64, "available_locally": True, "substituted": True,
              "original_model": "text-embedding-3-small", "native_retrieval_code_unchanged": True}
    monkeypatch.setattr(local_rbot_embeddings, "local_embedding_context", lambda root, runtime: source)
    environment = {"RBOT_LOCAL_EMBEDDING_RUNTIME": "/unit-test-derived-runtime",
                   "MCTS_MODEL_TRANSPORT": "codex_session_file_spool"}
    context = model_context("rbot", environment)
    assert context["requested_role_models"] == {"rewrite": "gpt-4o"}
    assert context["actual_role_models"] == {"rewrite": "gpt-6.1-sol"}
    assert context["reasoning_effort"] == "xhigh"
    assert context["embedding_source"] == source
    assert _safe_model_context(context)["embedding_source"] == source
    assert model_configuration("rbot", environment)["embedding_model"] == local_rbot_embeddings.MODEL_ID
    original = model_context("rbot", {})["embedding_source"]
    assert original["model"] == "text-embedding-3-small" and original["index_combined_dimension"] == 3139


def test_original_runtime_mutable_case_cache_is_copied_without_vendor_changes(tmp_path, monkeypatch):
    from mcts_rewrite.upstream_runtime import prepare_rbot_runtime
    source, original = fixture_source(tmp_path)
    monkeypatch.delenv("RBOT_LOCAL_EMBEDDING_RUNTIME", raising=False)
    target = original / "my_rewriter/stackoverflow-rewrite-rules-query-optimization.jsonl"
    target.parent.mkdir()
    vendor = source / "my_rewriter/stackoverflow-rewrite-rules-query-optimization.jsonl"
    target.symlink_to(vendor)
    assert prepare_rbot_runtime(tmp_path) == original
    assert not target.is_symlink()
    target.write_text("runtime-only learned case rules\n")
    assert vendor.read_text() == '{"id":1}\n'
