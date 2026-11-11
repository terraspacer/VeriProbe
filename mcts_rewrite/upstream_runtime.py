"""Writable upstream runtime overlay; pinned vendor sources remain untouched."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import types
import zipfile

SESSION_TRANSPORT = "codex_session_file_spool"
SESSION_MODEL = "gpt-6.1-sol"
SESSION_EFFORT = "xhigh"
SESSION_PROVENANCE_FIELDS = ("model_context", "session_run_id", "session_spool_path", "session_completion_count",
    "session_completed_request_ids", "session_completion_audit_sha256", "session_workers", "session_bridge_source_sha256")


def model_context(system: str, environment: dict, audit: dict | None = None) -> dict:
    """Describe each model role and keep embedding provenance separate."""
    session = environment.get("MCTS_MODEL_TRANSPORT") == SESSION_TRANSPORT
    if system == "rbot":
        requested = {"rewrite": "gpt-3.5-turbo-0125" if "gpt3" in environment.get("RBOT_MODEL_TYPE", "") else "gpt-4o"}
        embeddings = {"provider": "released_openai_vectors_plus_live_query_embeddings", "model": "text-embedding-3-small",
                      "text_embedding_dimension": 1536, "index_combined_dimension": 3139,
                      "requires_live_query_embeddings": True, "available_via_session_bridge": False, "substituted": False}
        if environment.get("RBOT_LOCAL_EMBEDDING_RUNTIME"):
            from .local_rbot_embeddings import local_embedding_context
            root = Path(__file__).resolve().parents[1]
            selected = Path(environment["RBOT_LOCAL_EMBEDDING_RUNTIME"])
            embeddings = local_embedding_context(root, selected if selected.is_absolute() else root / selected)
    else:
        requested = {role.lower(): environment.get(role + "_MODEL", "") for role in ("REASONING", "DECISION", "ASSISTANT")}
        embeddings = {"provider": "local_bm25", "model": None, "requires_live_query_embeddings": False, "substituted": False}
    actual = {role: SESSION_MODEL if session else name for role, name in requested.items()}
    context = {"transport": SESSION_TRANSPORT if session else "configured_upstream_api",
               "requested_role_models": requested, "actual_role_models": actual,
               "actual_serving_model": SESSION_MODEL if session else (next(iter(set(actual.values()))) if len(set(actual.values())) == 1 else None),
               "reasoning_effort": SESSION_EFFORT if session else None,
               "model_substitution": any(actual[role] != requested[role] for role in requested),
               "published_model_configuration_reproduced": False if session else None,
               "sampling_controls_reproduced": False if session else None,
               "requested_sampling_options_not_enforced": (audit or {}).get("requested_sampling_options_not_enforced", []),
               "usage_known": False if session else None, "embedding_source": embeddings}
    return context


def session_transport_provenance(system: str, environment=None) -> dict:
    environment = os.environ if environment is None else environment
    if environment.get("MCTS_MODEL_TRANSPORT") != SESSION_TRANSPORT:
        return {"model_context": model_context(system, environment)}
    run_id, spool = environment.get("MCTS_SESSION_BRIDGE_RUN_ID", ""), environment.get("MCTS_SESSION_BRIDGE_SPOOL", "")
    records, sampling = [], set()
    source_hash = None
    if spool and run_id:
        directory = Path(spool)
        metadata = json.loads((directory / "server.json").read_text())
        if metadata.get("actual_serving_model") != SESSION_MODEL or metadata.get("reasoning_effort") != SESSION_EFFORT:
            raise ValueError("Session bridge serving model/effort mismatch")
        source_hash = metadata.get("bridge_source_sha256")
        for path in sorted((directory / "requests").glob("req_*/completion.json")):
            completion = json.loads(path.read_text())
            provenance = completion.get("session_provenance", {})
            if provenance.get("run_id") != run_id:
                continue
            content = completion["choices"][0]["message"]["content"]
            request = json.loads((path.parent / "request.json").read_text())
            request_hash = hashlib.sha256(json.dumps(request["request"], sort_keys=True, separators=(",", ":"),
                                                       ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            if (provenance.get("actual_serving_model") != SESSION_MODEL or provenance.get("reasoning_effort") != SESSION_EFFORT
                    or provenance.get("transport") != SESSION_TRANSPORT or provenance.get("request_sha256") != request_hash
                    or provenance.get("content_sha256") != hashlib.sha256(content.encode()).hexdigest()):
                raise ValueError("Session completion audit mismatch")
            records.append({field: provenance[field] for field in
                            ("request_id", "request_sha256", "content_sha256", "worker_id", "requested_model")})
            sampling.update(provenance.get("requested_sampling_options_not_enforced", []))
    audit = {"requested_sampling_options_not_enforced": sorted(sampling)}
    return {"model_context": model_context(system, environment, audit), "session_run_id": run_id,
            "session_spool_path": spool, "session_completion_count": len(records),
            "session_completed_request_ids": [record["request_id"] for record in records],
            "session_workers": sorted({record["worker_id"] for record in records}),
            "session_completion_audit_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
            "session_bridge_source_sha256": source_hash}


def prepare_rbot_runtime(root: Path) -> Path:
    root = Path(root).resolve()
    if os.environ.get("RBOT_LOCAL_EMBEDDING_RUNTIME"):
        from .local_rbot_embeddings import validated_variant_runtime
        selected = Path(os.environ["RBOT_LOCAL_EMBEDDING_RUNTIME"])
        return validated_variant_runtime(root, selected if selected.is_absolute() else root / selected)
    source = root / "vendor/LLM4Rewrite"
    runtime = root / ".cache/mcts_rewrite/rbot_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name in {"rag", "my_rewriter", ".git"}:
            continue
        target = runtime / path.name
        if not target.exists():
            target.symlink_to(path, target_is_directory=path.is_dir())
    for name in ("rag", "my_rewriter"):
        destination = runtime / name
        destination.mkdir(exist_ok=True)
        for path in (source / name).iterdir():
            if path.is_file() and path.suffix != ".zip":
                target = destination / path.name
                if name == "my_rewriter" and path.suffix in {".json", ".jsonl"}:
                    if target.is_symlink():
                        target.unlink()
                    if not target.exists():
                        shutil.copy2(path, target)
                elif not target.exists():
                    target.symlink_to(path)
    archive = source / "rag/stackoverflow-rewrite-embed.zip"
    metadata_path = runtime / "extraction.json"
    if not metadata_path.exists():
        extracted = []
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                # The pinned archive contains only four JSONLs and macOS
                # metadata.  Explicitly constrain extraction to flat JSONLs.
                path = Path(info.filename)
                if len(path.parts) == 1 and path.suffix == ".jsonl":
                    destination = runtime / "rag" / path.name
                    if destination.is_symlink():
                        destination.unlink()
                    with handle.open(info) as inp, destination.open("wb") as out:
                        while chunk := inp.read(1024 * 1024):
                            out.write(chunk)
                    extracted.append({"path": path.name, "bytes": info.file_size})
        metadata_path.write_text(json.dumps({"source_archive": str(archive.relative_to(root)),
                                             "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                                             "extracted_files": extracted}, indent=2))
    return runtime


def load_rbot_rewrite_wrapper(source: Path, jar: Path):
    """Run upstream Python wrapper with its sole jar-directory path corrected."""
    wrapper_path = source / "my_rewriter/rewrite.py"
    tree = ast.parse(wrapper_path.read_text(), str(wrapper_path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "local_lib_dir" for t in node.targets):
            node.value = ast.Constant(str(jar.parent))
    ast.fix_missing_locations(tree)
    module = types.ModuleType("my_rewriter.rewrite")
    module.__file__ = str(wrapper_path)
    sys.modules[module.__name__] = module
    exec(compile(tree, str(wrapper_path), "exec"), module.__dict__)
    return module


def prepare_quite_runtime(root: Path, runtime: Path) -> Path:
    """Copy mutable QUITE source/cache paths for a single fresh execution.

    Its original statistics helper rewrites src/utils/data_distribution.py and
    its knowledge-base loader can replace a pickle cache.  Real copies prevent
    either behavior from changing the pinned checkout or another run.
    """
    source = Path(root).resolve() / "vendor/QUITE"
    runtime = Path(runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    for directory in ("src", "config_file"):
        shutil.copytree(source / directory, runtime / directory,
                        ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    shutil.copy2(source / "run.py", runtime / "run.py")
    cli = runtime / "run.py"
    text = cli.read_text()
    broken_call = "    recommender_output_file = await run_hint_recommender(args, directories, dbms, rewriter_output_file)"
    if broken_call not in text:
        raise RuntimeError("Pinned QUITE CLI shape changed; re-audit its recommender flag handling")
    cli.write_text(text.replace(broken_call, "    if args.enable_recommender:\n    " + broken_call))
    patches = [{"path": "run.py", "patch": "honor_enable_recommender_flag", "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "runtime_sha256": hashlib.sha256(cli.read_bytes()).hexdigest()}]
    if os.environ.get("MCTS_MODEL_TRANSPORT") == SESSION_TRANSPORT:
        agent = runtime / "src/Query_Rewriter/agent_definition.py"
        original = agent.read_text()
        old = '''                else:
                    if delta.content != "" and is_answering is False:
                        is_answering = True
                        break'''
        new = '''                else:
                    # Session transport supplies a public assistant analysis,
                    # never a hidden reasoning transcript.
                    if delta.content:
                        print(delta.content, end='', flush=True)
                        reasoning_content += delta.content'''
        if old not in original:
            raise RuntimeError("Pinned QUITE reasoning transport changed; re-audit the public-text adapter")
        agent.write_text(original.replace(old, new, 1))
        patches.append({"path": "src/Query_Rewriter/agent_definition.py", "patch": "consume_public_assistant_text_for_analysis",
                        "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
                        "runtime_sha256": hashlib.sha256(agent.read_bytes()).hexdigest(),
                        "private_reasoning_exposed": False})
    (runtime / "runtime_compatibility.json").write_text(json.dumps(patches, indent=2))
    for directory in ("dataset", "documents"):
        (runtime / directory).symlink_to(source / directory, target_is_directory=True)
    return runtime
