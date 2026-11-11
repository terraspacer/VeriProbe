from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request

import pytest

from mcts_rewrite.fresh_baselines import bound_session_environment, model_configuration
from mcts_rewrite.upstream_runtime import prepare_quite_runtime, session_transport_provenance

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("session_model_bridge", ROOT / "scripts/session_model_bridge.py")
bridge_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge_module)
SECRET = "unit-test-only-" + "x" * 32


@pytest.fixture
def bridge(tmp_path):
    value = bridge_module.Bridge(tmp_path, SECRET, timeout=3, poll_interval=0.005)
    yield value
    value.closed.set()


def wait_pending(bridge):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        rows = bridge_module.pending_requests(bridge.directory)
        if rows:
            return rows[0]
        time.sleep(0.005)
    raise AssertionError("Test request was not queued")


def answer(bridge, content="synthetic unit-test answer"):
    pending = wait_pending(bridge)
    request = bridge_module.claim_request(bridge.directory, pending["request_id"], "unit-test-harness")
    bridge_module.submit_response(bridge.directory, pending["request_id"], "unit-test-harness", content)
    return request


def test_exact_queue_claim_response_and_honest_model_metadata(bridge):
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "A literal $() shell string is just prompt text."}],
            "temperature": 0, "max_tokens": 50, "stream": False}
    raw = json.dumps(body).encode()
    with ThreadPoolExecutor() as executor:
        future = executor.submit(bridge.complete, body, raw, run_id="demo")
        request = answer(bridge)
        completion = future.result()
    assert request["request"] == body and request["messages"] == body["messages"]
    assert request["request_sha256"] == bridge_module.digest(body)
    assert request["options"] == {"temperature": 0, "max_tokens": 50, "stream": False}
    assert completion["model"] == "gpt-6.1-sol"
    assert completion["usage"] == bridge_module.UNKNOWN_USAGE
    provenance = completion["session_provenance"]
    assert provenance["requested_model"] == "gpt-4o" and provenance["model_substitution"] is True
    assert provenance["reasoning_effort"] == "xhigh" and provenance["usage_known"] is False
    assert provenance["requested_sampling_options_not_enforced"] == ["max_tokens", "temperature"]
    assert SECRET not in "".join(path.read_text() for path in bridge.directory.rglob("*.json"))
    with pytest.raises((FileExistsError, bridge_module.ProtocolError)):
        bridge_module.claim_request(bridge.directory, request["request_id"], "another-test-harness")


@pytest.mark.parametrize("extra", [{"tools": []}, {"n": 2}, {"n": 1.0}, {"logprobs": True},
                                    {"response_format": {"type": "json_schema"}}, {"reasoning_effort": "low"},
                                    {"messages": [{"role": "tool", "content": "x"}]},
                                    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "http://example.invalid"}]}]}])
def test_unsupported_features_fail_before_queueing(bridge, extra):
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "unit test"}], **extra}
    with pytest.raises(bridge_module.ProtocolError):
        bridge.complete(body, json.dumps(body).encode())
    assert not list((bridge.directory / "requests").iterdir())


def test_no_worker_times_out_and_late_answer_is_rejected(tmp_path):
    bridge = bridge_module.Bridge(tmp_path, SECRET, timeout=0.03, poll_interval=0.005)
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "unit test"}]}
    with pytest.raises(TimeoutError):
        bridge.complete(body, json.dumps(body).encode())
    request = next((bridge.directory / "requests").iterdir())
    assert json.loads((request / "state.json").read_text())["status"] == "timeout"
    with pytest.raises(bridge_module.ProtocolError):
        bridge_module.claim_request(bridge.directory, request.name, "unit-test-harness")


def test_http_auth_loopback_and_completed_response_sse(bridge):
    with pytest.raises(bridge_module.ProtocolError):
        bridge_module.make_server(bridge, "0.0.0.0", 0)
    server = bridge_module.make_server(bridge, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(url + "/health")
        assert missing.value.code == 401
        body = {"model": "native-QUITE-role", "messages": [{"role": "user", "content": "unit test"}],
                "stream": True, "stream_options": {"include_usage": True}}
        request = urllib.request.Request(url + "/runs/demo_run/v1/chat/completions", data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + SECRET})
        with ThreadPoolExecutor() as executor:
            response = executor.submit(lambda: urllib.request.urlopen(request).read().decode())
            queued = answer(bridge, "public answer, not a hidden thought transcript")
            stream = response.result()
        events = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: {")]
        assert queued["run_id"] == "demo_run"
        assert events[0]["choices"][0]["delta"]["content"] == "public answer, not a hidden thought transcript"
        assert "reasoning_content" not in stream
        assert events[-1]["usage"] == bridge_module.UNKNOWN_USAGE
        assert "data: [DONE]" in stream
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(urllib.request.Request(url + "/v1/embeddings", data=b"{}", headers={
                "Content-Type": "application/json", "Authorization": "Bearer " + SECRET}))
    finally:
        server.shutdown()
        server.server_close()


def test_session_audit_is_run_bound_and_role_model_context_is_preserved(bridge):
    body = {"model": "native-QUITE-role", "messages": [{"role": "user", "content": "unit test"}], "temperature": 0}
    with ThreadPoolExecutor() as executor:
        future = executor.submit(bridge.complete, body, json.dumps(body).encode(), run_id="selected_run")
        answer(bridge)
        future.result()
    env = {"MCTS_MODEL_TRANSPORT": bridge_module.TRANSPORT, "MCTS_SESSION_BRIDGE_SPOOL": str(bridge.directory),
           "MCTS_SESSION_BRIDGE_RUN_ID": "selected_run", "REASONING_MODEL": "requested-reasoner",
           "DECISION_MODEL": "requested-decision", "ASSISTANT_MODEL": "requested-assistant"}
    audit = session_transport_provenance("quite", env)
    assert audit["session_completion_count"] == 1 and audit["session_workers"] == ["unit-test-harness"]
    context = audit["model_context"]
    assert context["requested_role_models"]["reasoning"] == "requested-reasoner"
    assert set(context["actual_role_models"].values()) == {"gpt-6.1-sol"}
    assert context["embedding_source"]["provider"] == "local_bm25"
    assert context["published_model_configuration_reproduced"] is False
    assert model_configuration("quite", env)["model_context"]["actual_serving_model"] == "gpt-6.1-sol"
    bound = bound_session_environment({**env, "REASONING_MODEL_URL": "http://127.0.0.1:8765/v1"}, "another_run")
    assert bound["REASONING_MODEL_URL"] == "http://127.0.0.1:8765/runs/another_run/v1"
    assert env["MCTS_SESSION_BRIDGE_RUN_ID"] == "selected_run"
    env["MCTS_SESSION_BRIDGE_RUN_ID"] = "other_run"
    assert session_transport_provenance("quite", env)["session_completion_count"] == 0


def test_quite_session_patch_is_only_in_writable_overlay(tmp_path, monkeypatch):
    source = ROOT / "vendor/QUITE/src/Query_Rewriter/agent_definition.py"
    before = source.read_bytes()
    monkeypatch.setenv("MCTS_MODEL_TRANSPORT", bridge_module.TRANSPORT)
    runtime = prepare_quite_runtime(ROOT, tmp_path / "runtime")
    patched = runtime / "src/Query_Rewriter/agent_definition.py"
    assert "reasoning_content += delta.content" in patched.read_text()
    patches = json.loads((runtime / "runtime_compatibility.json").read_text())
    assert patches[-1]["patch"] == "consume_public_assistant_text_for_analysis"
    assert patches[-1]["private_reasoning_exposed"] is False
    assert source.read_bytes() == before


def test_installed_openai_and_haystack_accept_nullable_unknown_usage():
    python = ROOT / ".venv-quite/bin/python"
    if not python.exists():
        pytest.skip("Optional isolated upstream environment is not installed")
    code = '''import httpx
from openai import OpenAI
from haystack.components.generators.openai import OpenAIGenerator
payload={"id":"chatcmpl-unit","object":"chat.completion","created":1,"model":"gpt-6.1-sol","choices":[{"index":0,"message":{"role":"assistant","content":"unit test"},"finish_reason":"stop"}],"usage":{"prompt_tokens":None,"completion_tokens":None,"total_tokens":None}}
client=OpenAI(api_key="unit-test-only",base_url="http://127.0.0.1:1/v1",http_client=httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,json=payload))))
completion=client.chat.completions.create(model="native-model",messages=[{"role":"user","content":"unit test"}])
message=OpenAIGenerator._build_message(completion,completion.choices[0])
assert message.meta["usage"]["total_tokens"] is None
assert message.text=="unit test"
'''
    subprocess.run([str(python), "-c", code], check=True, timeout=60, capture_output=True, text=True)


def test_http_model_preserves_session_response_provenance(bridge, monkeypatch):
    from mcts_rewrite.adapters import HTTPModel
    monkeypatch.setenv("MCTS_MODEL_API_KEY", SECRET)
    server = bridge_module.make_server(bridge, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        model = HTTPModel(f"http://127.0.0.1:{server.server_port}/runs/hybrid/v1/chat/completions", "gpt-4o", timeout=3)
        with ThreadPoolExecutor() as executor:
            future = executor.submit(model, {"task": "cost_critic", "sql": "select 1;"})
            answer(bridge, '{"predicted_ms":12,"rationale":"synthetic unit-test estimate"}')
            result = future.result()
        assert result["predicted_ms"] == 12
        session = result["provenance"]["session_transport"]
        assert result["provenance"]["requested_model"] == "gpt-4o"
        assert result["provenance"]["actual_model"] == "gpt-6.1-sol"
        assert session["requested_model"] == "gpt-4o" and session["actual_serving_model"] == "gpt-6.1-sol"
        assert session["run_id"] == "hybrid" and session["usage_known"] is False
    finally:
        server.shutdown()
        server.server_close()
