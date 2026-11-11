"""Offline protocol checks: synthetic content only; no server or model starts."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from email.message import Message
import importlib.util
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from scripts.run_stage_a import BudgetedModel

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("session_model_bridge_v2", ROOT / "scripts/session_model_bridge_v2.py")
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
SECRET = "offline-unit-test-only-" + "x" * 32
BODY = {"model": "gpt-6.1-sol", "messages": [{"role": "user", "content": "Synthetic transport unit test"}]}


@pytest.fixture
def bridge(tmp_path):
    value = transport.Bridge(tmp_path, SECRET, timeout=600, poll_interval=0.005)
    yield value
    value.stop("Offline unit test cleanup")


def wait_request(bridge):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        rows = transport.pending_requests(bridge.directory)
        if rows:
            return transport.read_json(bridge.directory / "requests" / rows[0]["request_id"] / "request.json")
        time.sleep(0.005)
    raise AssertionError("Synthetic request was not queued")


def failure_for(request):
    return transport.launch_failure_envelope(request, supervisor_id="offline-supervisor", worker_id="never-ran-worker",
                failure_code="model_capacity", failure_message="Synthetic observed launch-capacity error")


def answer(bridge, request):
    transport.claim_request(bridge.directory, request["request_id"], "offline-worker")
    transport.submit_response(bridge.directory, request["request_id"], "offline-worker", "Synthetic unit-test response")


def test_terminal_launch_failure_is_prompt_honest_and_consumes_one_call(bridge):
    ledger = {}
    model = BudgetedModel(lambda payload: bridge.complete(BODY, json.dumps(BODY).encode(), run_id="new_study"),
                          arm="mcts", ledger=ledger, query_id="synthetic")
    with ThreadPoolExecutor() as pool:
        future = pool.submit(model, {"task": "synthetic"})
        request = wait_request(bridge)
        failure = failure_for(request)
        transport.submit_launch_failure(bridge.directory, failure)
        with pytest.raises(transport.WorkerLaunchFailure) as observed:
            future.result(timeout=0.5)
    directory = bridge.directory / "requests" / request["request_id"]
    assert observed.value.failure == failure
    assert observed.value.error_payload()["error"]["type"] == "session_generation_error"
    assert observed.value.error_payload()["error"]["retryable"] is False
    assert transport.read_json(directory / "state.json")["status"] == "launch_failed"
    assert request["run_id"] == "new_study" and request["actual_serving_model"] is None
    assert transport.read_json(bridge.directory / "server.json")["actual_serving_model"] is None
    assert failure["actual_serving_model"] is None and failure["reasoning_effort"] is None
    assert failure["usage"] == {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    assert not any((directory / name).exists() for name in ("claim.json", "response.json", "completion.json"))
    assert not transport.pending_requests(bridge.directory)
    assert len(list((bridge.directory / "requests").iterdir())) == 1
    assert ledger["total_invocations"] == 1 and len(ledger["calls"]) == 1
    assert ledger["calls"][0]["status"] == "error" and ledger["calls"][0]["retry"] is False
    with pytest.raises(transport.ProtocolError):
        transport.claim_request(bridge.directory, request["request_id"], "late-worker")
    with pytest.raises(transport.ProtocolError):
        transport.submit_launch_failure(bridge.directory, failure)


@pytest.mark.parametrize("key,value", [
    ("session_id", "another_session"), ("run_id", "another_run"), ("request_sha256", "0" * 64),
    ("launch_token", "0" * 64), ("terminal", False), ("phase", "after_claim"), ("schema_version", True),
    ("actual_serving_model", "gpt-6.1-sol"), ("reasoning_effort", "xhigh"),
    ("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
    ("observed_unix", 1), ("observed_unix", float("inf")), ("failure_code", "retry_later"),
    ("failure_code", []), ("request_id", []), ("observed_unix", 10 ** 400),
])
def test_mismatched_or_invalid_signal_cannot_end_an_active_request(bridge, key, value):
    with ThreadPoolExecutor() as pool:
        future = pool.submit(bridge.complete, BODY, json.dumps(BODY).encode())
        request = wait_request(bridge)
        failure = failure_for(request)
        failure[key] = value
        try:
            with pytest.raises(transport.ProtocolError):
                transport.submit_launch_failure(bridge.directory, failure)
            directory = bridge.directory / "requests" / request["request_id"]
            assert transport.read_json(directory / "state.json")["status"] == "queued"
            assert not (directory / "launch_failure.json").exists()
            assert not future.done()
        finally:
            answer(bridge, request)
        assert future.result(timeout=0.5)["choices"][0]["message"]["content"] == "Synthetic unit-test response"


def test_stale_failure_for_identical_input_is_not_rebound_to_a_new_request(bridge):
    with ThreadPoolExecutor() as pool:
        first = pool.submit(bridge.complete, BODY, json.dumps(BODY).encode())
        first_request = wait_request(bridge)
        first_failure = failure_for(first_request)
        transport.submit_launch_failure(bridge.directory, first_failure)
        with pytest.raises(transport.WorkerLaunchFailure):
            first.result(timeout=0.5)
        second = pool.submit(bridge.complete, BODY, json.dumps(BODY).encode())
        second_request = wait_request(bridge)
        assert first_request["request_sha256"] == second_request["request_sha256"]
        assert first_request["request_id"] != second_request["request_id"]
        assert first_request["launch_token"] != second_request["launch_token"]
        stale = {**first_failure, "request_id": second_request["request_id"]}
        try:
            with pytest.raises(transport.ProtocolError, match="launch_token"):
                transport.submit_launch_failure(bridge.directory, stale)
            assert not second.done()
        finally:
            answer(bridge, second_request)
        assert second.result(timeout=0.5)["session_provenance"]["request_id"] == second_request["request_id"]
    assert transport.read_json(bridge.directory / "requests" / first_request["request_id"] / "launch_failure.json") == first_failure


def test_maximum_failure_message_round_trips_even_with_json_escaping(bridge):
    with ThreadPoolExecutor() as pool:
        future = pool.submit(bridge.complete, BODY, json.dumps(BODY).encode())
        request = wait_request(bridge)
        failure = transport.launch_failure_envelope(request, supervisor_id="offline-supervisor",
                    failure_code="worker_launch_error", failure_message="\x00" * 4096)
        transport.submit_launch_failure(bridge.directory, failure)
        with pytest.raises(transport.WorkerLaunchFailure) as observed:
            future.result(timeout=0.5)
    assert observed.value.failure["failure_message"] == "\x00" * 4096


def test_completed_response_wins_race_with_visible_failure_and_cannot_be_overwritten(bridge):
    with ThreadPoolExecutor() as pool:
        future = pool.submit(bridge.complete, BODY, json.dumps(BODY).encode())
        request = wait_request(bridge)
        failure = failure_for(request)
        directory = bridge.directory / "requests" / request["request_id"]
        transport.claim_request(bridge.directory, request["request_id"], "offline-worker")
        with pytest.raises(transport.ProtocolError):
            transport.submit_launch_failure(bridge.directory, failure)
        # Hold the shared process lock so both files become visible to the reader
        # together. Normal supervisor API calls already reject this conflict.
        with transport._locked(directory):
            transport.legacy.submit_response(bridge.directory, request["request_id"], "offline-worker", "Synthetic race response")
            transport.atomic_json(directory / "launch_failure.json", failure)
        completion = future.result(timeout=0.5)
    assert completion["choices"][0]["message"]["content"] == "Synthetic race response"
    assert completion["session_provenance"]["actual_serving_model"] == "gpt-6.1-sol"
    assert completion["session_provenance"]["model_substitution"] is False
    assert transport.read_json(directory / "state.json")["status"] == "completed"
    with pytest.raises(transport.ProtocolError):
        transport.submit_launch_failure(bridge.directory, failure)
    assert transport.read_json(directory / "completion.json") == completion


@pytest.mark.parametrize("external_stop", [False, True])
def test_stop_ends_wait_cleanly_and_prevents_future_protocol_work(bridge, external_stop):
    with ThreadPoolExecutor() as pool:
        future = pool.submit(bridge.complete, BODY, json.dumps(BODY).encode())
        request = wait_request(bridge)
        if external_stop:
            transport.atomic_json(bridge.directory / "STOP", {"reason": "Synthetic external stop"})
        else:
            bridge.stop("Synthetic local stop")
        with pytest.raises(transport.SessionStopped):
            future.result(timeout=0.5)
    directory = bridge.directory / "requests" / request["request_id"]
    assert transport.read_json(directory / "state.json")["status"] == "stopped"
    assert not (directory / "completion.json").exists()
    assert transport.pending_requests(bridge.directory) == []
    with pytest.raises(transport.SessionStopped):
        bridge.complete(BODY, json.dumps(BODY).encode())
    with pytest.raises(transport.ProtocolError):
        transport.claim_request(bridge.directory, request["request_id"], "late-worker")
    with pytest.raises(transport.ProtocolError):
        transport.submit_launch_failure(bridge.directory, failure_for(request))
    assert len(list((bridge.directory / "requests").iterdir())) == 1


def test_no_signal_still_times_out_without_fabricating_a_result(tmp_path):
    bridge = transport.Bridge(tmp_path, SECRET, timeout=0.025, poll_interval=0.005)
    with pytest.raises(TimeoutError):
        bridge.complete(BODY, json.dumps(BODY).encode())
    directory = next((bridge.directory / "requests").iterdir())
    assert transport.read_json(directory / "state.json")["status"] == "timeout"
    assert not (directory / "completion.json").exists()


def test_v2_rejects_legacy_spools_and_model_substitution(tmp_path, bridge):
    legacy_bridge = transport.legacy.Bridge(tmp_path, SECRET)
    with pytest.raises(transport.ProtocolError, match="legacy spools"):
        transport.pending_requests(legacy_bridge.directory)
    with pytest.raises(transport.ProtocolError, match="substitution"):
        bridge.complete({**BODY, "model": "another-model"}, b"{}")
    assert not list((bridge.directory / "requests").iterdir())


@pytest.mark.parametrize("kind,status,error_type", [("failure", 422, "session_generation_error"), ("stop", 410, "session_stopped")])
def test_http_error_contract_without_starting_or_binding_a_server(monkeypatch, kind, status, error_type):
    class NoSocketServer:
        def __init__(self, address, handler):
            self.handler = handler
    monkeypatch.setattr(transport, "ThreadingHTTPServer", NoSocketServer)
    failure = {"synthetic": "only used to check HTTP error serialization"}
    exception = transport.WorkerLaunchFailure({**failure, "failure_code": "model_capacity"}) if kind == "failure" else transport.SessionStopped()
    def complete(*args, **kwargs):
        raise exception
    bridge = SimpleNamespace(secret=SECRET, run_id="synthetic", max_body=8192, timeout=600, complete=complete)
    handler_type = transport.make_server(bridge).handler
    handler = object.__new__(handler_type)
    handler.path = "/v1/chat/completions"
    handler.headers = Message()
    raw = json.dumps(BODY).encode()
    for key, value in {"Authorization": "Bearer " + SECRET, "Content-Type": "application/json", "Content-Length": str(len(raw))}.items():
        handler.headers[key] = value
    handler.rfile, handler.wfile = io.BytesIO(raw), io.BytesIO()
    handler.connection = SimpleNamespace(settimeout=lambda value: None)
    statuses = []
    handler.send_response = statuses.append
    handler.send_header = lambda *args: None
    handler.end_headers = lambda: None
    handler.do_POST()
    assert statuses == [status]
    error = json.loads(handler.wfile.getvalue())["error"]
    assert error["type"] == error_type and error["retryable"] is False


def test_raw_mismatched_failure_is_protocol_error_not_a_valid_launch_failure(bridge):
    with ThreadPoolExecutor() as pool:
        future = pool.submit(bridge.complete, BODY, json.dumps(BODY).encode())
        request = wait_request(bridge)
        invalid = deepcopy(failure_for(request))
        invalid["launch_token"] = "0" * 64
        directory = bridge.directory / "requests" / request["request_id"]
        transport.atomic_json(directory / "launch_failure.json", invalid)
        with pytest.raises(transport.ProtocolError, match="binding mismatch"):
            future.result(timeout=0.5)
    assert transport.read_json(directory / "state.json")["status"] == "error"
    assert not (directory / "completion.json").exists()
