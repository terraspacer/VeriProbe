#!/usr/bin/env python3
"""Opt-in session transport with terminal, request-bound worker-launch failures.

No model is invoked here. The supervisor supplies observed launch failures; a
failure is accepted only before any worker claims the request. Legacy spools and
the legacy bridge are deliberately not upgraded in place.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import secrets
import socket
import time

_spec = importlib.util.spec_from_file_location("_session_model_bridge_v1", Path(__file__).with_name("session_model_bridge.py"))
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
ProtocolError = legacy.ProtocolError
atomic_json, read_json, digest = legacy.atomic_json, legacy.read_json, legacy.digest
SERVING_MODEL, REASONING_EFFORT = legacy.SERVING_MODEL, legacy.REASONING_EFFORT
TRANSPORT, SECRET_ENV, UNKNOWN_USAGE = legacy.TRANSPORT, legacy.SECRET_ENV, legacy.UNKNOWN_USAGE
FAILURE_CODES = {"model_capacity", "worker_launch_error"}
FAILURE_MAX_BYTES = 32_768
FAILURE_FIELDS = {"schema_version", "kind", "terminal", "phase", "session_id", "run_id", "request_id",
                  "request_sha256", "launch_token", "supervisor_id", "worker_id", "failure_code",
                  "failure_message", "observed_unix", "actual_serving_model", "reasoning_effort", "usage"}


class WorkerLaunchFailure(RuntimeError):
    def __init__(self, failure):
        self.failure = failure
        super().__init__("Session worker launch failed before claim: " + failure["failure_code"])

    def error_payload(self):
        return {"error": {"message": str(self), "type": "session_generation_error", "code": "worker_launch_failed",
                          "retryable": False, "generation_failure": self.failure}}


class SessionStopped(RuntimeError):
    pass


def _v2_session(session):
    session = Path(session)
    metadata = read_json(session / "server.json")
    if not isinstance(metadata, dict) or metadata.get("protocol_version") != 2:
        raise ProtocolError("A new v2 session spool is required; legacy spools are immutable")
    return session


def _request_directory(session, request_id):
    if not isinstance(request_id, str):
        raise ProtocolError("Invalid request id")
    return legacy._request_directory(session, request_id)


@contextmanager
def _locked(directory):
    fd = os.open(directory / ".protocol.lock", os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _identity(value):
    return isinstance(value, str) and 0 < len(value) <= 200 and not any(ord(char) < 32 for char in value)


def _validate_failure(failure, request):
    if not isinstance(failure, dict) or set(failure) != FAILURE_FIELDS:
        raise ProtocolError("Expected exactly the v2 worker-launch failure fields")
    for key in ("session_id", "run_id", "request_id", "request_sha256", "launch_token"):
        if failure[key] != request[key]:
            raise ProtocolError("Worker-launch failure request binding mismatch: " + key)
    observed = failure["observed_unix"]
    message = failure["failure_message"]
    if (type(failure["schema_version"]) is not int or failure["schema_version"] != 2
            or failure["kind"] != "worker_launch_failure" or failure["terminal"] is not True
            or failure["phase"] != "before_claim" or not isinstance(failure["failure_code"], str)
            or failure["failure_code"] not in FAILURE_CODES
            or not _identity(failure["supervisor_id"])
            or failure["worker_id"] is not None and not _identity(failure["worker_id"])
            or not isinstance(message, str) or not message or len(message.encode()) > 4096
            or isinstance(observed, bool) or not isinstance(observed, (int, float))
            or not request["created_unix"] <= observed <= min(time.time(), request["deadline_unix"])
            or failure["actual_serving_model"] is not None or failure["reasoning_effort"] is not None
            or failure["usage"] != UNKNOWN_USAGE):
        raise ProtocolError("Invalid terminal before-claim worker-launch failure")
    return failure


def launch_failure_envelope(request, *, supervisor_id, failure_code, failure_message, worker_id=None,
                            observed_unix=None):
    """Bind an actual observed launch failure to the supervisor's queued request."""
    failure = {key: request[key] for key in ("session_id", "run_id", "request_id", "request_sha256", "launch_token")}
    failure.update(schema_version=2, kind="worker_launch_failure", terminal=True, phase="before_claim",
                   supervisor_id=supervisor_id, worker_id=worker_id, failure_code=failure_code,
                   failure_message=failure_message, observed_unix=time.time() if observed_unix is None else observed_unix,
                   actual_serving_model=None, reasoning_effort=None, usage=dict(UNKNOWN_USAGE))
    return _validate_failure(failure, request)


def pending_requests(session):
    session = _v2_session(session)
    rows = legacy.pending_requests(session)
    for row in rows:
        request = read_json(session / "requests" / row["request_id"] / "request.json")
        row.update({key: request[key] for key in ("session_id", "launch_token", "configured_serving_model")})
    return rows


def claim_request(session, request_id, worker_id):
    session = _v2_session(session)
    directory = _request_directory(session, request_id)
    with _locked(directory):
        if not _identity(worker_id):
            raise ProtocolError("A bounded session worker identity is required")
        if (directory / "launch_failure.json").exists():
            raise ProtocolError("Request has a terminal worker-launch failure")
        return legacy.claim_request(session, request_id, worker_id)


def submit_response(session, request_id, worker_id, content):
    session = _v2_session(session)
    directory = _request_directory(session, request_id)
    with _locked(directory):
        if (directory / "launch_failure.json").exists():
            raise ProtocolError("Request has a terminal worker-launch failure")
        return legacy.submit_response(session, request_id, worker_id, content)


def submit_launch_failure(session, failure):
    """Commit one validated terminal failure; never claim, retry, or respond."""
    session = _v2_session(session)
    if not isinstance(failure, dict):
        raise ProtocolError("Expected a worker-launch failure object")
    directory = _request_directory(session, failure.get("request_id", ""))
    with _locked(directory):
        if (session / "STOP").exists():
            raise ProtocolError("The supervised session has stopped")
        request, state = read_json(directory / "request.json"), read_json(directory / "state.json")
        if request["request_sha256"] != digest(request["request"]):
            raise ProtocolError("Queued request hash mismatch")
        _validate_failure(failure, request)
        if (state.get("status") != "queued" or request["deadline_unix"] <= time.time()
                or any((directory / name).exists() for name in ("claim.json", "response.json", "completion.json", "launch_failure.json"))):
            raise ProtocolError("Launch failure requires an active, unclaimed request with no terminal result")
        atomic_json(directory / "launch_failure.json", failure)
        atomic_json(directory / "state.json", {"status": "launch_failed", "deadline_unix": request["deadline_unix"],
                    "failure_code": failure["failure_code"], "actual_serving_model": None, "usage_known": False})
        return failure


class Bridge(legacy.Bridge):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        metadata = read_json(self.directory / "server.json")
        metadata.update(protocol_version=2, actual_serving_model=None, reasoning_effort=None,
                        configured_serving_model=SERVING_MODEL, configured_reasoning_effort=REASONING_EFFORT,
                        bridge_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        helper_source_sha256=hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest())
        atomic_json(self.directory / "server.json", metadata)

    def _stopped(self):
        return self.closed.is_set() or (self.directory / "STOP").exists()

    def stop(self, reason="Supervisor stopped the session"):
        atomic_json(self.directory / "STOP", {"stopped_at": datetime.now(timezone.utc).isoformat(), "reason": reason})
        self.closed.set()

    def complete(self, body, raw, *, run_id=None):
        legacy.validate_request(body)
        if body["model"] != SERVING_MODEL:
            raise ProtocolError("v2 requires the configured serving model; model substitution is unsupported")
        run_id = run_id or self.run_id
        if not legacy.LABEL.fullmatch(run_id):
            raise ProtocolError("Invalid run identifier")
        if self._stopped():
            raise SessionStopped("The supervised session has stopped")
        if not self.pending.acquire(blocking=False):
            raise ProtocolError("Bridge pending-request limit reached")
        directory = None
        try:
            if self._stopped():
                raise SessionStopped("The supervised session has stopped")
            request_id = "req_" + secrets.token_hex(16)
            directory = self.directory / "requests" / request_id
            directory.mkdir(mode=0o700)
            created = time.time()
            deadline = created + self.timeout
            options = {key: value for key, value in body.items() if key not in {"model", "messages"}}
            request = {"request_id": request_id, "session_id": self.session_id, "run_id": run_id,
                       "request": body, "requested_model": body["model"], "messages": body["messages"], "options": options,
                       "request_sha256": digest(body), "messages_sha256": digest(body["messages"]), "options_sha256": digest(options),
                       "raw_body_sha256": hashlib.sha256(raw).hexdigest(), "created_unix": created, "deadline_unix": deadline,
                       "actual_serving_model": None, "reasoning_effort": None, "configured_serving_model": SERVING_MODEL,
                       "configured_reasoning_effort": REASONING_EFFORT, "transport": TRANSPORT, "protocol_version": 2,
                       "launch_token": secrets.token_hex(32), "usage_known": False,
                       "requested_sampling_options_not_enforced": sorted(set(options) & {"temperature", "top_p", "seed",
                           "max_tokens", "max_completion_tokens", "stop", "presence_penalty", "frequency_penalty"})}
            atomic_json(directory / "state.json", {"status": "queued", "deadline_unix": deadline})
            atomic_json(directory / "request.json", request)
            while True:
                with _locked(directory):
                    if self._stopped():
                        atomic_json(directory / "state.json", {"status": "stopped", "deadline_unix": deadline})
                        raise SessionStopped("The supervised session has stopped")
                    if time.time() >= deadline:
                        atomic_json(directory / "state.json", {"status": "timeout", "deadline_unix": deadline})
                        raise TimeoutError("No actual session worker response arrived before the bounded deadline")
                    # A valid completed response takes priority over a concurrently visible failure signal.
                    if (directory / "response.json").exists():
                        completion = self._completion(directory, request)
                        atomic_json(directory / "completion.json", completion)
                        atomic_json(directory / "state.json", {"status": "completed", "deadline_unix": deadline})
                        return completion
                    if (directory / "launch_failure.json").exists():
                        failure = _validate_failure(read_json(directory / "launch_failure.json", FAILURE_MAX_BYTES), request)
                        if (directory / "claim.json").exists():
                            raise ProtocolError("A before-claim launch failure cannot terminate a claimed request")
                        atomic_json(directory / "state.json", {"status": "launch_failed", "deadline_unix": deadline,
                                    "failure_code": failure["failure_code"], "actual_serving_model": None, "usage_known": False})
                        raise WorkerLaunchFailure(failure)
                self.closed.wait(min(self.poll_interval, max(0, deadline - time.time())))
        except (WorkerLaunchFailure, SessionStopped, TimeoutError):
            raise
        except (ProtocolError, ValueError, OSError):
            if directory is not None:
                atomic_json(directory / "state.json", {"status": "error"})
            raise
        finally:
            self.pending.release()

    def _completion(self, directory, request):
        response = read_json(directory / "response.json", 4_194_304 + 8192)
        claim = read_json(directory / "claim.json")
        if (response.get("request_id") != request["request_id"] or response.get("request_sha256") != request["request_sha256"]
                or response.get("worker_id") != claim.get("worker_id") or response.get("actual_serving_model") != SERVING_MODEL
                or response.get("reasoning_effort") != REASONING_EFFORT or response.get("transport") != TRANSPORT
                or response.get("finish_reason") != "stop" or response.get("usage") != UNKNOWN_USAGE
                or not isinstance(response.get("content"), str) or not response["content"]
                or response.get("content_sha256") != hashlib.sha256(response["content"].encode()).hexdigest()):
            raise ProtocolError("Session response provenance/hash mismatch")
        if request["request"].get("response_format") == {"type": "json_object"} and not isinstance(json.loads(response["content"]), dict):
            raise ProtocolError("The upstream requested a JSON object")
        provenance = {key: request[key] for key in ("request_id", "session_id", "run_id", "requested_model", "transport",
                      "request_sha256", "messages_sha256", "options_sha256", "requested_sampling_options_not_enforced")}
        provenance.update(actual_serving_model=SERVING_MODEL, reasoning_effort=REASONING_EFFORT, protocol_version=2,
                          worker_id=response["worker_id"], content_sha256=response["content_sha256"], model_substitution=False,
                          published_model_configuration_reproduced=False, usage_known=False,
                          streaming_mode="completed_response_as_sse" if request["request"].get("stream") else "nonstreaming")
        return {"id": "chatcmpl-" + request["request_id"], "object": "chat.completion", "created": int(request["created_unix"]),
                "model": SERVING_MODEL, "choices": [{"index": 0, "message": {"role": "assistant", "content": response["content"]},
                "finish_reason": "stop", "logprobs": None}], "usage": dict(UNKNOWN_USAGE), "session_provenance": provenance}


def make_server(bridge, host="127.0.0.1", port=8765):
    """Same local text API as v1, with explicit nonretry launch-failure errors."""
    if host not in ("127.0.0.1", "::1"):
        raise ProtocolError("The bridge may bind only to a literal localhost address")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def _send(self, status, value):
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)

        def _authorized(self):
            return hmac.compare_digest(self.headers.get("Authorization", "").encode(), ("Bearer " + bridge.secret).encode())

        def do_GET(self):
            if not self._authorized():
                return self._send(401, {"error": {"message": "Unauthorized", "type": "authentication_error"}})
            if self.path != "/health":
                return self._send(404, {"error": {"message": "Unsupported route"}})
            return self._send(200, {"session_id": bridge.session_id, "protocol_version": 2, "actual_serving_model": None,
                                   "configured_serving_model": SERVING_MODEL, "transport": TRANSPORT,
                                   "stopped": bridge._stopped(), "queued_requests": len(pending_requests(bridge.directory))})

        def do_POST(self):
            if not self._authorized():
                return self._send(401, {"error": {"message": "Unauthorized", "type": "authentication_error"}})
            try:
                run_id = legacy._route(self.path, bridge.run_id)
                if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Encoding"):
                    raise ProtocolError("Encoded or chunked requests are unsupported")
                if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                    raise ProtocolError("Expected application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= bridge.max_body:
                    raise ProtocolError("Missing or excessive Content-Length")
                self.connection.settimeout(min(30, bridge.timeout))
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ProtocolError("Incomplete body")
                body = json.loads(raw, object_pairs_hook=legacy._object_pairs,
                                  parse_constant=lambda value: (_ for _ in ()).throw(ProtocolError("Nonfinite JSON number")))
                completion = bridge.complete(body, raw, run_id=run_id)
                if not body.get("stream"):
                    return self._send(200, completion)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                common = {key: completion[key] for key in ("id", "created", "model", "session_provenance")}
                common.update(object="chat.completion.chunk", usage=None)
                chunks = [{**common, "choices": [{"index": 0, "delta": {"role": "assistant", "content": completion["choices"][0]["message"]["content"]}, "finish_reason": None}]},
                          {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
                if body.get("stream_options", {}).get("include_usage"):
                    chunks.append({**common, "choices": [], "usage": dict(UNKNOWN_USAGE)})
                for chunk in chunks:
                    self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
            except WorkerLaunchFailure as exc:
                self._send(422, exc.error_payload())
            except SessionStopped:
                self._send(410, {"error": {"message": "Supervised session stopped", "type": "session_stopped", "retryable": False}})
            except TimeoutError:
                self._send(504, {"error": {"message": "Session worker deadline expired", "type": "session_timeout", "retryable": False}})
            except (ProtocolError, ValueError, OSError):
                self._send(400, {"error": {"message": "Unsupported or invalid session bridge request/response", "type": "invalid_request_error"}})

    class LocalServer(ThreadingHTTPServer):
        daemon_threads = True
        address_family = socket.AF_INET6 if host == "::1" else socket.AF_INET

    return LocalServer((host, int(port)), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--spool", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--session-id")
    serve.add_argument("--run-id", default="default")
    serve.add_argument("--timeout", type=float, default=240)
    serve.add_argument("--max-pending", type=int, default=8)
    for command in ("pending", "claim", "respond", "fail-launch", "stop"):
        child = sub.add_parser(command)
        child.add_argument("--session", type=Path, required=True)
        if command in {"claim", "respond"}:
            child.add_argument("--request-id", required=True)
            child.add_argument("--worker-id", required=True)
        if command == "respond":
            child.add_argument("--content-file", type=Path, required=True)
        if command == "fail-launch":
            child.add_argument("--envelope-file", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "serve":
        bridge = Bridge(args.spool, os.environ.get(SECRET_ENV, ""), session_id=args.session_id,
                        run_id=args.run_id, timeout=args.timeout, max_pending=args.max_pending)
        server = make_server(bridge, args.host, args.port)
        authority = "[" + args.host + "]" if ":" in args.host else args.host
        print(json.dumps({"session_directory": str(bridge.directory), "base_url": f"http://{authority}:{server.server_port}/v1",
                          "configured_serving_model": SERVING_MODEL, "actual_serving_model": None,
                          "protocol_version": 2, "transport": TRANSPORT, "secret_environment_variable": SECRET_ENV}), flush=True)
        try:
            server.serve_forever()
        finally:
            bridge.stop()
            server.server_close()
    elif args.command == "pending":
        print(json.dumps(pending_requests(args.session), indent=2))
    elif args.command == "claim":
        print(json.dumps(claim_request(args.session, args.request_id, args.worker_id), ensure_ascii=False, indent=2))
    elif args.command == "respond":
        print(json.dumps(submit_response(args.session, args.request_id, args.worker_id, args.content_file.read_text())))
    elif args.command == "fail-launch":
        print(json.dumps(submit_launch_failure(args.session, read_json(args.envelope_file, FAILURE_MAX_BYTES)), indent=2))
    else:
        session = _v2_session(args.session)
        atomic_json(session / "STOP", {"stopped_at": datetime.now(timezone.utc).isoformat(), "reason": "Supervisor stop command"})


if __name__ == "__main__":
    main()
