#!/usr/bin/env python3
"""Local OpenAI-compatible text transport to explicitly supervised session agents.

This program never invokes a model, shell, or external network endpoint. A real
session worker must claim each prompt and write the actual assistant answer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import threading
import time
from urllib.parse import urlsplit

SERVING_MODEL = "gpt-6.1-sol"
REASONING_EFFORT = "xhigh"
TRANSPORT = "codex_session_file_spool"
SECRET_ENV = "MCTS_SESSION_BRIDGE_SECRET"
REQUEST_ID = re.compile(r"req_[0-9a-f]{32}")
LABEL = re.compile(r"[A-Za-z0-9_-]{1,96}")
UNKNOWN_USAGE = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}


class ProtocolError(ValueError):
    pass


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + "." + secrets.token_hex(8) + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("Duplicate JSON keys are unsupported")
        result[key] = value
    return result


def read_json(path: Path, limit=8_388_608):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ProtocolError("Expected a bounded regular JSON file")
    return json.loads(path.read_text(), object_pairs_hook=_object_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ProtocolError("Nonfinite JSON number")))


def validate_request(body: dict) -> dict:
    if not isinstance(body, dict):
        raise ProtocolError("Request must be an object")
    allowed = {"model", "messages", "stream", "stream_options", "temperature", "top_p", "n",
               "max_tokens", "max_completion_tokens", "stop", "response_format", "seed",
               "presence_penalty", "frequency_penalty", "reasoning_effort", "logprobs"}
    if set(body) - allowed:
        raise ProtocolError("Unsupported request fields: " + ", ".join(sorted(set(body) - allowed)))
    if not isinstance(body.get("model"), str) or not body["model"] or len(body["model"]) > 200:
        raise ProtocolError("A bounded model name is required")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ProtocolError("A nonempty text conversation is required")
    for message in messages:
        if not isinstance(message, dict) or set(message) - {"role", "content", "name"}:
            raise ProtocolError("Only role/content/name text messages are supported")
        if message.get("role") not in {"system", "developer", "user", "assistant"}:
            raise ProtocolError("Tool/function messages are unsupported")
        if "name" in message and (not isinstance(message["name"], str) or not LABEL.fullmatch(message["name"])):
            raise ProtocolError("Invalid message name")
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or not content:
            raise ProtocolError("Only string or text-part content is supported")
        if any(not isinstance(part, dict) or set(part) != {"type", "text"} or part["type"] != "text"
               or not isinstance(part["text"], str) for part in content):
            raise ProtocolError("Images/audio/files and unknown content parts are unsupported")
    if "stream" in body and not isinstance(body["stream"], bool):
        raise ProtocolError("stream must be boolean")
    if type(body.get("n", 1)) is not int or body.get("n", 1) != 1:
        raise ProtocolError("Exactly one completion is supported")
    for name, low, high in (("temperature", 0, 2), ("top_p", 0, 1),
                            ("presence_penalty", -2, 2), ("frequency_penalty", -2, 2)):
        value = body.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or not low <= value <= high):
            raise ProtocolError("Invalid " + name)
    for name in ("max_tokens", "max_completion_tokens"):
        value = body.get(name)
        if value is not None and (type(value) is not int or value < 1):
            raise ProtocolError("Invalid " + name)
    if body.get("seed") is not None and type(body["seed"]) is not int:
        raise ProtocolError("seed must be an integer")
    if body.get("logprobs") is not None and body.get("logprobs") is not False:
        raise ProtocolError("Token log probabilities are unavailable")
    if body.get("reasoning_effort") not in (None, "xhigh"):
        raise ProtocolError("Only the explicitly selected xhigh session effort is supported")
    if body.get("response_format") not in (None, {"type": "text"}, {"type": "json_object"}):
        raise ProtocolError("Only text and json_object response formats are supported")
    stop = body.get("stop")
    if stop is not None and not (isinstance(stop, str) or isinstance(stop, list)
                                and len(stop) <= 4 and all(isinstance(item, str) for item in stop)):
        raise ProtocolError("Invalid stop sequences")
    options = body.get("stream_options")
    if options is not None and (not body.get("stream") or not isinstance(options, dict)
                               or set(options) - {"include_usage"}
                               or "include_usage" in options and not isinstance(options["include_usage"], bool)):
        raise ProtocolError("Only streaming include_usage is supported")
    return body


def _request_directory(session: Path, request_id: str) -> Path:
    if not REQUEST_ID.fullmatch(request_id):
        raise ProtocolError("Invalid request id")
    directory = session / "requests" / request_id
    if directory.is_symlink() or not directory.is_dir():
        raise ProtocolError("Unknown request id")
    return directory


def pending_requests(session: Path) -> list[dict]:
    if (session / "STOP").exists():
        return []
    pending = []
    for path in sorted((session / "requests").glob("req_*/request.json")):
        state = read_json(path.parent / "state.json")
        if state.get("status") == "queued" and state.get("deadline_unix", 0) > time.time():
            request = read_json(path)
            pending.append({"request_id": request["request_id"], "run_id": request["run_id"],
                            "requested_model": request["requested_model"], "request_sha256": request["request_sha256"],
                            "claimed": (path.parent / "claim.json").exists()})
    return pending


def claim_request(session: Path, request_id: str, worker_id: str) -> dict:
    if not worker_id or len(worker_id) > 200 or any(ord(char) < 32 for char in worker_id):
        raise ProtocolError("A bounded session worker identity is required")
    if (session / "STOP").exists():
        raise ProtocolError("The supervised session has stopped")
    directory = _request_directory(session, request_id)
    state = read_json(directory / "state.json")
    if state.get("status") != "queued" or state["deadline_unix"] <= time.time():
        raise ProtocolError("Request is no longer queued")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(directory / "claim.json", flags, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump({"worker_id": worker_id, "claimed_at": datetime.now(timezone.utc).isoformat()}, handle)
    request = read_json(directory / "request.json")
    if request["request_sha256"] != digest(request["request"]):
        raise ProtocolError("Queued request hash mismatch")
    return request


def submit_response(session: Path, request_id: str, worker_id: str, content: str) -> dict:
    if (session / "STOP").exists():
        raise ProtocolError("The supervised session has stopped")
    directory = _request_directory(session, request_id)
    request = read_json(directory / "request.json")
    claim = read_json(directory / "claim.json")
    state = read_json(directory / "state.json")
    if claim.get("worker_id") != worker_id or state.get("status") != "queued" or state["deadline_unix"] <= time.time():
        raise ProtocolError("Worker does not own an active request")
    if not isinstance(content, str) or not content or len(content.encode()) > 4_194_304:
        raise ProtocolError("Expected a bounded nonempty actual assistant response")
    if request["request"].get("response_format") == {"type": "json_object"}:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ProtocolError("The upstream requested a JSON object")
    response = {"request_id": request_id, "request_sha256": request["request_sha256"],
                "content": content, "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "worker_id": worker_id, "actual_serving_model": SERVING_MODEL,
                "reasoning_effort": REASONING_EFFORT, "transport": TRANSPORT,
                "finish_reason": "stop", "usage": UNKNOWN_USAGE,
                "created_at": datetime.now(timezone.utc).isoformat()}
    if (directory / "response.json").exists():
        raise ProtocolError("Response already exists")
    atomic_json(directory / "response.json", response)
    return {"request_id": request_id, "content_sha256": response["content_sha256"]}


class Bridge:
    def __init__(self, spool: Path, secret: str, *, session_id=None, run_id="default", timeout=240,
                 poll_interval=0.1, max_pending=8, max_body=8_388_608):
        if not isinstance(secret, str) or len(secret.encode()) < 32:
            raise ProtocolError("Set a random per-session secret of at least 32 bytes in " + SECRET_ENV)
        self.session_id = session_id or "session_" + secrets.token_hex(16)
        if not LABEL.fullmatch(self.session_id) or not LABEL.fullmatch(run_id):
            raise ProtocolError("Invalid session/run identifier")
        if timeout <= 0 or poll_interval <= 0 or max_pending < 1 or max_body < 1:
            raise ProtocolError("Timeout, poll interval and resource limits must be positive")
        self.directory = Path(spool).resolve() / self.session_id
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.directory.chmod(0o700)
        (self.directory / "requests").mkdir(mode=0o700)
        self.secret, self.run_id, self.timeout = secret, run_id, float(timeout)
        self.poll_interval, self.max_body = poll_interval, max_body
        self.pending = threading.BoundedSemaphore(max_pending)
        self.closed = threading.Event()
        atomic_json(self.directory / "server.json", {"session_id": self.session_id, "transport": TRANSPORT,
                    "actual_serving_model": SERVING_MODEL, "reasoning_effort": REASONING_EFFORT,
                    "supervised_session_workers_required": True, "usage_known": False,
                    "created_at": datetime.now(timezone.utc).isoformat(), "timeout_s": self.timeout,
                    "bridge_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def complete(self, body: dict, raw: bytes, *, run_id=None) -> dict:
        validate_request(body)
        if not self.pending.acquire(blocking=False):
            raise ProtocolError("Bridge pending-request limit reached")
        directory = None
        try:
            request_id = "req_" + secrets.token_hex(16)
            directory = self.directory / "requests" / request_id
            directory.mkdir(mode=0o700)
            created, deadline = time.time(), time.time() + self.timeout
            options = {key: value for key, value in body.items() if key not in {"model", "messages"}}
            envelope = {"request_id": request_id, "session_id": self.session_id, "run_id": run_id or self.run_id,
                        "request": body, "requested_model": body["model"], "messages": body["messages"],
                        "options": options, "request_sha256": digest(body), "messages_sha256": digest(body["messages"]),
                        "options_sha256": digest(options), "raw_body_sha256": hashlib.sha256(raw).hexdigest(),
                        "created_unix": created, "deadline_unix": deadline,
                        "actual_serving_model": SERVING_MODEL, "reasoning_effort": REASONING_EFFORT,
                        "transport": TRANSPORT, "usage_known": False,
                        "requested_sampling_options_not_enforced": sorted(set(options) &
                             {"temperature", "top_p", "seed", "max_tokens", "max_completion_tokens", "stop",
                              "presence_penalty", "frequency_penalty"})}
            atomic_json(directory / "state.json", {"status": "queued", "deadline_unix": deadline})
            atomic_json(directory / "request.json", envelope)
            while time.time() < deadline and not self.closed.is_set():
                if (directory / "response.json").exists():
                    response = read_json(directory / "response.json", 4_194_304 + 8192)
                    claim = read_json(directory / "claim.json")
                    if (response.get("request_id") != request_id or response.get("request_sha256") != envelope["request_sha256"]
                            or response.get("worker_id") != claim.get("worker_id")
                            or response.get("actual_serving_model") != SERVING_MODEL or response.get("reasoning_effort") != REASONING_EFFORT
                            or response.get("transport") != TRANSPORT or response.get("finish_reason") != "stop"
                            or response.get("usage") != UNKNOWN_USAGE or not isinstance(response.get("content"), str)
                            or not response["content"] or response.get("content_sha256") != hashlib.sha256(response["content"].encode()).hexdigest()):
                        raise ProtocolError("Session response provenance/hash mismatch")
                    if body.get("response_format") == {"type": "json_object"} and not isinstance(json.loads(response["content"]), dict):
                        raise ProtocolError("The upstream requested a JSON object")
                    provenance = {key: envelope[key] for key in ("request_id", "session_id", "run_id", "requested_model",
                                  "actual_serving_model", "reasoning_effort", "transport", "request_sha256",
                                  "messages_sha256", "options_sha256", "requested_sampling_options_not_enforced")}
                    provenance.update({"worker_id": response["worker_id"], "content_sha256": response["content_sha256"],
                                       "model_substitution": body["model"] != SERVING_MODEL,
                                       "published_model_configuration_reproduced": False, "usage_known": False,
                                       "streaming_mode": "completed_response_as_sse" if body.get("stream") else "nonstreaming"})
                    completion = {"id": "chatcmpl-" + request_id, "object": "chat.completion", "created": int(created),
                                  "model": SERVING_MODEL, "choices": [{"index": 0, "message": {"role": "assistant", "content": response["content"]},
                                  "finish_reason": "stop", "logprobs": None}], "usage": dict(UNKNOWN_USAGE),
                                  "session_provenance": provenance}
                    atomic_json(directory / "completion.json", completion)
                    atomic_json(directory / "state.json", {"status": "completed", "deadline_unix": deadline})
                    return completion
                self.closed.wait(min(self.poll_interval, max(0, deadline - time.time())))
            atomic_json(directory / "state.json", {"status": "timeout", "deadline_unix": deadline})
            raise TimeoutError("No actual session worker response arrived before the bounded deadline")
        except TimeoutError:
            raise
        except (ProtocolError, ValueError, OSError):
            if directory is not None:
                atomic_json(directory / "state.json", {"status": "error"})
            raise
        finally:
            self.pending.release()


def _route(path: str, default_run: str):
    parsed = urlsplit(path)
    if parsed.query or parsed.fragment:
        raise ProtocolError("Query strings are unsupported")
    if parsed.path in ("/v1/chat/completions", "/chat/completions"):
        return default_run
    match = re.fullmatch(r"/runs/([A-Za-z0-9_-]{1,96})/v1/chat/completions", parsed.path)
    if match:
        return match.group(1)
    raise ProtocolError("Only chat completions are supported; embeddings and arbitrary routes are unavailable")


def make_server(bridge: Bridge, host="127.0.0.1", port=8765):
    if host not in ("127.0.0.1", "::1"):
        raise ProtocolError("The bridge may bind only to a literal localhost address")
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, format, *args):
            pass  # Never print authorization headers or prompt text.
        def _send(self, status, value):
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
        def _authorized(self):
            supplied = self.headers.get("Authorization", "")
            return hmac.compare_digest(supplied.encode(), ("Bearer " + bridge.secret).encode())
        def do_GET(self):
            if not self._authorized():
                return self._send(401, {"error": {"message": "Unauthorized", "type": "authentication_error"}})
            if self.path != "/health":
                return self._send(404, {"error": {"message": "Unsupported route"}})
            self._send(200, {"session_id": bridge.session_id, "actual_serving_model": SERVING_MODEL,
                             "reasoning_effort": REASONING_EFFORT, "transport": TRANSPORT,
                             "queued_requests": len(pending_requests(bridge.directory)), "supervised_workers_required": True})
        def do_POST(self):
            if not self._authorized():
                return self._send(401, {"error": {"message": "Unauthorized", "type": "authentication_error"}})
            try:
                run_id = _route(self.path, bridge.run_id)
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
                body = json.loads(raw, object_pairs_hook=_object_pairs,
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
                common.update({"object": "chat.completion.chunk", "usage": None})
                chunks = [{**common, "choices": [{"index": 0, "delta": {"role": "assistant", "content": completion["choices"][0]["message"]["content"]}, "finish_reason": None}]},
                          {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
                if body.get("stream_options", {}).get("include_usage"):
                    chunks.append({**common, "choices": [], "usage": dict(UNKNOWN_USAGE)})
                for chunk in chunks:
                    self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
            except TimeoutError:
                self._send(504, {"error": {"message": "Session worker deadline expired", "type": "session_timeout"}})
            except (ProtocolError, ValueError, OSError):
                self._send(400, {"error": {"message": "Unsupported or invalid session bridge request/response", "type": "invalid_request_error"}})
    class LocalServer(ThreadingHTTPServer):
        daemon_threads = True
        address_family = socket.AF_INET6 if host == "::1" else socket.AF_INET
    server = LocalServer((host, int(port)), Handler)
    return server


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
    for command in ("pending", "claim", "respond"):
        child = sub.add_parser(command)
        child.add_argument("--session", type=Path, required=True)
        if command != "pending":
            child.add_argument("--request-id", required=True)
            child.add_argument("--worker-id", required=True)
        if command == "respond":
            child.add_argument("--content-file", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "serve":
        bridge = Bridge(args.spool, os.environ.get(SECRET_ENV, ""), session_id=args.session_id,
                        run_id=args.run_id, timeout=args.timeout, max_pending=args.max_pending)
        server = make_server(bridge, args.host, args.port)
        authority = "[" + args.host + "]" if ":" in args.host else args.host
        print(json.dumps({"session_directory": str(bridge.directory), "base_url": f"http://{authority}:{server.server_port}/v1",
                          "actual_serving_model": SERVING_MODEL, "reasoning_effort": REASONING_EFFORT,
                          "transport": TRANSPORT, "secret_environment_variable": SECRET_ENV}), flush=True)
        try:
            server.serve_forever()
        finally:
            bridge.closed.set()
            server.server_close()
    elif args.command == "pending":
        print(json.dumps(pending_requests(args.session), indent=2))
    elif args.command == "claim":
        print(json.dumps(claim_request(args.session, args.request_id, args.worker_id), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(submit_response(args.session, args.request_id, args.worker_id, args.content_file.read_text())))


if __name__ == "__main__":
    main()
