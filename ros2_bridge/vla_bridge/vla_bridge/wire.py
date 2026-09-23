"""Torch-free client for the GR00T PolicyServer wire protocol.

Why this exists rather than `from gr00t.policy import PolicyClient`: importing gr00t
drags in torch, transformers, diffusers and ~4GB of wheels. The ROS2 container has no
business carrying any of that — the model runs on the GPU box. The wire format is
msgpack over a ZMQ REQ socket, and it is small enough to restate exactly.

Mirrors `gr00t/policy/server_client.py::MsgSerializer` at commit 9b37aa1. If upstream
changes the encoding, `test_wire_matches_upstream.py` fails on the Mac (where gr00t IS
installed) and tells you before the robot does.

Dependencies: pyzmq, msgpack, numpy. No torch.
"""

from __future__ import annotations

import io
from typing import Any

import msgpack
import numpy as np
import zmq


def _encode(obj: Any) -> Any:
    """ndarray -> {'__ndarray_class__': True, 'as_npy': <npy bytes>}."""
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    # ModalityConfig is a server-side dataclass; the bridge only ever reads it as a
    # plain dict, so it is deliberately left undecoded rather than reimplemented.
    if "__ModalityConfig_class__" in obj:
        return obj["as_json"]
    return obj


def to_bytes(data: Any) -> bytes:
    return msgpack.packb(data, default=_encode)


def from_bytes(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_decode)


class PolicyClient:
    """Minimal REQ-socket client. One outstanding request at a time, by protocol.

    ZMQ REQ sockets are strict lockstep: a send must be followed by a recv before the
    next send, and a timeout leaves the socket unusable. Both cases rebuild the socket
    rather than wedging the node, which matters because a wedged policy client on a
    robot means no new actions and a arm still executing the last chunk.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 5000,
        api_token: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.context = zmq.Context.instance()
        self.socket: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        try:
            self.socket.send(to_bytes(request))
            reply = self.socket.recv()
        except zmq.error.ZMQError as exc:
            self._connect()  # REQ socket is unrecoverable after a timeout
            raise TimeoutError(f"policy server {self.host}:{self.port} did not reply") from exc

        if reply == b"ERROR":
            raise RuntimeError("policy server returned ERROR — check the server-side log")

        decoded = from_bytes(reply)
        # Upstream's PolicyServer catches every handler exception and replies with an
        # ordinary msgpack dict `{"error": str(e)}` — not the b"ERROR" sentinel above.
        # Passing that through means a server-side failure arrives looking like an action
        # dict: the node hands it to decode_chunk and reports "no usable action keys",
        # burying the real message. Raise it instead, with the server's own text.
        if isinstance(decoded, dict) and set(decoded) == {"error"}:
            raise RuntimeError(f"policy server error on {endpoint!r}: {decoded['error']}")
        return decoded

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except (TimeoutError, RuntimeError):
            return False

    def get_action(self, observation: dict, options: dict | None = None) -> dict:
        """Request one action chunk.

        Two details are dictated by upstream's `PolicyServer`, and both were wrong here
        until this client was first run against a real server rather than a fake:

        1. The server dispatches with `handler(**request["data"])`, so the payload must
           be `{"observation": ..., "options": ...}`. Sending the observation bare makes
           the server call `get_action(video=..., state=..., language=...)` and fail with
           an unexpected-keyword error.
        2. `BasePolicy.get_action` returns `(action, info)`, which msgpack delivers as a
           two-element list. The caller wants the action, so unwrap it here rather than
           leaving every consumer to index `[0]`.
        """
        reply = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        if isinstance(reply, (list, tuple)) and len(reply) == 2 and isinstance(reply[0], dict):
            return reply[0]
        return reply

    def get_modality_config(self) -> Any:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
