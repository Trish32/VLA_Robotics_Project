"""How the client behaves when the server fails, times out, or answers oddly.

The failure path had a hole that the happy path could never show: upstream's
`PolicyServer` catches every handler exception and replies with an ordinary msgpack dict
`{"error": str(e)}`. The client passed that straight through, so a server-side crash
arrived at the node looking like an action — and the node reported "no usable action keys
in reply", which is true, unhelpful, and hides the real message. Found by pointing the
bridge at the real DiVLA server and sending it a bad observation
(`DexVLA_Robotics/tests/test_ros_bridge_against_real_divla.py`); see ../../bug_log.txt [4].

No rclpy here, so it runs on the Mac as well as in the container.
"""

import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_bridge import wire  # noqa: E402

PORT = 5597  # distinct from the end-to-end test's 5599


class ScriptedServer(threading.Thread):
    """Replies with a queued payload per request. Encodes with our own serializer,
    which `test_wire_matches_upstream.py` proves is byte-identical to upstream's."""

    daemon = True

    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)
        self.ready = threading.Event()
        self._stop_evt = threading.Event()

    def run(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://*:{PORT}")
        sock.setsockopt(zmq.RCVTIMEO, 200)
        self.ready.set()
        while not self._stop_evt.is_set():
            try:
                sock.recv()
            except zmq.error.Again:
                continue
            payload = self.replies.pop(0) if self.replies else {"ok": True}
            sock.send(payload if isinstance(payload, bytes) else wire.to_bytes(payload))
        sock.close(linger=0)
        ctx.term()

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)


@pytest.fixture
def serve():
    servers = []

    def start(replies):
        server = ScriptedServer(replies)
        server.start()
        assert server.ready.wait(timeout=5)
        servers.append(server)
        return wire.PolicyClient("localhost", PORT, timeout_ms=2000)

    yield start
    for server in servers:
        server.stop()


def test_server_side_error_dict_raises_with_the_servers_message(serve):
    client = serve([{"error": "observation has no 'video' entries"}])

    with pytest.raises(RuntimeError, match="observation has no 'video' entries"):
        client.get_action({"video": {}})


def test_the_endpoint_name_is_in_the_error(serve):
    """Which call failed matters when three endpoints are in play at startup."""
    client = serve([{"error": "boom"}])

    with pytest.raises(RuntimeError, match="get_modality_config"):
        client.get_modality_config()


def test_an_action_that_merely_contains_an_error_key_is_not_treated_as_a_failure(serve):
    """Only a lone `error` key is the server's failure envelope.

    A policy is free to return diagnostics alongside its action, and swallowing those
    would turn a working robot into a silent one.
    """
    chunk = np.zeros((4, 7), np.float32)
    client = serve([[{"action.joints": chunk, "error": 0.02}, {}]])

    action = client.get_action({"video": {}})

    assert set(action) == {"action.joints", "error"}


def test_the_action_info_pair_is_unwrapped(serve):
    chunk = np.zeros((4, 7), np.float32)
    client = serve([[{"action.joints": chunk}, {"reasoning": "pick it up"}]])

    action = client.get_action({"video": {}})

    assert set(action) == {"action.joints"}, "the info half must not leak through"


def test_the_error_sentinel_still_raises(serve):
    """Some server versions send the bare b'ERROR' sentinel instead of a dict."""
    client = serve([b"ERROR"])

    with pytest.raises(RuntimeError, match="ERROR"):
        client.call_endpoint("get_action", {"observation": {}})


def test_a_timeout_raises_and_leaves_the_client_usable():
    """A ZMQ REQ socket is unusable after a timeout, so the client rebuilds it.

    Without that, one slow reply wedges the node permanently: no further action ever
    arrives, and the arm keeps executing the last chunk it was given.
    """
    # Nothing is listening one port up, so the first call must time out.
    client = wire.PolicyClient("localhost", PORT + 1, timeout_ms=300)

    with pytest.raises(TimeoutError):
        client.get_action({"video": {}})
    assert client.ping() is False, "must answer rather than raise after a timeout"
    client.close()
