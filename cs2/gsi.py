"""Local HTTP POST receiver for CS2's official Game State Integration.

No third-party dependencies and no process-memory fallback. Each POST is a current
snapshot, not a patch; `previously` and `added` describe changes and are not state.
"""

import argparse
import copy
import hmac
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass(frozen=True)
class GSISnapshot:
    data: dict
    received_at: float
    sequence: int


class GSIStore:
    def __init__(self, history_size=512):
        if history_size < 1:
            raise ValueError("history_size must be positive")
        self._history = deque(maxlen=history_size)
        self._condition = threading.Condition()
        self._sequence = 0

    def publish(self, payload, *, received_at=None):
        if not isinstance(payload, dict):
            raise ValueError("GSI payload must be an object")
        provider = payload.get("provider")
        if not isinstance(provider, dict) or provider.get("appid") != 730:
            raise ValueError("Expected CS2 GSI provider.appid = 730")
        if received_at is not None and not math.isfinite(received_at):
            raise ValueError("received_at must be finite")
        data = copy.deepcopy({k: v for k, v in payload.items() if k not in ("auth", "previously", "added")})
        with self._condition:
            stamp = time.monotonic() if received_at is None else received_at
            if self._history and stamp < self._history[-1].received_at:
                raise ValueError("GSI receive timestamps must be nondecreasing")
            self._sequence += 1
            self._history.append(GSISnapshot(data, stamp, self._sequence))
            self._condition.notify_all()

    def latest(self):
        with self._condition:
            return copy.deepcopy(self._history[-1]) if self._history else None

    def at_or_before(self, timestamp):
        with self._condition:
            return next((copy.deepcopy(s) for s in reversed(self._history) if s.received_at <= timestamp), None)

    def wait_for_update(self, after_sequence=0, timeout=5.0):
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._history and self._history[-1].sequence > after_sequence, timeout)
            return copy.deepcopy(self._history[-1]) if ready else None


class GSIReceiver:
    """Explicit start/close lifecycle; loopback only. Port 0 is useful in tests."""

    def __init__(self, host="127.0.0.1", port=3000, *, token=None, store=None):
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("GSIReceiver only listens on loopback")
        self.host, self.port, self.token = host, port, token
        self.store = store if store is not None else GSIStore()
        self._server = self._thread = None

    @property
    def address(self):
        return self._server.server_address if self._server else (self.host, self.port)

    def start(self):
        if self._server is not None:
            return self
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(5.0)

            def log_message(self, *args):
                pass  # Never print auth tokens or Steam IDs from payloads.

            def reply(self, status):
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                if self.path != "/":
                    self.reply(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 1024 * 1024:
                        self.reply(413)
                        return
                    payload = json.loads(self.rfile.read(length), parse_constant=lambda _: None)
                    if not isinstance(payload, dict):
                        raise ValueError("Expected an object")
                    if receiver.token is not None:
                        auth = payload.get("auth")
                        token = auth.get("token") if isinstance(auth, dict) else None
                        if not isinstance(token, str) or not hmac.compare_digest(token.encode(), receiver.token.encode()):
                            self.reply(403)
                            return
                    receiver.store.publish(payload)
                except (ValueError, UnicodeError, OSError):
                    self.reply(400)
                    return
                self.reply(200)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, name="cs2-gsi", daemon=True)
        self._thread.start()
        return self

    def close(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join()
            self._server = self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()


def main():
    parser = argparse.ArgumentParser(description="Monitor official CS2 GSI; no game memory access.")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--token", help="Optional auth.token; must match the game's cfg")
    args = parser.parse_args()
    try:
        with GSIReceiver(port=args.port, token=args.token) as receiver:
            print(f"Waiting for CS2 GSI on http://127.0.0.1:{receiver.address[1]}/ (Ctrl+C to stop)")
            sequence = 0
            while True:
                snapshot = receiver.store.wait_for_update(sequence, timeout=1.0)
                if snapshot is not None:
                    sequence = snapshot.sequence
                    player = snapshot.data.get("player")
                    player = player if isinstance(player, dict) else {}
                    map_state = snapshot.data.get("map")
                    map_state = map_state if isinstance(map_state, dict) else {}
                    print(json.dumps({"sequence": sequence, "map": map_state.get("name"),
                                      "round": snapshot.data.get("round"), "state": player.get("state")}, ensure_ascii=False))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
