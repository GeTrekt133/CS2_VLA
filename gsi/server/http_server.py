"""
HTTP Server for CS2 Game State Integration.

Receives POST requests from CS2 with game state JSON data.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Callable, Optional, List
import json
import threading
import time
import signal
from dataclasses import dataclass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in a new thread."""
    allow_reuse_address = True
    daemon_threads = True


@dataclass
class GSIPayload:
    """Parsed GSI payload from CS2."""

    timestamp: float
    raw_data: dict

    # Core data sections
    provider: dict
    player: Optional[dict]
    map_info: Optional[dict]
    round_info: Optional[dict]
    allplayers: Optional[dict]
    bomb: Optional[dict]
    phase_countdowns: Optional[dict]

    @classmethod
    def from_json(cls, data: dict) -> "GSIPayload":
        """Create GSIPayload from raw JSON data."""
        return cls(
            timestamp=time.time(),
            raw_data=data,
            provider=data.get("provider", {}),
            player=data.get("player"),
            map_info=data.get("map"),
            round_info=data.get("round"),
            allplayers=data.get("allplayers"),
            bomb=data.get("bomb"),
            phase_countdowns=data.get("phase_countdowns"),
        )

    def is_in_game(self) -> bool:
        """Check if player is currently in a game."""
        return self.player is not None and self.map_info is not None


class GSIRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for GSI POST requests."""

    # Class-level callback and auth (set by GSIServer)
    callback: Callable[[GSIPayload], None] = None
    auth_token: Optional[str] = None
    verbose: bool = False

    def log_message(self, format, *args):
        """Suppress default HTTP logging unless verbose."""
        if self.verbose:
            print(f"[GSI HTTP] {format % args}")

    def do_POST(self):
        """Handle incoming GSI data."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_response(400)
            self.end_headers()
            return

        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode("utf-8"))

            # Validate auth token if configured
            if self.auth_token:
                auth = data.get("auth", {})
                if auth.get("token") != self.auth_token:
                    self.send_response(401)
                    self.end_headers()
                    return

            # Parse payload
            payload = GSIPayload.from_json(data)

            # Call registered callback
            if self.callback:
                try:
                    self.callback(payload)
                except Exception as e:
                    print(f"[GSI] Callback error: {e}")

            self.send_response(200)
            self.end_headers()

        except json.JSONDecodeError as e:
            print(f"[GSI] JSON decode error: {e}")
            self.send_response(400)
            self.end_headers()

    def do_GET(self):
        """Health check endpoint."""
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"GSI Server Running")


class GSIServer:
    """
    GSI HTTP Server.

    Receives game state updates from CS2 and dispatches to handlers.

    Usage:
        server = GSIServer(port=3000)
        server.register_callback(my_handler)
        server.start()
        ...
        server.stop()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 3000,
        auth_token: Optional[str] = None,
        verbose: bool = False,
    ):
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.verbose = verbose

        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._callbacks: List[Callable[[GSIPayload], None]] = []

        # Stats
        self._received_count = 0
        self._last_payload: Optional[GSIPayload] = None

    def register_callback(self, callback: Callable[[GSIPayload], None]):
        """Register a callback for GSI updates."""
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[GSIPayload], None]):
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _dispatch(self, payload: GSIPayload):
        """Dispatch payload to all registered callbacks."""
        self._received_count += 1
        self._last_payload = payload

        for callback in self._callbacks:
            try:
                callback(payload)
            except Exception as e:
                print(f"[GSIServer] Callback error: {e}")

    def start(self):
        """Start the GSI server in a background thread."""
        if self._running:
            print("[GSIServer] Already running")
            return

        # Configure handler class
        GSIRequestHandler.callback = self._dispatch
        GSIRequestHandler.auth_token = self.auth_token
        GSIRequestHandler.verbose = self.verbose

        self._server = ThreadedHTTPServer((self.host, self.port), GSIRequestHandler)
        self._server.socket.settimeout(1.0)  # Allow periodic checking for stop
        self._running = True

        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

        print(f"[GSIServer] Started on http://{self.host}:{self.port}")

    def _serve(self):
        """Server loop."""
        while self._running:
            try:
                self._server.handle_request()
            except Exception:
                pass  # Timeout or shutdown

    def stop(self):
        """Stop the GSI server."""
        self._running = False
        if self._server:
            try:
                self._server.server_close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print("[GSIServer] Stopped")

    @property
    def received_count(self) -> int:
        """Number of payloads received."""
        return self._received_count

    @property
    def last_payload(self) -> Optional[GSIPayload]:
        """Most recent payload."""
        return self._last_payload

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running


# Simple test
if __name__ == "__main__":
    def on_update(payload: GSIPayload):
        if payload.is_in_game():
            player = payload.player or {}
            print(f"HP: {player.get('state', {}).get('health', '?')}, "
                  f"Round: {payload.map_info.get('round', '?') if payload.map_info else '?'}")

    server = GSIServer(port=3000, verbose=True)
    server.register_callback(on_update)
    server.start()

    print("Press Ctrl+C to stop...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
