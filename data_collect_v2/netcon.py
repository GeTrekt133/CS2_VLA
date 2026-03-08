"""
Netcon TCP Client — sends console commands to CS2 via -netconport.

CS2 launch option: -netconport 2121
Protocol: Plain TCP. Send UTF-8 command terminated by \\n.
"""

import socket
import time
import threading
from typing import Optional


class NetconClient:
    """TCP client for CS2's netcon protocol."""

    def __init__(self, host: str = "127.0.0.1", port: int = 2121, timeout: float = 5.0):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def connect(self, retries: int = 5, delay: float = 2.0) -> bool:
        """Connect to CS2 netcon with retry logic."""
        for attempt in range(retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self._timeout)
                sock.connect((self._host, self._port))
                self._sock = sock
                self._drain()
                print(f"[Netcon] Connected to {self._host}:{self._port}")
                return True
            except (ConnectionRefusedError, OSError, socket.timeout) as e:
                print(f"[Netcon] Connection attempt {attempt+1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
        print("[Netcon] FAILED: Could not connect to CS2 netcon. "
              "Ensure CS2 is running with -netconport 2121")
        return False

    def _drain(self):
        """Drain any pending data from the socket (non-blocking read)."""
        if self._sock is None:
            return
        self._sock.setblocking(False)
        try:
            while True:
                data = self._sock.recv(4096)
                if not data:
                    break
        except BlockingIOError:
            pass
        finally:
            self._sock.setblocking(True)
            self._sock.settimeout(self._timeout)

    def send(self, command: str) -> bool:
        """Send a console command. Returns True on success."""
        with self._lock:
            if self._sock is None:
                print(f"[Netcon] Not connected, cannot send: {command}")
                return False
            try:
                self._sock.sendall((command + "\n").encode("utf-8"))
                return True
            except (BrokenPipeError, OSError) as e:
                print(f"[Netcon] Send failed: {e}")
                self._sock = None
                return False

    def reconnect(self) -> bool:
        """Close and reconnect."""
        self.close()
        return self.connect()

    def close(self):
        """Close the TCP connection."""
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None

    @property
    def is_connected(self) -> bool:
        return self._sock is not None
