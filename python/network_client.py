"""TCP client for communicating with the Isaac Lua mod."""

import json
import logging
import socket
import time
from collections.abc import Callable

try:
    import orjson

    def _loads(data: str | bytes):
        return orjson.loads(data)

    def _dumps(data: dict) -> str:
        return orjson.dumps(data).decode()

except ImportError:  # pragma: no cover - fallback when orjson is unavailable
    _loads = json.loads

    def _dumps(data: dict) -> str:
        return json.dumps(data, separators=(",", ":"))

log = logging.getLogger("NetworkClient")


class NetworkClient:
    """Manages the TCP socket lifecycle for Isaac mod communication.

    Handles connect, disconnect, reconnect with exponential backoff,
    JSON-newline framed send/receive, and buffer flushing.
    """

    MAX_RECONNECT_RETRIES = 5
    RECONNECT_BACKOFF_BASE = 1.0  # seconds, doubles each retry

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float,
        on_connect: Callable[["NetworkClient"], None] | None = None,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self.sock_file = None
        self.connection_id = 0
        self._on_connect = on_connect

    def set_on_connect(self, on_connect: Callable[["NetworkClient"], None] | None) -> None:
        """Set a callback that runs after every successful TCP connect/reconnect."""
        self._on_connect = on_connect

    def connect(self):
        if self.sock is not None:
            return
        last_err = None
        for attempt in range(self.MAX_RECONNECT_RETRIES + 1):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                self.sock_file = self.sock.makefile("r")
                self.connection_id += 1
                if self._on_connect is not None:
                    try:
                        self._on_connect(self)
                    except Exception:
                        self.disconnect()
                        raise
                log.info("Connected to %s:%d", self.host, self.port)
                return
            except (ConnectionError, OSError) as e:
                last_err = e
                self.disconnect()
                if attempt < self.MAX_RECONNECT_RETRIES:
                    delay = self.RECONNECT_BACKOFF_BASE * (2 ** attempt)
                    log.warning(
                        "Connect failed to %s:%d (%s), retry %d/%d in %.1fs...",
                        self.host, self.port, e,
                        attempt + 1, self.MAX_RECONNECT_RETRIES, delay,
                    )
                    time.sleep(delay)
        raise ConnectionError(
            f"Failed to connect to {self.host}:{self.port} "
            f"after {self.MAX_RECONNECT_RETRIES + 1} attempts: {last_err}"
        )

    def disconnect(self):
        """Clean up socket state."""
        if self.sock_file:
            try:
                self.sock_file.close()
            except OSError:
                pass
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.sock_file = None

    def reconnect(self):
        """Disconnect and reconnect (retry logic is in connect)."""
        self.disconnect()
        self.connect()

    def send(self, data: dict):
        self.connect()
        msg = _dumps(data) + "\n"
        try:
            self.sock.sendall(msg.encode())
        except (ConnectionError, OSError) as e:
            if isinstance(e, (TimeoutError, socket.timeout)):
                raise
            log.warning("Send failed (%s), attempting reconnect", e)
            self.reconnect()
            self.sock.sendall(msg.encode())

    def receive(self) -> dict:
        self.connect()
        try:
            line = self.sock_file.readline()
            if not line:
                raise ConnectionError("Connection closed by game")
            return _loads(line)
        except (ConnectionError, OSError) as e:
            if isinstance(e, (TimeoutError, socket.timeout)):
                raise
            log.warning("Receive failed (%s), attempting reconnect", e)
            self.reconnect()
            line = self.sock_file.readline()
            if not line:
                raise ConnectionError("Connection closed by game after reconnect")
            return _loads(line)

    def flush(self) -> int:
        """Flush stale data from the TCP buffer. Returns bytes flushed."""
        self.connect()
        flushed_bytes = 0
        self.sock.setblocking(False)
        try:
            while True:
                data = self.sock.recv(65536)
                if not data:
                    break
                flushed_bytes += len(data)
        except (BlockingIOError, OSError):
            pass
        self.sock.setblocking(True)
        self.sock.settimeout(self.timeout)
        self.sock_file = self.sock.makefile("r")
        return flushed_bytes
