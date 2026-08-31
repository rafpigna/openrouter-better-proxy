"""SSE Log Handler — thread-safe log streaming for dashboard clients.

Architecture:
  - logging.Handler.emit() writes formatted records into a thread-safe
    deque (recent buffer) and into per-client queue.Queue instances.
  - The SSE endpoint polls its queue with a short timeout (0.5s) and
    yields log events as SSE data frames.
  - A heartbeat comment is sent every 30s per client to keep the
    connection alive through proxies / load balancers.
"""

import asyncio
import json
import logging
import queue
import threading
import time
from collections import deque


class SSELogHandler(logging.Handler):
    """Thread-safe logging handler that buffers records for SSE clients.

    Max 20 simultaneous clients. Each client gets a queue.Queue(maxsize=500).
    Slow clients (queue full) are silently dropped to prevent blocking the
    logging path.
    """

    MAX_CLIENTS = 20
    CLIENT_QUEUE_MAXSIZE = 500
    RECENT_BUFFER_SIZE = 100
    HEARTBEAT_INTERVAL = 30  # seconds
    SSE_POLL_TIMEOUT = 0.5  # seconds

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self._lock = threading.Lock()
        # Circular buffer of recent log records (thread-safe under lock)
        self._recent: deque[str] = deque(maxlen=self.RECENT_BUFFER_SIZE)
        # Per-client queues: cid -> queue.Queue
        self._clients: dict[int, queue.Queue] = {}
        self._next_cid = 0

    # ------------------------------------------------------------------
    # logging.Handler interface  (called from ARBITRARY threads)
    # ------------------------------------------------------------------

    def emit(self, record: logging.LogRecord) -> None:
        """Format and dispatch a log record to all connected clients.

        The SSE frame is built HERE, once per record, with the timestamp
        taken from record.created (the moment the record was logged),
        NOT from the moment a client receives/replays it. Both the
        recent buffer and the per-client queues carry ready-made frames,
        so replayed records keep their original time.
        """
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover
            self.handleError(record)
            return

        frame = format_log_record(message, record.created)

        with self._lock:
            self._recent.append(frame)
            stale: list[int] = []
            for cid, q in self._clients.items():
                try:
                    q.put_nowait(frame)
                except queue.Full:
                    stale.append(cid)
            for cid in stale:
                del self._clients[cid]

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    def register_client(self) -> tuple[int, queue.Queue]:
        """Register a new SSE client.

        Returns (cid, queue) where queue is pre-filled with recent log records.
        Raises RuntimeError if max clients reached.
        """
        q: queue.Queue = queue.Queue(maxsize=self.CLIENT_QUEUE_MAXSIZE)

        with self._lock:
            if len(self._clients) >= self.MAX_CLIENTS:
                raise RuntimeError(
                    f"Max SSE clients reached ({self.MAX_CLIENTS})"
                )
            cid = self._next_cid
            self._next_cid += 1
            self._clients[cid] = q

        # Seed recent buffer (no lock needed — we own the queue)
        with self._lock:
            recent_snapshot = list(self._recent)
        for record in recent_snapshot:
            try:
                q.put_nowait(record)
            except queue.Full:
                break  # recent buffer may be trimmed

        return cid, q

    def remove_client(self, cid: int) -> None:
        """Disconnect an SSE client."""
        with self._lock:
            self._clients.pop(cid, None)

    def disconnect_all(self) -> None:
        """Disconnect every SSE client (used at shutdown)."""
        with self._lock:
            self._clients.clear()

    def get_recent(self) -> list[str]:
        """Snapshot of the recent buffer (for diagnostics)."""
        with self._lock:
            return list(self._recent)


def format_log_record(record: str, created: float | None = None) -> str:
    """Wrap a log message as an SSE data frame.

    `created` is the LogRecord.created epoch (seconds). When given, the
    frame timestamp reflects when the record was LOGGED; when omitted
    (legacy callers), it falls back to the current time.
    """
    safe = record.replace("\n", "\\n")
    ts = time.localtime(created) if created is not None else time.localtime()
    payload = json.dumps(
        {"t": time.strftime("%Y-%m-%d %H:%M:%S", ts), "m": safe}
    )
    return f"data: {payload}\n\n"


async def sse_log_generator(
    log_handler: SSELogHandler,
) -> str:
    """Async generator for the /api/logs SSE endpoint.

    Yields SSE-formatted log records and heartbeats.
    The caller is responsible for catching CancelledError in the
    finally block.
    """
    cid, log_queue = log_handler.register_client()

    loop = asyncio.get_running_loop()
    poll_timeout = SSELogHandler.SSE_POLL_TIMEOUT
    heartbeat_interval = SSELogHandler.HEARTBEAT_INTERVAL

    last_heartbeat = time.monotonic()

    try:
        while True:
            try:
                # Non-blocking poll with a short timeout,
                #    # running inside the default executor to avoid blocking the
                # event loop on queue.get().
                # The queue carries READY-MADE SSE frames (built in emit()
                # with the record's original timestamp) — yield verbatim.
                frame = await loop.run_in_executor(
                    None, log_queue.get, True, poll_timeout
                )
                yield frame
                last_heartbeat = time.monotonic()
            except queue.Empty:
                # No new records —    check if a heartbeat is due.
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_interval:
                    yield ": heartbeat SSE\n\n"
                    last_heartbeat = now
                continue
    except GeneratorExit:
        # Client disconnected the SSE stream
        raise
    finally:
        log_handler.remove_client(cid)