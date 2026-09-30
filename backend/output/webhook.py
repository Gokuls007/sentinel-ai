"""Posts alerts as JSON to an HTTP webhook (Slack/Discord-compatible bridges, n8n, custom services).

Delivery happens on a background thread so a slow endpoint never stalls the video pipeline.
If the endpoint falls behind, the oldest undelivered alerts are dropped (and logged).
"""

import contextlib
import json
import logging
import queue
import threading
import urllib.request

logger = logging.getLogger("sentinel.webhook")


class WebhookNotifier:
    def __init__(self, url: str, timeout: float = 5.0, max_pending: int = 100):
        self.url = url
        self.timeout = timeout
        self._queue: queue.Queue[dict | None] = queue.Queue(maxsize=max_pending)
        self.sent = 0
        self.failed = 0
        self.dropped = 0
        self._thread = threading.Thread(target=self._worker, name="webhook", daemon=True)
        self._thread.start()

    def send(self, payload: dict) -> None:
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            self.dropped += 1
            logger.warning("Webhook queue full; dropping alert %s", payload.get("alert_id"))

    def _post(self, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "sentinel-ai"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            resp.read()

    def _worker(self) -> None:
        while True:
            payload = self._queue.get()
            if payload is None:
                return
            try:
                self._post(payload)
                self.sent += 1
            except Exception as e:
                self.failed += 1
                logger.warning("Webhook delivery failed: %s", e)

    def close(self, timeout: float = 5.0) -> None:
        with contextlib.suppress(queue.Full):
            self._queue.put(None, timeout=timeout)
        self._thread.join(timeout)
