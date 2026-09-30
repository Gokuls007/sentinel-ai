"""JSON webhook channel (Slack/Discord bridges, n8n, custom services): POSTs the event."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from notifications.base import Notification, Notifier, NotifierError


class WebhookNotifier(Notifier):
    name = "webhook"

    def __init__(self, url: str, timeout: float = 5.0):
        if not url:
            raise ValueError("WebhookNotifier needs a URL")
        self.url = url
        self.timeout = timeout

    def send(self, notification: Notification) -> None:
        body = {
            **notification.event.to_alert_dict(),
            "event": notification.event.to_dict(),
            "title": notification.title,
            "text": notification.text,
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body, default=str).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "User-Agent": "sentinel-ai"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()
        except (urllib.error.URLError, OSError) as exc:
            raise NotifierError(f"webhook POST failed: {exc}") from None
