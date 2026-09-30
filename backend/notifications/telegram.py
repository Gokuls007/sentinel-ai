"""Telegram Bot API channel: the alert's thumbnail with the summary as its caption.

Create a bot with @BotFather (gives TELEGRAM_BOT_TOKEN), send it any message, then read your
chat id from https://api.telegram.org/bot<token>/getUpdates (TELEGRAM_CHAT_ID).
The token is part of every request URL, so error messages are scrubbed of it before they
are raised or logged.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import requests

from notifications.base import Notification, Notifier, NotifierError

API = "https://api.telegram.org"
CAPTION_LIMIT = 1024  # Telegram's limit for photo captions
MESSAGE_LIMIT = 4096


class TelegramNotifier(Notifier):
    name = "telegram"

    def __init__(self, bot_token: str, chat_id: str, *, timeout: float = 15.0, session: Any = None):
        if not bot_token or not chat_id:
            raise ValueError("TelegramNotifier needs a bot token and a chat id")
        self._token = bot_token
        self._chat_id = str(chat_id)
        self._timeout = timeout
        self._session = session or requests.Session()

    def _url(self, method: str) -> str:
        return f"{API}/bot{self._token}/{method}"

    def _scrub(self, text: str) -> str:
        return text.replace(self._token, "***")

    def send(self, notification: Notification) -> None:
        text = f"{notification.title}\n\n{notification.text}"
        image = notification.image_path
        try:
            if image and os.path.isfile(image):
                with open(image, "rb") as photo:
                    response = self._session.post(
                        self._url("sendPhoto"),
                        data={"chat_id": self._chat_id, "caption": text[:CAPTION_LIMIT]},
                        files={"photo": (os.path.basename(image), photo, "image/jpeg")},
                        timeout=self._timeout,
                    )
            else:
                response = self._session.post(
                    self._url("sendMessage"),
                    data={"chat_id": self._chat_id, "text": text[:MESSAGE_LIMIT]},
                    timeout=self._timeout,
                )
        except requests.RequestException as exc:
            raise NotifierError(self._scrub(f"Telegram request failed: {exc}")) from None
        if response.status_code != 200:
            detail = ""
            with contextlib.suppress(ValueError):
                detail = response.json().get("description", "")
            raise NotifierError(self._scrub(f"Telegram returned {response.status_code}: {detail}"))
