"""Alert notifications: Telegram, email (SMTP) and webhook behind one ``Notifier`` interface."""

from __future__ import annotations

import logging

from notifications.base import Notification, Notifier, NotifierError, summarize
from notifications.dispatcher import NotificationDispatcher

logger = logging.getLogger("sentinel.notifications")

__all__ = [
    "Notification",
    "NotificationDispatcher",
    "Notifier",
    "NotifierError",
    "build_notifiers",
    "summarize",
]


def build_notifiers(config) -> list[Notifier]:
    """Every channel whose settings are present in ``config`` (a ``SentinelConfig``)."""
    from notifications.email_smtp import EmailNotifier
    from notifications.telegram import TelegramNotifier
    from notifications.webhook import WebhookNotifier

    n = config.notifications
    notifiers: list[Notifier] = []
    if n.telegram_enabled:
        notifiers.append(TelegramNotifier(n.telegram_bot_token, n.telegram_chat_id))
    if n.email_enabled:
        notifiers.append(
            EmailNotifier(
                n.smtp_host, n.smtp_port, n.smtp_from, n.smtp_to,
                username=n.smtp_user, password=n.smtp_password,
                starttls=n.smtp_starttls, use_ssl=n.smtp_ssl,
            )
        )
    if config.output.webhook_url:
        notifiers.append(WebhookNotifier(config.output.webhook_url))
    logger.info("Notification channels: %s", ", ".join(x.name for x in notifiers) or "none")
    return notifiers
