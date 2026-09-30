"""Email channel over SMTP, with the thumbnail attached. For Gmail use an app password."""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage

from notifications.base import Notification, Notifier, NotifierError


class EmailNotifier(Notifier):
    name = "email"

    def __init__(
        self,
        host: str,
        port: int,
        sender: str,
        recipients: list[str],
        *,
        username: str = "",
        password: str = "",
        starttls: bool = True,
        use_ssl: bool = False,
        timeout: float = 20.0,
        smtp_factory=None,
    ):
        if not host or not sender or not recipients:
            raise ValueError("EmailNotifier needs SMTP_HOST, SMTP_FROM and SMTP_TO")
        self.host, self.port = host, port
        self.sender, self.recipients = sender, recipients
        self.username, self._password = username, password
        self.starttls, self.use_ssl, self.timeout = starttls, use_ssl, timeout
        self._factory = smtp_factory

    def _message(self, n: Notification) -> EmailMessage:
        msg = EmailMessage()
        msg["Subject"] = n.title
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        msg.set_content(n.text)
        if n.image_path and os.path.isfile(n.image_path):
            with open(n.image_path, "rb") as f:
                msg.add_attachment(
                    f.read(), maintype="image", subtype="jpeg", filename=os.path.basename(n.image_path)
                )
        return msg

    def _connect(self):
        if self._factory:
            return self._factory(self.host, self.port, self.timeout)
        if self.use_ssl:
            return smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout,
                                    context=ssl.create_default_context())
        return smtplib.SMTP(self.host, self.port, timeout=self.timeout)

    def send(self, notification: Notification) -> None:
        try:
            with self._connect() as smtp:
                if self.starttls and not self.use_ssl:
                    smtp.starttls(context=ssl.create_default_context())
                if self.username:
                    smtp.login(self.username, self._password)
                smtp.send_message(self._message(notification))
        except (smtplib.SMTPException, OSError) as exc:
            message = str(exc).replace(self._password, "***") if self._password else str(exc)
            raise NotifierError(f"email to {self.host}:{self.port} failed: {message}") from None
