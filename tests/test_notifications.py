"""Notification channels (fake Telegram/SMTP/webhook servers), summaries, and debounce."""

import http.server
import json
import threading
import time
from typing import ClassVar

import pytest

from config.settings import SentinelConfig
from events import Event
from notifications import NotificationDispatcher, NotifierError, build_notifiers, summarize
from notifications.base import Notification, Notifier
from notifications.email_smtp import EmailNotifier
from notifications.telegram import TelegramNotifier
from notifications.webhook import WebhookNotifier


def fall(track_id=7, ts=1_790_000_000.0, thumb=None, **kw) -> Event:
    return Event(type="fall", severity="critical", start_ts=ts, end_ts=ts, track_id=track_id,
                 alert_id="ALT-FALL01", message="Fall detected: person down and not moving",
                 confidence=0.9, thumbnail_path=thumb, **kw)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Recorder(Notifier):
    name = "recorder"

    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def send(self, notification):
        if self.fail:
            raise NotifierError("boom")
        self.sent.append(notification)


# --- summary ------------------------------------------------------------------------


def test_summary_has_the_useful_facts():
    n = summarize(fall(zone_id="loading"))
    assert "fall" in n.title and "critical" in n.title
    for part in ("Fall detected", "Camera: cam-0", "Zone: loading", "track #7", "90%", "ALT-FALL01"):
        assert part in n.text


# --- debounce -------------------------------------------------------------------------


def test_one_notification_per_type_and_track_within_debounce():
    clock, rec = Clock(), Recorder()
    d = NotificationDispatcher([rec], debounce_s=60, min_severity="medium", clock=clock)
    d.handle(fall(track_id=1))
    d.handle(fall(track_id=1))                 # same key, inside the window
    d.handle(fall(track_id=2))                 # different person
    d.handle(Event(type="zone_intrusion", severity="high", start_ts=1, end_ts=1, track_id=1))  # other type
    clock.now = 61
    d.handle(fall(track_id=1))                 # window over
    d.flush()
    assert len(rec.sent) == 4
    assert d.stats["debounced"] == 1
    d.close()


def test_severity_filter():
    rec = Recorder()
    d = NotificationDispatcher([rec], min_severity="high", clock=Clock())
    d.handle(Event(type="loitering", severity="low", start_ts=1, end_ts=1))
    d.handle(Event(type="time_exceeded", severity="medium", start_ts=1, end_ts=1))
    d.handle(fall())
    d.flush()
    assert [n.event.type for n in rec.sent] == ["fall"]
    assert d.stats["below_severity"] == 2
    d.close()


def test_a_failing_channel_does_not_block_the_others():
    bad, good = Recorder(fail=True), Recorder()
    d = NotificationDispatcher([bad, good], clock=Clock())
    d.handle(fall())
    d.flush()
    assert len(good.sent) == 1 and d.stats["failed"] == 1 and d.stats["sent"] == 1
    d.close()


def test_handle_returns_immediately_even_if_a_channel_is_slow():
    class Slow(Recorder):
        def send(self, notification):
            time.sleep(0.5)
            super().send(notification)

    slow = Slow()
    d = NotificationDispatcher([slow], clock=Clock())
    started = time.perf_counter()
    d.handle(fall())
    assert time.perf_counter() - started < 0.1  # the pipeline never waits on the network
    d.flush()
    assert len(slow.sent) == 1
    d.close()


# --- Telegram ------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {"ok": True}

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.calls, self.response, self.exc = [], response or FakeResponse(), exc

    def post(self, url, **kwargs):
        files = kwargs.get("files")
        if files:  # read the upload now, while the file is still open
            name, handle, mime = files["photo"]
            kwargs = {**kwargs, "files": {"photo": (name, handle.read(), mime)}}
        self.calls.append((url, kwargs))
        if self.exc:
            raise self.exc
        return self.response


def test_telegram_sends_photo_with_caption_when_thumbnail_exists(tmp_path):
    thumb = tmp_path / "snapshot_ALT-FALL01.jpg"
    thumb.write_bytes(b"\xff\xd8\xff fake jpeg")
    session = FakeSession()
    TelegramNotifier("123:SECRET", "42", session=session).send(summarize(fall(thumb=str(thumb))))
    (url, kwargs), = session.calls
    assert url.endswith("/bot123:SECRET/sendPhoto")
    assert kwargs["data"]["chat_id"] == "42" and "Fall detected" in kwargs["data"]["caption"]
    assert kwargs["files"]["photo"][1].startswith(b"\xff\xd8\xff")


def test_telegram_falls_back_to_text_without_thumbnail():
    session = FakeSession()
    TelegramNotifier("123:SECRET", "42", session=session).send(summarize(fall()))
    assert session.calls[0][0].endswith("/sendMessage")


def test_telegram_errors_never_contain_the_token():
    import requests

    for session in (
        FakeSession(response=FakeResponse(401, {"description": "Unauthorized"})),
        FakeSession(exc=requests.ConnectionError("failed to reach https://api.telegram.org/bot123:SECRET/x")),
    ):
        with pytest.raises(NotifierError) as info:
            TelegramNotifier("123:SECRET", "42", session=session).send(summarize(fall()))
        assert "SECRET" not in str(info.value)


# --- email ------------------------------------------------------------------------------


class FakeSMTP:
    instances: ClassVar[list] = []

    def __init__(self, host, port, timeout):
        self.host, self.port, self.messages, self.logged_in, self.tls = host, port, [], None, False
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        self.tls = True

    def login(self, user, password):
        self.logged_in = (user, password)

    def send_message(self, msg):
        self.messages.append(msg)


def test_email_sends_summary_with_thumbnail_attached(tmp_path):
    thumb = tmp_path / "snap.jpg"
    thumb.write_bytes(b"\xff\xd8\xffjpeg")
    FakeSMTP.instances.clear()
    EmailNotifier("smtp.test", 587, "sentinel@test", ["ops@test"], username="u", password="p",
                  smtp_factory=FakeSMTP).send(summarize(fall(thumb=str(thumb))))
    smtp = FakeSMTP.instances[0]
    msg = smtp.messages[0]
    assert smtp.tls and smtp.logged_in == ("u", "p")
    assert msg["To"] == "ops@test" and "fall" in msg["Subject"]
    assert [p.get_filename() for p in msg.iter_attachments()] == ["snap.jpg"]


# --- webhook (real local HTTP server) ----------------------------------------------------


def test_webhook_posts_event_json():
    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    WebhookNotifier(f"http://127.0.0.1:{httpd.server_port}/hook").send(summarize(fall()))
    httpd.shutdown()
    (body,) = received
    assert body["alert_type"] == "fall" and body["event"]["track_id"] == 7 and body["title"]


# --- config --------------------------------------------------------------------------------


def test_channels_enabled_only_when_configured():
    cfg = SentinelConfig()
    assert build_notifiers(cfg) == []
    cfg.notifications.telegram_bot_token, cfg.notifications.telegram_chat_id = "1:x", "2"
    cfg.output.webhook_url = "http://hook.test"
    assert [n.name for n in build_notifiers(cfg)] == ["telegram", "webhook"]


def test_notification_settings_from_env(monkeypatch):
    for k, v in {"TELEGRAM_BOT_TOKEN": "1:abc", "TELEGRAM_CHAT_ID": "99", "NOTIFY_DEBOUNCE_S": "30",
                 "NOTIFY_MIN_SEVERITY": "high", "SMTP_HOST": "smtp.test", "SMTP_FROM": "a@test",
                 "SMTP_TO": "b@test, c@test", "CAMERA_ID": "dock-cam"}.items():
        monkeypatch.setenv(k, v)
    cfg = SentinelConfig.from_env(env_file=None)
    n = cfg.notifications
    assert n.telegram_enabled and n.email_enabled and n.debounce_s == 30 and n.min_severity == "high"
    assert n.smtp_to == ["b@test", "c@test"] and cfg.camera_id == "dock-cam"


def test_notification_type_is_plain_data():
    n = Notification(event=fall(), title="t", text="x")
    assert n.image_path is None
