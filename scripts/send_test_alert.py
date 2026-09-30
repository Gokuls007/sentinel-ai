"""Send one test fall alert through the real notification path, to check your channels.

    python scripts/send_test_alert.py

It builds a clearly labelled TEST fall event, gives it a real thumbnail (a frame from the
corridor sample, with a banner), and pushes it through the same NotificationDispatcher the
pipeline uses, to every channel configured in .env (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID,
SMTP_*, WEBHOOK_URL). Nothing is written to the event database.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "backend"))

from config.settings import SentinelConfig
from core.samples import ensure_sample
from events import Event
from notifications import NotificationDispatcher, build_notifiers


def test_thumbnail() -> str:
    cap = cv2.VideoCapture(str(ensure_sample("corridor")))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read a frame from the corridor sample")
    cv2.rectangle(frame, (10, 10), (520, 50), (0, 0, 255), -1)
    cv2.putText(frame, "TEST ALERT: Sentinel AI notification check", (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    path = os.path.join(tempfile.mkdtemp(prefix="sentinel-test-"), "snapshot_TEST.jpg")
    cv2.imwrite(path, frame)
    return path


def main() -> int:
    cfg = SentinelConfig.from_env()
    notifiers = build_notifiers(cfg)
    if not notifiers:
        print("No notification channel is configured. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID\n"
              "(or SMTP_HOST/SMTP_FROM/SMTP_TO, or WEBHOOK_URL) to .env, then run this again.")
        return 1
    now = time.time()
    event = Event(
        type="fall", severity="critical", start_ts=now, end_ts=now, camera_id=cfg.camera_id,
        track_id=0, alert_id="ALT-TEST00",
        message="TEST: fall detected (notification check, not a real event)",
        confidence=1.0, thumbnail_path=test_thumbnail(),
    )
    dispatcher = NotificationDispatcher(notifiers, debounce_s=0, min_severity="low")
    dispatcher.handle(event)
    dispatcher.close(timeout=30)
    s = dispatcher.stats
    print(f"channels: {', '.join(n.name for n in notifiers)} | sent {s['sent']} | failed {s['failed']}")
    return 0 if s["failed"] == 0 and s["sent"] == len(notifiers) else 1


if __name__ == "__main__":
    raise SystemExit(main())
