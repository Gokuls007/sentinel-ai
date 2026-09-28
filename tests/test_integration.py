"""End-to-end: the real models on the bundled corridor sample must flag the doorways.

Slow (about 30 s on CPU) and downloads the sample + YOLO weights on first run:
    pytest -m slow
"""

import cv2
import pytest

pytestmark = pytest.mark.slow


def test_corridor_sample_produces_door_intrusions(tmp_config):
    from core.pipeline import SentinelPipeline
    from core.samples import SAMPLES, ensure_sample
    from main import apply_demo_config

    path = ensure_sample("corridor")
    apply_demo_config(tmp_config, SAMPLES["corridor"].config_path)
    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    tmp_config.frame_width, tmp_config.frame_height = w, h
    tmp_config.detector.device = "cpu"
    pipeline = SentinelPipeline(tmp_config)

    alerts, i = [], 0
    while i < 300:                                   # first 25 s of the clip
        ok, frame = cap.read()
        if not ok:
            break
        alerts += pipeline.process_frame(frame, 1000.0 + i / fps).alerts
        i += 1
    cap.release()
    pipeline.clip_recorder.flush()

    types = [a.alert_type for a in alerts]
    assert types.count("zone_intrusion") >= 2
    assert "fall" not in types                       # nobody falls in this clip
    events = pipeline.event_logger.get_events(limit=100)
    assert len(events) == len(alerts)
