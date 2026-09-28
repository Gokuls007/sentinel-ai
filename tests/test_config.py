import json

import pytest

import main
from config.settings import SentinelConfig

ENV_VARS = ["VIDEO_SOURCE", "LOOP_VIDEO", "TARGET_FPS", "FORCE_CPU", "CORS_ORIGINS", "PORT",
            "DETECTION_CONFIDENCE", "WEBHOOK_URL", "ZONES_FILE"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_without_env():
    cfg = SentinelConfig.from_env(env_file=None)
    assert cfg.source == "0" and cfg.loop is False and cfg.server.port == 8000


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("VIDEO_SOURCE", "clip.mp4")
    monkeypatch.setenv("LOOP_VIDEO", "yes")
    monkeypatch.setenv("TARGET_FPS", "12")
    monkeypatch.setenv("FORCE_CPU", "true")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("DETECTION_CONFIDENCE", "0.35")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test ,")
    monkeypatch.setenv("WEBHOOK_URL", "http://hook.test/x")
    cfg = SentinelConfig.from_env(env_file=None)
    assert cfg.source == "clip.mp4"
    assert cfg.loop is True
    assert cfg.target_fps == 12
    assert cfg.detector.device == "cpu"
    assert cfg.server.port == 9000
    assert cfg.detector.confidence_threshold == pytest.approx(0.35)
    assert cfg.cors_origins == ["http://a.test", "http://b.test"]
    assert cfg.output.webhook_url == "http://hook.test/x"


def test_empty_env_value_keeps_default(monkeypatch):
    monkeypatch.setenv("TARGET_FPS", "")
    assert SentinelConfig.from_env(env_file=None).target_fps == 25


@pytest.fixture
def fake_sample(monkeypatch, tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"")
    monkeypatch.setattr(main, "ensure_sample", lambda name: video)
    monkeypatch.setattr(SentinelConfig, "from_env", classmethod(lambda cls, env_file=".env": cls()))
    return video


def test_demo_flag_uses_sample_zones_and_overrides(fake_sample):
    cfg = main.build_config(main.parse_args(["--demo", "hallway"]))
    assert cfg.source == str(fake_sample)
    assert cfg.loop is True
    assert cfg.zone.zones_file.endswith("hallway_demo.json")
    assert cfg.loiter.time_threshold == 10.0          # from the scenario's overrides
    assert cfg.zone.alert_cooldown == 20.0


def test_source_demo_means_default_demo(fake_sample):
    cfg = main.build_config(main.parse_args(["--source", "demo"]))
    assert cfg.zone.zones_file.endswith("corridor_demo.json")


def test_env_video_source_demo(fake_sample, monkeypatch):
    monkeypatch.setattr(SentinelConfig, "from_env",
                        classmethod(lambda cls, env_file=".env": cls(source="demo")))
    cfg = main.build_config(main.parse_args([]))
    assert cfg.source == str(fake_sample)


def test_no_loop_flag_wins(fake_sample):
    cfg = main.build_config(main.parse_args(["--demo", "--no-loop", "--port", "8123"]))
    assert cfg.loop is False and cfg.server.port == 8123


def test_unknown_override_is_ignored(tmp_path):
    path = tmp_path / "demo.json"
    path.write_text(json.dumps({"zones": [], "overrides": {"loiter": {"nope": 1},
                                                           "missing": {"x": 1}}}))
    cfg = SentinelConfig()
    main.apply_demo_config(cfg, path)
    assert cfg.zone.zones_file == str(path)
    assert not hasattr(cfg.loiter, "nope")
