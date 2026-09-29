import datetime as dt
from pathlib import Path

import pytest

from src.reconstruction.pipeline import resolve_matcher
from src.utils.config import AppConfig, ConfigError, config_from_dict, load_config
from src.utils.paths import create_run, list_runs, next_run_id


def test_default_config_file_loads():
    config = load_config()
    assert config.capture.mode in ("orbit_camera", "turntable")
    assert config.reconstruction.max_image_size > 0


def test_overrides_are_merged():
    config = load_config(overrides={"reconstruction": {"matcher": "exhaustive", "dense": {"enabled": False}}})
    assert config.reconstruction.matcher == "exhaustive"
    assert config.reconstruction.dense.enabled is False
    assert config.reconstruction.dense.window_radius == 5  # untouched nested default


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="Unknown config key"):
        config_from_dict({"video": {"sample_fsp": 2}})


def test_invalid_value_rejected():
    with pytest.raises(ConfigError, match="interval_seconds"):
        config_from_dict({"video": {"interval_seconds": 0}})


def test_round_trip():
    config = AppConfig()
    assert config_from_dict(config.to_dict()) == config


def test_run_ids_increment(tmp_path: Path):
    runs_dir = tmp_path / "runs with space"
    day = dt.date(2026, 9, 29)
    assert next_run_id(runs_dir, day) == "2026-09-29_001"
    first = create_run(runs_dir, day)
    second = create_run(runs_dir, day)
    assert (first.run_id, second.run_id) == ("2026-09-29_001", "2026-09-29_002")
    assert second.frames_dir.is_dir() and second.logs_dir.is_dir()
    assert [r.run_id for r in list_runs(runs_dir)] == ["2026-09-29_002", "2026-09-29_001"]
    assert next_run_id(runs_dir, dt.date(2026, 9, 30)) == "2026-09-30_001"


@pytest.mark.parametrize(
    "matcher,kind,count,expected",
    [
        ("auto", "video", 50, "sequential"),
        ("auto", "photos", 50, "exhaustive"),
        ("auto", "mixed", 50, "exhaustive"),
        ("auto", "photos", 500, "sequential"),
        ("exhaustive", "video", 50, "exhaustive"),
    ],
)
def test_resolve_matcher(matcher, kind, count, expected):
    cfg = AppConfig().reconstruction
    cfg.matcher = matcher
    assert resolve_matcher(cfg, kind, count) == expected
