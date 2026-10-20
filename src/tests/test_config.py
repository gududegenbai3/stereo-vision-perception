# -*- coding: utf-8 -*-
import json
import tempfile
from pathlib import Path

import pytest

from stereo_min.config import ConfigError, GeometryCfg, SgbmCfg, load_config


def write_cfg(tmp: Path, payload: dict) -> Path:
    p = tmp / "cfg.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


VALID = {"sgbm": {"num_disparities": 64, "block_size": 7}}


def test_load_valid_config():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(write_cfg(Path(tmp), VALID))
    assert cfg.sgbm.num_disparities == 64
    assert cfg.sgbm.p1 == 8 * 49
    assert cfg.geometry.usable is False


def test_missing_file():
    with pytest.raises(ConfigError):
        load_config("not_exists.json")


def test_missing_sgbm_field():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ConfigError):
            load_config(write_cfg(Path(tmp), {"geometry": {}}))


def test_unknown_field_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ConfigError):
            load_config(write_cfg(Path(tmp), {"sgbm": {"num_disparities": 64, "unknown": 1}}))


def test_bad_num_disparities():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ConfigError):
            load_config(write_cfg(Path(tmp), {"sgbm": {"num_disparities": 100, "block_size": 7}}))


def test_bad_json():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(p)


def test_geometry_half_configured():
    with pytest.raises(ConfigError):
        GeometryCfg(fx_px=700.0).validate()


def test_sgbm_even_block_size():
    with pytest.raises(ConfigError):
        SgbmCfg(block_size=4).validate()
