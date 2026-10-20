# -*- coding: utf-8 -*-
import numpy as np
import pytest

from stereo_min.config import GeometryCfg
from stereo_min.depth import depth_stats, disparity_to_depth, valid_mask

GEO = GeometryCfg(fx_px=1000.0, baseline_m=0.1)


def test_depth_formula():
    disp = np.array([[10.0]], dtype=np.float32)
    depth, valid = disparity_to_depth(disp, GEO)
    assert valid[0, 0]
    assert depth[0, 0] == pytest.approx(10.0)   # 1000 * 0.1 / 10


def test_invalid_disparities_give_nan():
    disp = np.array([[0.0, -1.0], [np.nan, np.inf]], dtype=np.float32)
    depth, valid = disparity_to_depth(disp, GEO)
    assert not valid.any()
    assert np.isnan(depth).all()


def test_depth_requires_geometry():
    with pytest.raises(ValueError):
        disparity_to_depth(np.ones((2, 2), dtype=np.float32), GeometryCfg())


def test_valid_mask_margin():
    disp = np.ones((10, 10), dtype=np.float32)
    assert int(valid_mask(disp).sum()) == 100
    assert int(valid_mask(disp, roi_margin_px=2).sum()) == 36


def test_depth_stats():
    depth = np.array([[1.0, 2.0], [np.nan, np.nan]], dtype=np.float32)
    stats = depth_stats(depth)
    assert stats["depth_valid_ratio"] == pytest.approx(0.5)
    assert stats["depth_m_median"] == pytest.approx(1.5)
