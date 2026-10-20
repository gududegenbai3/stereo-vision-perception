# -*- coding: utf-8 -*-
import numpy as np
import pytest

from stereo_min import metrics as M


def test_perfect_disparity():
    gt = np.full((8, 8), 4.0, dtype=np.float32)
    out = M.disparity_metrics(gt, gt)
    assert out["epe"] == pytest.approx(0.0)
    assert out["coverage"] == pytest.approx(1.0)
    assert out["bad1"] == pytest.approx(0.0)
    assert out["eval_pixels"] == 64


def test_invalid_prediction_not_mixed_into_mean():
    gt = np.full((4, 4), 5.0, dtype=np.float32)
    disp = np.zeros((4, 4), dtype=np.float32)
    disp[:, :2] = 5.0
    out = M.disparity_metrics(disp, gt)
    assert out["epe"] == pytest.approx(0.0)
    assert out["coverage"] == pytest.approx(0.5)


def test_gt_invalid_pixels_ignored():
    disp = np.array([[5.0, 100.0]], dtype=np.float32)
    gt = np.array([[5.0, -1.0]], dtype=np.float32)
    out = M.disparity_metrics(disp, gt)
    assert out["gt_pixels"] == 1
    assert out["epe"] == pytest.approx(0.0)


def test_bad_thresholds_counted():
    gt = np.full((2, 2), 10.0, dtype=np.float32)
    disp = np.array([[10.0, 11.5], [13.0, 20.0]], dtype=np.float32)
    out = M.disparity_metrics(disp, gt, bad_thresholds=(1.0, 3.0))
    # 误差为 0 / 1.5 / 3.0 / 10.0：>1 有 3 个，>3 只有 1 个（3.0 不算超过）
    assert out["bad1"] == pytest.approx(0.75)
    assert out["bad3"] == pytest.approx(0.25)


def test_empty_overlap_returns_nan():
    gt = np.full((2, 2), 3.0, dtype=np.float32)
    disp = np.zeros((2, 2), dtype=np.float32)
    out = M.disparity_metrics(disp, gt)
    assert out["eval_pixels"] == 0
    assert np.isnan(out["epe"])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        M.disparity_metrics(np.zeros((2, 2), np.float32), np.zeros((2, 3), np.float32))
