# -*- coding: utf-8 -*-
import numpy as np
import pytest

from stereo_min import metrics as M


def test_perfect_prediction():
    disp = np.full((8, 8), 4.0, dtype=np.float32)
    gt = np.full((8, 8), 4.0, dtype=np.float32)
    s = M.summarize(disp, gt)
    assert s["epe"] == pytest.approx(0.0)
    assert s["coverage"] == pytest.approx(1.0)
    assert s["d1"] == pytest.approx(0.0)


def test_missing_prediction_is_penalised():
    disp = np.zeros((4, 4), dtype=np.float32)
    gt = np.full((4, 4), 5.0, dtype=np.float32)
    s = M.summarize(disp, gt, penalty=3.0)
    assert s["coverage"] == pytest.approx(0.0)
    assert s["epe"] == pytest.approx(3.0)


def test_gt_invalid_pixels_are_ignored():
    disp = np.array([[5.0, 100.0]], dtype=np.float32)
    gt = np.array([[5.0, -1.0]], dtype=np.float32)
    s = M.summarize(disp, gt)
    assert s["epe"] == pytest.approx(0.0)
    assert s["gt_pixels"] == 1


def test_risk_coverage_shapes_and_masking_effect():
    rng = np.random.default_rng(0)
    gt = np.full((20, 20), 1.0, dtype=np.float32)
    # 不确定度大的地方误差也大 → 按不确定性剔除应当降低风险
    u = rng.random((20, 20)).astype(np.float32)
    disp = (1.0 + 5.0 * u).astype(np.float32)
    err = M.error_map(disp, gt)
    cov, risk = M.risk_coverage(u, err, gt > 0, levels=10)
    assert cov.shape == risk.shape == (10,)
    assert np.all(np.diff(cov) > 0)
    assert risk[0] < risk[-1]
    assert np.isfinite(M.aurc(cov, risk))


def test_region_summarize():
    disp = np.zeros((2, 2), dtype=np.float32)
    gt = np.full((2, 2), 2.0, dtype=np.float32)
    region = np.array([[True, True], [False, False]])
    out = M.region_summarize(disp, gt, region, penalty=2.0)
    assert out["pixels"] == 2
    assert out["epe"] == pytest.approx(2.0)
