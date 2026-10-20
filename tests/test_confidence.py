# -*- coding: utf-8 -*-
import numpy as np

from stereo_min.confidence import LARGE, lrc_pair


def test_consistent_disparities_have_zero_uncertainty():
    # 视差恒为 3 时，最左侧 3 列在右图中找不到对应点，因此只检查有对应的区域
    d = np.full((5, 9), 3.0, dtype=np.float32)
    u, valid, err = lrc_pair(d, d, thresh=1.0)
    assert np.all(u[:, 3:] == 0.0)
    assert np.all(err[:, 3:] == 0.0)
    assert valid[:, 3:].all()
    assert not valid[:, :3].any()


def test_inconsistent_pixel_is_invalid():
    # 左图像素 x=5 对应右参考图的 x-d=1，把那一处改坏才能触发不一致
    d_l = np.full((5, 9), 3.0, dtype=np.float32)
    d_r = d_l.copy()
    d_r[2, 2] = 30.0
    u, valid, _ = lrc_pair(d_l, d_r, thresh=1.0)
    assert not valid[2, 5]
    assert u[2, 5] == LARGE


def test_border_and_missing_predictions_are_invalid():
    d_l = np.zeros((4, 6), dtype=np.float32)   # 全无效
    d_r = np.zeros((4, 6), dtype=np.float32)
    _, valid, _ = lrc_pair(d_l, d_r)
    assert not valid.any()


def test_shape_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        lrc_pair(np.zeros((4, 4), np.float32), np.zeros((4, 5), np.float32))
