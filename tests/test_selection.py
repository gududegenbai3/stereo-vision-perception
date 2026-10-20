# -*- coding: utf-8 -*-
import numpy as np

from stereo_min.selection import budget_pixels, select_random, select_top_k


def test_budget_count_is_exact():
    m = np.random.default_rng(0).random((13, 17)).astype(np.float32)
    for frac in (0.0, 0.01, 0.1, 0.5, 1.0):
        assert int(select_top_k(m, frac).sum()) == budget_pixels(m, frac)


def test_top_k_picks_largest_uncertainty():
    m = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = select_top_k(m, 0.1)
    assert mask[9, 5] and mask[9, 9]
    assert not mask[0, 0]


def test_selection_is_deterministic():
    m = np.random.default_rng(3).random((30, 30)).astype(np.float32)
    assert np.array_equal(select_top_k(m, 0.2), select_top_k(m, 0.2))


def test_random_strategy_count_and_seed():
    m = np.zeros((20, 20), dtype=np.float32)
    a = select_random(m, 0.25, np.random.default_rng(7))
    b = select_random(m, 0.25, np.random.default_rng(7))
    assert int(a.sum()) == 100
    assert np.array_equal(a, b)
