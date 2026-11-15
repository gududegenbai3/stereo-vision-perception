"""Eight-path pixel-BT SGM experiment, not a bit-exact OpenCV implementation.

Hirschmueller 2008 sections 2.1-2.3, equations 13-15. No learned components,
MI, multibaseline fusion or gap filling. All intensity arrays use uint8 0..255;
cost arrays are float32 (H,W,D), disparities float32 (H,W), in pixels.
"""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from .model import DisparityResult, RectificationGeometry, RectifiedPair

PenaltyMode = Literal["fixed", "adaptive", "matched_mean"]
DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))
INVALID_COST: float = 60000.0


@dataclass(frozen=True)
class SgmSettings:
    min_disparity: int
    num_disparities: int
    p1: float
    p2: float
    lr_tolerance_px: float


@dataclass(frozen=True)
class SgmOutput:
    raw: DisparityResult
    final: DisparityResult
    mean_p2_left: float


def validate_settings(settings: SgmSettings) -> None:
    """This experiment supports a nonnegative left-reference search interval."""
    if type(settings.min_disparity) is not int or settings.min_disparity < 0:
        raise ValueError("Nonnegative integer min_disparity required")
    if type(settings.num_disparities) is not int or settings.num_disparities < 3:
        raise ValueError("At least three disparity candidates required")
    if not np.isfinite([settings.p1, settings.p2, settings.lr_tolerance_px]).all():
        raise ValueError("Nonfinite SGM settings")
    if not 0 < settings.p1 <= settings.p2 or settings.lr_tolerance_px < 0:
        raise ValueError("Require 0<P1<=P2 and nonnegative LR tolerance")


def bt_cost_volume(left: np.ndarray, right: np.ndarray, settings: SgmSettings) -> np.ndarray:
    """Symmetric pixel BT via horizontal half-pixel interpolation; no wraparound."""
    validate_settings(settings)
    if left.shape != right.shape or left.ndim != 2 or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("BT requires equal uint8 grayscale (H,W)")
    height, width = left.shape
    if settings.min_disparity + settings.num_disparities >= width:
        raise ValueError("Search range too wide")
    a: np.ndarray = left.astype(np.float32)
    b: np.ndarray = right.astype(np.float32)
    def envelope(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        before: np.ndarray = np.concatenate((image[:, :1], image[:, :-1]), axis=1)
        after: np.ndarray = np.concatenate((image[:, 1:], image[:, -1:]), axis=1)
        return (np.minimum(image, np.minimum((image + before) * .5, (image + after) * .5)),
                np.maximum(image, np.maximum((image + before) * .5, (image + after) * .5)))
    amin, amax = envelope(a)
    bmin, bmax = envelope(b)
    cost: np.ndarray = np.full((height, width, settings.num_disparities), INVALID_COST, np.float32)
    for index in range(settings.num_disparities):
        disparity: int = settings.min_disparity + index
        end: int = width - disparity
        av: np.ndarray = a[:, disparity:]
        bv: np.ndarray = b[:, :end]
        first: np.ndarray = np.maximum(np.maximum(av - bmax[:, :end], bmin[:, :end] - av), 0)
        second: np.ndarray = np.maximum(np.maximum(bv - amax[:, disparity:], amin[:, disparity:] - bv), 0)
        cost[:, disparity:, index] = np.minimum(first, second)
    return cost


def penalty_values(current: np.ndarray, previous: np.ndarray, settings: SgmSettings) -> np.ndarray:
    """P2>=P1; max(1,delta) is an explicit finite zero-gradient convention."""
    delta: np.ndarray = np.abs(current.astype(np.float32) - previous.astype(np.float32))
    return np.maximum(settings.p1, settings.p2 / np.maximum(delta, 1)).astype(np.float32)


def mean_adaptive_penalty(gray: np.ndarray, settings: SgmSettings) -> float:
    """Compute an input-only scalar control over all valid directed neighbours."""
    total: float = 0.0
    count: int = 0
    height, width = gray.shape
    for dx, dy in DIRECTIONS:
        x0, x1 = max(0, dx), min(width, width + dx)
        y0, y1 = max(0, dy), min(height, height + dy)
        values: np.ndarray = penalty_values(gray[y0:y1, x0:x1], gray[y0-dy:y1-dy, x0-dx:x1-dx], settings)
        total += float(values.sum(dtype=np.float64))
        count += values.size
    return total / count


def path_step(cost: np.ndarray, previous: np.ndarray, p1: float, p2: np.ndarray | float) -> np.ndarray:
    """Equation 13 for independent frontier pixels, each row has D candidates."""
    best: np.ndarray = previous.min(axis=1)
    transition: np.ndarray = previous.copy()
    np.minimum(transition[:, 1:], previous[:, :-1] + p1, out=transition[:, 1:])
    np.minimum(transition[:, :-1], previous[:, 1:] + p1, out=transition[:, :-1])
    limit: np.ndarray = best + p2
    np.minimum(transition, limit[:, None], out=transition)
    return cost + transition - best[:, None]


def aggregate_paths(cost: np.ndarray, gray: np.ndarray, settings: SgmSettings, mode: PenaltyMode) -> tuple[np.ndarray, float]:
    """Vectorize independent rays along a row/column frontier; preserve 8 directions."""
    if mode not in ("fixed", "adaptive", "matched_mean"):
        raise ValueError("Unknown penalty mode")
    height, width, disparities = cost.shape
    result: np.ndarray = np.zeros_like(cost)
    mean_p2: float = mean_adaptive_penalty(gray, settings)
    scalar: float = mean_p2 if mode == "matched_mean" else settings.p2
    for dx, dy in DIRECTIONS:
        if dy == 0:
            order: range = range(width) if dx > 0 else range(width - 1, -1, -1)
            previous: np.ndarray = np.zeros((height, disparities), np.float32)
            for i, x in enumerate(order):
                if i == 0:
                    current: np.ndarray = cost[:, x, :].copy()
                else:
                    p2: np.ndarray | float = penalty_values(gray[:, x], gray[:, x-dx], settings) if mode == "adaptive" else scalar
                    current = path_step(cost[:, x, :], previous, settings.p1, p2)
                result[:, x, :] += current
                previous = current
        else:
            order = range(height) if dy > 0 else range(height - 1, -1, -1)
            previous = np.zeros((width, disparities), np.float32)
            x0, x1 = max(0, dx), min(width, width + dx)
            for i, y in enumerate(order):
                current = cost[y].copy()
                if i > 0:
                    p2 = penalty_values(gray[y, x0:x1], gray[y-dy, x0-dx:x1-dx], settings) if mode == "adaptive" else scalar
                    current[x0:x1] = path_step(cost[y, x0:x1], previous[x0-dx:x1-dx], settings.p1, p2)
                result[y] += current
                previous = current
    return result, mean_p2


def match_one(left: np.ndarray, right: np.ndarray, settings: SgmSettings, mode: PenaltyMode) -> tuple[np.ndarray, float]:
    """WTA and bounded parabolic subpixel fit. Grayscale uint8 in, float32 px out."""
    cost: np.ndarray = bt_cost_volume(left, right, settings)
    score, mean_p2 = aggregate_paths(cost, left, settings, mode)
    del cost
    best: np.ndarray = np.argmin(score, axis=2)
    y, x = np.indices(best.shape)
    middle: np.ndarray = score[y, x, best]
    before: np.ndarray = score[y, x, np.maximum(best - 1, 0)]
    after: np.ndarray = score[y, x, np.minimum(best + 1, settings.num_disparities - 1)]
    denominator: np.ndarray = before + after - 2 * middle
    offset: np.ndarray = np.zeros_like(middle)
    eligible: np.ndarray = (best > 0) & (best < settings.num_disparities - 1) & (denominator > 1e-6)
    offset[eligible] = np.clip((before[eligible] - after[eligible]) / (2 * denominator[eligible]), -.5, .5)
    return (best.astype(np.float32) + settings.min_disparity + offset).astype(np.float32), mean_p2


def lr_mask(left: np.ndarray, right: np.ndarray, left_valid: np.ndarray, right_valid: np.ndarray, tolerance_px: float) -> np.ndarray:
    """Both disparities are positive x_left-x_right; sample x_right=x_left-dL."""
    rows, columns = np.indices(left.shape)
    coordinate: np.ndarray = columns - left
    inside: np.ndarray = left_valid & np.isfinite(coordinate) & (coordinate >= 0) & (coordinate <= left.shape[1]-1)
    nearest: np.ndarray = np.rint(np.where(inside, coordinate, 0)).astype(np.int32)
    return inside & right_valid[rows, nearest] & (np.abs(left - right[rows, nearest]) <= tolerance_px)


def compute_sgm_disparity(pair: RectifiedPair, geometry: RectificationGeometry, settings: SgmSettings, mode: PenaltyMode) -> SgmOutput:
    """Fixed original-size Middlebury ROI; raw and LR-filtered results, no GT input."""
    validate_settings(settings)
    width, height = geometry.image_size
    if any(image.dtype != np.uint8 or image.shape != (height, width, 3) for image in (pair.left_bgr, pair.right_bgr)):
        raise ValueError("Expected matching uint8 BGR images")
    full: tuple[int, int, int, int] = (0, 0, width, height)
    if geometry.roi_left != full or geometry.roi_right != full:
        raise ValueError("Experimental backend currently supports full-image rectification ROIs only")
    roi = cv2.getValidDisparityROI(full, full, settings.min_disparity, settings.num_disparities, 1)
    x, y, rw, rh = roi
    if rw <= 0 or rh <= 0:
        raise ValueError("Empty SGM search ROI")
    left_gray: np.ndarray = cv2.cvtColor(pair.left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray: np.ndarray = cv2.cvtColor(pair.right_bgr, cv2.COLOR_BGR2GRAY)
    disparity, mean_p2 = match_one(left_gray, right_gray, settings, mode)
    valid: np.ndarray = np.zeros((height, width), bool)
    valid[y:y+rh, x:x+rw] = True
    raw_values: np.ndarray = np.where(valid, disparity, np.nan).astype(np.float32)
    raw: DisparityResult = DisparityResult(raw_values, valid.copy(), roi)
    left_median: np.ndarray = cv2.medianBlur(disparity, 3)
    flipped_right, _ = match_one(np.fliplr(right_gray).copy(), np.fliplr(left_gray).copy(), settings, mode)
    right_median: np.ndarray = np.fliplr(cv2.medianBlur(flipped_right, 3)).copy()
    right_valid: np.ndarray = np.fliplr(valid).copy()
    final_valid: np.ndarray = lr_mask(left_median, right_median, valid, right_valid, settings.lr_tolerance_px)
    values: np.ndarray = np.where(final_valid, left_median, np.nan).astype(np.float32)
    return SgmOutput(raw, DisparityResult(values, final_valid, roi), mean_p2)
