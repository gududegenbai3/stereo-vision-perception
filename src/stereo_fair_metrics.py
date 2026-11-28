"""Prediction-independent domains and coverage-aware stereo evaluation contracts."""

import cv2
import numpy as np


def evaluation_domains(arrays: dict[str, np.ndarray], maximum: int) -> dict[str, np.ndarray]:
    """Bool HW domains. Fixed ROI uses largest prespecified block (7), no output masks."""
    truth: np.ndarray = arrays["disparity"]
    height, width = truth.shape
    known: np.ndarray = arrays["known"] & np.isfinite(truth) & (truth > 0) & (truth < maximum) & (np.arange(width)[None, :]-truth >= 0)
    nonocc: np.ndarray = known & arrays["non_occluded"]
    roi: np.ndarray = np.zeros_like(known)
    x, y, w, h = cv2.getValidDisparityROI((0, 0, width, height), (0, 0, width, height), 0, maximum, 7)
    if w > 0 and h > 0:
        roi[y:y+h, x:x+w] = True
    return {"all_known": known, "non_occluded": nonocc, "fixed_roi": nonocc & roi}


def score_prediction(prediction: np.ndarray, truth: np.ndarray, domain: np.ndarray, cap: float = 20.) -> dict[str, object]:
    """Float64 errors; all missing values fail every threshold, never disappear from rates."""
    if prediction.shape != truth.shape or domain.shape != truth.shape or domain.dtype != np.bool_:
        raise ValueError("Prediction, GT and boolean domain must share HW dimensions")
    count: int = int(domain.sum())
    if count == 0:
        return {"reference": 0, "evaluated": 0, "missing": 0, "coverage": None, "mae_px": None, "bad2_or_missing": None}
    valid: np.ndarray = domain & np.isfinite(prediction)
    error: np.ndarray = np.abs(prediction[valid].astype(np.float64)-truth[valid].astype(np.float64))
    missing: int = count-error.size
    result: dict[str, object] = {"reference": count, "evaluated": int(error.size), "missing": missing, "coverage": error.size/count,
        "mae_px": float(error.mean()) if error.size else None, "median_error_px": float(np.median(error)) if error.size else None,
        "rmse_px": float(np.sqrt(np.mean(error**2))) if error.size else None,
        "capped_mae20_or_missing": float((np.minimum(error, cap).sum()+cap*missing)/count),
        "d1_or_missing": float(((error > np.maximum(3., .05*truth[valid]))).sum()+missing)/count}
    for threshold in (1, 2, 3, 5):
        result[f"bad{threshold}_or_missing"] = (int((error > threshold).sum())+missing)/count
    return result


def scene_source_summary(records: list[dict[str, object]], domain: str, metric: str) -> dict[str, object]:
    """Average observations within conservative group, groups within source, sources equally."""
    values: dict[str, dict[str, list[float]]] = {}
    for row in records:
        value: float | None = row["metrics"][domain].get(metric)
        if value is not None:
            values.setdefault(row["dataset"], {}).setdefault(row["group"], []).append(value)
    grouped: dict[str, dict[str, float]] = {source: {group: float(np.mean(points)) for group, points in groups.items()} for source, groups in values.items()}
    per_source: dict[str, float] = {source: float(np.mean(list(groups.values()))) for source, groups in grouped.items()}
    return {"source_equal_macro": float(np.mean(list(per_source.values()))) if per_source else None, "per_source": per_source, "per_group": grouped}


def paired_bootstrap(first: dict[str, dict[str, float]], second: dict[str, dict[str, float]], seed: int, repeats: int) -> dict[str, object]:
    """Paired source-stratified component bootstrap; descriptive for this fixed model pair."""
    if set(first) != set(second):
        raise ValueError("Paired sources must match")
    rng: np.random.Generator = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    differences: list[float] = []
    for source in sorted(first):
        if set(first[source]) != set(second[source]):
            raise ValueError("Paired components must match")
        delta: np.ndarray = np.array([first[source][group]-second[source][group] for group in sorted(first[source])], np.float64)
        sampled.append(rng.choice(delta, (repeats, len(delta)), replace=True).mean(axis=1))
        differences.extend(delta.tolist())
    distribution: np.ndarray = np.mean(sampled, axis=0)
    return {"difference_source_equal_macro": float(np.mean([np.mean([first[source][group]-second[source][group] for group in first[source]]) for source in first])),
        "component_bootstrap_95": np.quantile(distribution, [.025, .975]).tolist(),
        "component_wins": int((np.array(differences) < 0).sum()), "component_losses": int((np.array(differences) > 0).sum()),
        "component_ties": int((np.array(differences) == 0).sum()), "components": len(differences),
        "limit": "Conditional descriptive interval; conservative groups are not independently verified physical sites, one checkpoint each"}


def jump_band(truth_px: np.ndarray, threshold_px: float, radius_px: int) -> np.ndarray:
    """Dilate finite-neighbour GT jumps; unknown GT never creates jump seeds."""
    if truth_px.ndim != 2 or truth_px.dtype != np.float32:
        raise ValueError("GT must be float32 (H,W), px")
    if not np.isfinite(threshold_px) or threshold_px <= 0 or type(radius_px) is not int or radius_px < 0:
        raise ValueError("Invalid jump threshold/radius")
    finite: np.ndarray = np.isfinite(truth_px)
    safe: np.ndarray = np.where(finite, truth_px, 0)
    horizontal: np.ndarray = finite[:, 1:] & finite[:, :-1] & (np.abs(safe[:, 1:] - safe[:, :-1]) > threshold_px)
    vertical: np.ndarray = finite[1:, :] & finite[:-1, :] & (np.abs(safe[1:, :] - safe[:-1, :]) > threshold_px)
    seeds: np.ndarray = np.zeros_like(finite)
    seeds[:, 1:] |= horizontal
    seeds[:, :-1] |= horizontal
    seeds[1:, :] |= vertical
    seeds[:-1, :] |= vertical
    kernel: np.ndarray = np.ones((2 * radius_px + 1, 2 * radius_px + 1), np.uint8)
    return cv2.dilate(seeds.astype(np.uint8), kernel).astype(bool)


def texture_std(left_bgr: np.ndarray, window_px: int) -> np.ndarray:
    """Population std of grayscale intensity in a reflected-border local window."""
    if left_bgr.ndim != 3 or left_bgr.shape[2] != 3 or left_bgr.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR (H,W,3)")
    if type(window_px) is not int or window_px < 1 or window_px % 2 != 1:
        raise ValueError("Texture window must be positive odd integer")
    gray: np.ndarray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean: np.ndarray = cv2.boxFilter(gray, -1, (window_px, window_px), borderType=cv2.BORDER_REFLECT_101)
    square_mean: np.ndarray = cv2.boxFilter(gray * gray, -1, (window_px, window_px), borderType=cv2.BORDER_REFLECT_101)
    return np.sqrt(np.maximum(square_mean - mean * mean, 0)).astype(np.float32)
