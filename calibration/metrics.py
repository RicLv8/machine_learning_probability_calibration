"""Evaluation metrics and reliability-diagram statistics (NumPy only)."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
EPS = 1e-15  # clipping used only inside log_loss (same convention as sklearn)

def accuracy(y: FloatArray, p: FloatArray, threshold: float = 0.5) -> float:
    return float(np.mean((np.asarray(p) >= threshold).astype(int) == np.asarray(y).astype(int)))

def log_loss(y: FloatArray, p: FloatArray, eps: float = EPS) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def brier_score(y: FloatArray, p: FloatArray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))

def reliability_curve(y: FloatArray, p: FloatArray, n_bins: int = 10):
    """Equal-width bins on [0,1]. Returns (mean_pred, frac_pos, counts) for NON-empty bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)  # p == 1.0 goes in the last bin
    counts = np.bincount(bins, minlength=n_bins)
    sum_p = np.bincount(bins, weights=p, minlength=n_bins)
    sum_y = np.bincount(bins, weights=y, minlength=n_bins)
    ok = counts > 0
    return sum_p[ok] / counts[ok], sum_y[ok] / counts[ok], counts[ok]

def expected_calibration_error(y: FloatArray, p: FloatArray, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b / n) * |frac_pos_b - mean_pred_b|."""
    mp, fp, c = reliability_curve(y, p, n_bins)
    return float(np.sum(c / c.sum() * np.abs(fp - mp)))
