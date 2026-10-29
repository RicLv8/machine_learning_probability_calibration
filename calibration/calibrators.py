"""Platt scaling and isotonic regression (PAV), implemented from scratch with NumPy only.

Both calibrators map a real-valued classifier score f(x) to a probability P(y=1 | f).
They MUST be fitted on data that was not used to train the underlying classifier.
"""
from __future__ import annotations
import logging
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]

def _sigmoid(z: FloatArray) -> FloatArray:
    """Numerically stable logistic function 1 / (1 + exp(-z))."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

class PlattScaler:
    """Platt (1999) sigmoid calibration:  P(y=1|f) = 1 / (1 + exp(A f + B)).

    A, B are found by minimising the negative log-likelihood (paper Eq. 2) on a
    calibration set, using Newton's method with backtracking line search.
    Targets are smoothed (paper Eq. 4):  y+ = (N+ + 1)/(N+ + 2),  y- = 1/(N- + 2)
    to limit overfitting of the sigmoid.

    With z = A f + B and p = sigmoid(-z):  dNLL/dz = t - p,  d2NLL/dz2 = p (1 - p).
    """
    def __init__(self, max_iter: int = 100, tol: float = 1e-14, ridge: float = 1e-12) -> None:
        self.max_iter, self.tol, self.ridge = max_iter, tol, ridge
        self.A_: float = 0.0
        self.B_: float = 0.0

    @staticmethod
    def _nll(z: FloatArray, t: FloatArray) -> float:
        # -log p = softplus(z), -log(1-p) = softplus(-z)   (stable via logaddexp)
        return float(np.sum(t * np.logaddexp(0.0, z) + (1.0 - t) * np.logaddexp(0.0, -z)))

    def fit(self, scores: FloatArray, y: FloatArray) -> "PlattScaler":
        f = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(y).ravel().astype(int)
        if f.shape != y.shape:
            raise ValueError("scores and y must have the same length")
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Platt scaling needs both classes in the calibration set")
        t = np.where(y == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))

        A, B = 0.0, float(np.log((n_neg + 1.0) / (n_pos + 1.0)))  # Platt's initialisation
        loss = self._nll(A * f + B, t)
        it = 0
        for it in range(self.max_iter):
            p = _sigmoid(-(A * f + B))
            r = t - p
            grad = np.array([np.sum(r * f), np.sum(r)])           # gradient of NLL wrt (A, B)
            w = p * (1.0 - p)
            H = np.array([[np.sum(w * f * f), np.sum(w * f)],
                          [np.sum(w * f),     np.sum(w)]]) + self.ridge * np.eye(2)
            step = np.linalg.solve(H, grad)
            if abs(grad @ step) < self.tol:                        # Newton decrement -> converged
                break
            lam = 1.0                                              # backtracking line search
            while lam > 1e-12:
                A_new, B_new = A - lam * step[0], B - lam * step[1]
                new_loss = self._nll(A_new * f + B_new, t)
                if new_loss < loss - 1e-4 * lam * (grad @ step):   # Armijo condition
                    break
                lam *= 0.5
            else:
                break
            A, B, loss = A_new, B_new, new_loss
        self.A_, self.B_ = float(A), float(B)
        logger.debug("Platt fit: A=%.4f B=%.4f nll=%.4f iters=%d", self.A_, self.B_, loss, it + 1)
        return self

    def predict_proba(self, scores: FloatArray) -> FloatArray:
        return _sigmoid(-(self.A_ * np.asarray(scores, dtype=float) + self.B_))

class IsotonicCalibrator:
    """Isotonic regression via the Pair-Adjacent-Violators (PAV) algorithm (paper Table 1).

    Finds the non-decreasing, piecewise-constant m(f) minimising sum (y_i - m(f_i))^2.
    Implemented with a stack of pooled blocks (amortised O(n) after sorting).
    Tied scores are first merged (they must receive the same value).
    """
    def __init__(self) -> None:
        self.upper_edges_: FloatArray = np.array([])
        self.values_: FloatArray = np.array([])

    def fit(self, scores: FloatArray, y: FloatArray) -> "IsotonicCalibrator":
        f = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if f.shape != y.shape:
            raise ValueError("scores and y must have the same length")
        uniq, inv = np.unique(f, return_inverse=True)   # sorted unique scores, ties merged
        sums = np.bincount(inv, weights=y)
        wts = np.bincount(inv).astype(float)

        means: list[float] = []     # stack of pooled blocks: mean, weight, upper score edge
        weights: list[float] = []
        edges: list[float] = []
        for u, s, w in zip(uniq, sums, wts):
            means.append(s / w); weights.append(w); edges.append(float(u))
            while len(means) > 1 and means[-2] >= means[-1]:        # violator -> pool the pair
                w_new = weights[-2] + weights[-1]
                m_new = (weights[-2] * means[-2] + weights[-1] * means[-1]) / w_new
                means[-2:], weights[-2:] = [m_new], [w_new]
                edges[-2:] = [edges[-1]]
        self.values_ = np.asarray(means)
        self.upper_edges_ = np.asarray(edges)
        logger.debug("Isotonic fit: %d unique scores -> %d blocks", len(uniq), len(means))
        return self

    def predict_proba(self, scores: FloatArray) -> FloatArray:
        """Step function: value of the block whose upper edge is the first >= score
        (scores above the largest training score take the last block's value)."""
        f = np.asarray(scores, dtype=float).ravel()
        idx = np.searchsorted(self.upper_edges_, f, side="left")
        return self.values_[np.clip(idx, 0, len(self.values_) - 1)]
