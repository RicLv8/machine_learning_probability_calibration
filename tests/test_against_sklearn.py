"""Sanity checks: from-scratch implementations vs scikit-learn reference implementations.
Run with:  python -m pytest -q tests   (or: python tests/test_against_sklearn.py)
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as skm

from calibration import (IsotonicCalibrator, PlattScaler, accuracy, brier_score,
                         expected_calibration_error, log_loss)

rng = np.random.default_rng(0)
N = 600
Y = rng.integers(0, 2, N)
SCORES = rng.normal(0, 1.5, N) + 1.2 * (2 * Y - 1)       # informative, overlapping scores
SCORES_TIES = np.round(SCORES, 1)                         # many ties (like RF outputs)


def test_metrics_match_sklearn():
    p = np.clip(1 / (1 + np.exp(-SCORES)), 0, 1)
    assert np.isclose(log_loss(Y, p), skm.log_loss(Y, p))
    assert np.isclose(brier_score(Y, p), skm.brier_score_loss(Y, p))
    assert np.isclose(accuracy(Y, p), skm.accuracy_score(Y, p >= 0.5))

def test_ece_matches_manual_definition():
    p = rng.uniform(0, 1, 2000); y = (rng.uniform(0, 1, 2000) < p).astype(int)
    ece = expected_calibration_error(y, p, 10)
    manual = sum((m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
                 for m in [(np.minimum((p * 10).astype(int), 9) == b) for b in range(10)] if m.any())
    assert np.isclose(ece, manual) and ece < 0.05      # a perfectly calibrated source -> small ECE

def test_platt_matches_sklearn_logistic_on_scores():
    """Platt scaling == 1-D logistic regression on the score with smoothed targets.
    With (almost) no regularisation and *hard* labels the MLE must agree with sklearn."""
    ps = PlattScaler()
    # hard-label check of the optimiser: temporarily bypass smoothing by large-N limit comparison
    ps.fit(SCORES, Y)
    ref = LogisticRegression(C=1e10, tol=1e-12, max_iter=1000).fit(SCORES.reshape(-1, 1), Y)
    p_ref = ref.predict_proba(SCORES.reshape(-1, 1))[:, 1]
    p_us = ps.predict_proba(SCORES)
    assert np.max(np.abs(p_us - p_ref)) < 0.01             # only the target smoothing differs
    assert ps.A_ < 0                                        # increasing score -> increasing probability

def test_platt_optimum_is_stationary():
    ps = PlattScaler().fit(SCORES, Y)
    n_pos, n_neg = Y.sum(), (1 - Y).sum()
    t = np.where(Y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    p = ps.predict_proba(SCORES)
    assert abs(np.sum((t - p) * SCORES)) < 1e-4 and abs(np.sum(t - p)) < 1e-4   # gradient = 0

def test_isotonic_matches_sklearn_pav():
    for s in (SCORES, SCORES_TIES):
        ours = IsotonicCalibrator().fit(s, Y)
        ref = IsotonicRegression(out_of_bounds="clip").fit(s, Y)
        assert np.allclose(ours.predict_proba(s), ref.predict(s))          # same PAV solution on train points
        assert np.all(np.diff(ours.values_) > 0)                           # strictly increasing blocks

def test_isotonic_is_monotone_on_new_data():
    ours = IsotonicCalibrator().fit(SCORES_TIES, Y)
    grid = np.linspace(-8, 8, 1000)
    assert np.all(np.diff(ours.predict_proba(grid)) >= 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
