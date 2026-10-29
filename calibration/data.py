"""Dataset loaders"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer, load_digits, make_classification

Array = NDArray[np.float64]

def load_dataset(name: str) -> tuple[Array, Array]:
    """Return (X, y) with y in {0,1}."""
    if name == "breast_cancer":      # UCI Breast Cancer Wisconsin (Diagnostic): 569 x 30, malignant=1
        d = load_breast_cancer()
        return d.data.astype(float), (d.target == 0).astype(int)
    if name == "digits":             # UCI Optical Recognition of Handwritten Digits: 1797 x 64
        d = load_digits()            # binarised: digit >= 5 -> 1 (balanced, non-trivial)
        return d.data.astype(float), (d.target >= 5).astype(int)
    raise ValueError(f"unknown dataset {name!r}")

def make_synthetic(n: int = 30000, seed: int = 0) -> tuple[Array, Array]:
    """Noisy, mildly non-linear synthetic problem used ONLY for the calibration-set-size study."""
    X, y = make_classification(n_samples=n, n_features=20, n_informative=6, n_redundant=4,
                               n_clusters_per_class=3, class_sep=0.8, flip_y=0.05, random_state=seed)
    return X, y
