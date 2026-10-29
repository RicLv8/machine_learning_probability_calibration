# Calibration of Probabilities — Platt Scaling & Isotonic Regression

Author: Riccardo Gaglio (84235A)

Assignment 2, based on *Predicting Good Probabilities with Supervised Learning*
(Niculescu-Mizil & Caruana, ICML 2005). Final report: [`report/main.pdf`](report/main.pdf).

Logistic Regression and Random Forest are trained on two UCI datasets (Breast Cancer Wisconsin, Digits binarised
as `>=5`), calibrated with **Platt scaling** and **isotonic regression (PAV)**, and compared with
**accuracy, log-loss, Brier score** (+ ECE) and **reliability diagrams**.

## What is implemented from scratch
| From scratch (NumPy only) | Off-the-shelf (allowed by the assignment) |
|---|---|
| Platt scaling (Newton's method, smoothed targets) — `calibration/calibrators.py` | `LogisticRegression`, `RandomForestClassifier`, `StandardScaler` |
| Isotonic regression / PAV — `calibration/calibrators.py` | `train_test_split`, dataset loading |
| Accuracy, log-loss, Brier, ECE, reliability bins — `calibration/metrics.py` | Matplotlib (plotting only) |

`tests/test_against_sklearn.py` checks the from-scratch code against scikit-learn reference implementations.

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
python -m pytest -q                  # 6 tests
python run_experiments.py            # ~1-2 min, regenerates results/figures and results/tables
cd report && latexmk -pdf main.tex   # rebuilds the report from the generated tables/figures
```
All randomness is seeded (splits use seeds 0..29). Datasets ship with scikit-learn: no download required.
Results were obtained with Python 3.14.5, NumPy 2.5.3, scikit-learn 1.9.1 and Matplotlib 3.11.2; Random Forest results may differ slightly with other library versions. The code logs the Python, NumPy and scikit-learn versions at the start of each run.


## Protocol (no data leakage)
Stratified **50% train / 25% calibration / 25% test**, repeated over 30 random splits.
Classifier (and its scaler) fitted on train only → calibrators fitted on calibration scores only →
all metrics/plots computed on test only. No hyper-parameter tuning.

## Layout
```
calibration/        calibrators.py (Platt, PAV) · metrics.py · data.py
run_experiments.py  full pipeline -> results/
tests/              unit tests vs scikit-learn
results/figures/    reliability diagrams (pooled + single split), calibration-set-size study
results/tables/     LaTeX/CSV/JSON tables used by the report
report/             main.tex, main.pdf
```
