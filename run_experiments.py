"""This file contains every output and figure of the project.

    python run_experiments.py
    python run_experiments.py --repeats 5

Protocol (no leakage): stratified train / calibration / test split.
  * classifier fitted on TRAIN only (the scaler of the LR pipeline is part of the fit)
  * Platt / isotonic fitted on CALIBRATION scores only
  * all reported metrics computed on TEST only
Repeated over R random splits to report mean +- std.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from calibration import (IsotonicCalibrator, PlattScaler, accuracy, brier_score,
                         expected_calibration_error, log_loss, reliability_curve)
from calibration.data import load_dataset, make_synthetic

log = logging.getLogger("experiments")
OUT = Path("results")
DATASETS = {"breast_cancer": "Breast Cancer Wisconsin", "digits": "Digits (binarised: $\\geq 5$)"}
MODELS = ("Logistic Regression", "Random Forest")
VARIANTS = ("Uncalibrated", "Platt", "Isotonic")
COLORS = {"Uncalibrated": "#d62728", "Platt": "#1f77b4", "Isotonic": "#2ca02c"}
METRICS = {"Accuracy": accuracy, "Log-loss": log_loss, "Brier": brier_score, "ECE": expected_calibration_error}


##### models ######
def make_model(name: str, seed: int):
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)


def get_scores(name: str, model, X: np.ndarray) -> np.ndarray:
    """Score f(x) fed to the calibrators: LR -> decision function (logit); RF -> P(y=1) vote fraction."""
    return model.decision_function(X) if name == "Logistic Regression" else model.predict_proba(X)[:, 1]


def split3(X, y, seed: int, p_train=0.5, p_cal=0.25):
    """Stratified train/calibration/test split (50/25/25)."""
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=p_train, stratify=y, random_state=seed)
    X_ca, X_te, y_ca, y_te = train_test_split(X_rest, y_rest, train_size=p_cal / (1 - p_train),
                                              stratify=y_rest, random_state=seed)
    return X_tr, y_tr, X_ca, y_ca, X_te, y_te


def calibrated_probs(name, model, X_ca, y_ca, X_te):
    """Return {variant: test probabilities} + fitted calibrators."""
    s_ca, s_te = get_scores(name, model, X_ca), get_scores(name, model, X_te)
    platt, iso = PlattScaler().fit(s_ca, y_ca), IsotonicCalibrator().fit(s_ca, y_ca)
    unc = 1 / (1 + np.exp(-s_te)) if name == "Logistic Regression" else s_te   # LR: sigmoid(decision fn)
    return {"Uncalibrated": unc, "Platt": platt.predict_proba(s_te), "Isotonic": iso.predict_proba(s_te)}, platt, iso


##### main experiment ######
def run_main(repeats: int):
    rows, pooled, extra = [], {}, {}
    for ds in DATASETS:
        X, y = load_dataset(ds)
        log.info("Dataset %-14s n=%d d=%d positives=%.1f%%", ds, *X.shape, 100 * y.mean())
        for m in MODELS:
            pooled[(ds, m)] = {v: ([], []) for v in VARIANTS}
        for seed in range(repeats):
            X_tr, y_tr, X_ca, y_ca, X_te, y_te = split3(X, y, seed)
            for m in MODELS:
                model = make_model(m, seed).fit(X_tr, y_tr)
                probs, platt, iso = calibrated_probs(m, model, X_ca, y_ca, X_te)
                for v in VARIANTS:
                    rows.append(dict(dataset=ds, model=m, variant=v, seed=seed,
                                     **{k: f(y_te, probs[v]) for k, f in METRICS.items()}))
                    pooled[(ds, m)][v][0].append(y_te); pooled[(ds, m)][v][1].append(probs[v])
                if seed == 0:
                    extra[f"{ds}|{m}"] = dict(n_train=len(y_tr), n_cal=len(y_ca), n_test=len(y_te),
                                              platt_A=platt.A_, platt_B=platt.B_, iso_blocks=len(iso.values_),
                                              iso_frac_exact01=float(np.mean(np.isin(iso.predict_proba(get_scores(m, model, X_te)), [0., 1.]))))
                    # single-split figure (seed 0)
                    pooled[(ds, m)]["seed0"] = {v: (y_te, probs[v]) for v in VARIANTS}
    return rows, pooled, extra


def summarise(rows):
    """mean/std per (dataset, model, variant) and paired 'Brier improved vs uncalibrated' counts."""
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["model"], r["variant"])].append(r)
    out = {}
    for (ds, m, v), rs in g.items():
        d = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs], ddof=1))) for k in METRICS}
        if v != "Uncalibrated":
            base = {r["seed"]: r for r in g[(ds, m, "Uncalibrated")]}
            d["wins_brier"] = int(sum(r["Brier"] < base[r["seed"]]["Brier"] for r in rs))
            d["wins_logloss"] = int(sum(r["Log-loss"] < base[r["seed"]]["Log-loss"] for r in rs))
            d["n"] = len(rs)
        out[(ds, m, v)] = d
    return out


def write_tables(summary, repeats):
    """LaTeX (booktabs) + CSV tables that the report \\input's."""
    lines = []
    for ds, title in DATASETS.items():
        lines += [f"\\begin{{table}}[H]\\centering\\small",
                  f"\\caption{{Test-set metrics on {title} (mean $\\pm$ std over {repeats} random splits). "
                  f"Best value per model in bold. $\\Delta$: number of splits in which the calibrator improved the metric over the uncalibrated model.}}",
                  f"\\label{{tab:{ds}}}",
                  "\\resizebox{\\textwidth}{!}{\\begin{tabular}{llccccc}\\toprule",
                  "Model & Variant & Accuracy $\\uparrow$ & Log-loss $\\downarrow$ & Brier $\\downarrow$ & ECE $\\downarrow$ & $\\Delta$ Brier / LL \\\\\\midrule"]
        for m in MODELS:
            best = {k: min((summary[(ds, m, v)][k][0] for v in VARIANTS), key=(lambda x: -x) if k == "Accuracy" else (lambda x: x))
                    for k in METRICS}
            for v in VARIANTS:
                d = summary[(ds, m, v)]
                cells = []
                for k, fmt in (("Accuracy", "{:.4f}"), ("Log-loss", "{:.4f}"), ("Brier", "{:.4f}"), ("ECE", "{:.4f}")):
                    s = f"{fmt.format(d[k][0])}\\,$\\pm$\\,{fmt.format(d[k][1])}"
                    cells.append(f"\\textbf{{{s}}}" if abs(d[k][0] - best[k]) < 1e-12 else s)
                w = f"{d['wins_brier']}/{d['n']} \\; {d['wins_logloss']}/{d['n']}" if "wins_brier" in d else "--"
                lines.append(f"{m if v == 'Uncalibrated' else ''} & {v} & " + " & ".join(cells) + f" & {w} \\\\")
            if m != MODELS[-1]:
                lines.append("\\midrule")
        lines += ["\\bottomrule\\end{tabular}}\\end{table}", ""]
    (OUT / "tables" / "main_tables.tex").write_text("\n".join(lines))
    with open(OUT / "tables" / "summary.csv", "w") as fh:
        fh.write("dataset,model,variant,acc_mean,acc_std,logloss_mean,logloss_std,brier_mean,brier_std,ece_mean,ece_std\n")
        for (ds, m, v), d in summary.items():
            fh.write(f"{ds},{m},{v}," + ",".join(f"{d[k][0]:.6f},{d[k][1]:.6f}" for k in METRICS) + "\n")


############ Figures ###########
def plot_reliability(ds: str, pooled, key: str, fname: str, suptitle: str):
    """2 columns (models) x 2 rows (reliability curve / histogram of predictions)."""
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), gridspec_kw=dict(height_ratios=[3, 1.4]), sharex=True)
    for j, m in enumerate(MODELS):
        ax, axh = axes[0, j], axes[1, j]
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="Perfect calibration")
        for v in VARIANTS:
            ys, ps = pooled[(ds, m)][key][v] if key != "pooled" else pooled[(ds, m)][v]
            y_all = np.concatenate(ys) if key == "pooled" else ys
            p_all = np.concatenate(ps) if key == "pooled" else ps
            mp, fp, cnt = reliability_curve(y_all, p_all, 10)
            ax.plot(mp, fp, "o-", color=COLORS[v], ms=5, lw=1.6,
                    label=f"{v} (ECE={expected_calibration_error(y_all, p_all):.3f})")
            axh.hist(p_all, bins=20, range=(0, 1), histtype="step", color=COLORS[v], lw=1.5)
        ax.set_title(m); ax.set_ylim(-0.03, 1.03); ax.set_xlim(-0.01, 1.01); ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc="upper left")
        axh.set_xlabel("Mean predicted probability"); axh.set_yscale("log"); axh.grid(alpha=.3)
        if j == 0:
            ax.set_ylabel("Fraction of positives"); axh.set_ylabel("Count (log)")
    fig.suptitle(suptitle, fontsize=11); fig.tight_layout()
    fig.savefig(OUT / "figures" / fname, dpi=160); plt.close(fig)


####### calibration-set-size study ######
def run_learning_curve(repeats: int = 10, sizes=tuple(2 ** k for k in range(5, 14))):
    X, y = make_synthetic()
    res = {(m, v): {s: {"Brier": [], "Log-loss": []} for s in sizes} for m in MODELS for v in VARIANTS}
    for r in range(repeats):
        rng = np.random.default_rng(r); idx = rng.permutation(len(y))
        te, tr, pool = idx[:10000], idx[10000:14000], idx[14000:]      # test 10k, train 4k, calibration pool 16k
        for m in MODELS:
            model = make_model(m, r).fit(X[tr], y[tr])
            for s in sizes:
                ca = pool[:s]
                if len(np.unique(y[ca])) < 2:
                    continue
                probs, _, _ = calibrated_probs(m, model, X[ca], y[ca], X[te])
                for v in VARIANTS:
                    res[(m, v)][s]["Brier"].append(brier_score(y[te], probs[v]))
                    res[(m, v)][s]["Log-loss"].append(log_loss(y[te], probs[v]))
        log.info("learning curve repeat %d/%d done", r + 1, repeats)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True)
    for i, metric in enumerate(("Brier", "Log-loss")):
        for j, m in enumerate(MODELS):
            ax = axes[i, j]
            for v in VARIANTS:
                mu = np.array([np.mean(res[(m, v)][s][metric]) for s in sizes])
                sd = np.array([np.std(res[(m, v)][s][metric]) for s in sizes])
                if v == "Uncalibrated":
                    ax.axhline(mu.mean(), color=COLORS[v], ls="--", label=v)
                else:
                    ax.plot(sizes, mu, "o-", color=COLORS[v], ms=3, label=v)
                    ax.fill_between(sizes, mu - sd, mu + sd, color=COLORS[v], alpha=.15)
            ax.set_xscale("log", base=2); ax.set_xticks(list(sizes)); ax.set_xticklabels([str(v) for v in sizes], rotation=60, fontsize=7); ax.minorticks_off(); ax.grid(alpha=.3)
            ax.set_title(f"{m} -- {metric}"); ax.set_xlabel("Calibration set size"); ax.set_ylabel(metric)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "calibration_set_size.pdf"); plt.close(fig)
    summ = {f"{m}|{v}|{s}": {k: float(np.mean(res[(m, v)][s][k])) for k in ("Brier", "Log-loss")}
            for m in MODELS for v in VARIANTS for s in sizes}
    (OUT / "tables" / "calibration_set_size.json").write_text(json.dumps(summ, indent=1))
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for sub in ("figures", "tables"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9})
    log.info("python %s | numpy %s | sklearn %s", platform.python_version(), np.__version__, sklearn.__version__)

    rows, pooled, extra = run_main(args.repeats)
    summary = summarise(rows)
    write_tables(summary, args.repeats)
    for ds, title in DATASETS.items():
        clean = title.replace("$\\geq 5$", ">=5")
        plot_reliability(ds, pooled, "pooled", f"reliability_{ds}.pdf",
                         f"{clean}: test predictions pooled over {args.repeats} random splits")
        plot_reliability(ds, pooled, "seed0", f"reliability_{ds}_single_split.pdf",
                         f"{clean}: single split (seed 0)")
    (OUT / "tables" / "extra_stats.json").write_text(json.dumps(extra, indent=1))
    run_learning_curve(repeats=10)
    log.info("Done. Outputs in %s/", OUT)


if __name__ == "__main__":
    main()
