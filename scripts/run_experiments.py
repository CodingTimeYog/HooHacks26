"""
Seed-sensitivity sweep for the holdout evaluation.

Training is fully deterministic for a fixed random_state (verified: repeated
runs produce byte-identical metadata), so averaging repeated same-seed runs is
meaningless. What IS worth knowing is how much the reported holdout metrics
move when the XGBoost random_state changes — i.e. whether the headline numbers
are a lucky seed. This script refits the evaluation models across N seeds using
the exact chronological split from forecaster.train() and reports the spread
of the pre-shock holdout metrics.

Usage (from project root):
    python scripts/run_experiments.py            # 20 seeds
    python scripts/run_experiments.py --seeds 50
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from src.features.engineer import FEATURE_COLS, TARGET_COLS, load_feature_store  # noqa: E402
from src.models.forecaster import (  # noqa: E402
    HOLDOUT_START,
    SHOCK_TARGET_START,
    XGB_PARAMS,
    _directional_accuracy,
    _rmse,
)

# Mirrors the classifier params in forecaster.train() (defined locally there).
XGB_CLF_PARAMS = {
    "n_estimators":     100,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "objective":        "binary:logistic",
    "random_state":     42,
    "n_jobs":           -1,
}


def sweep(n_seeds: int) -> None:
    df = load_feature_store().dropna(subset=FEATURE_COLS + TARGET_COLS).copy()
    for h in (1, 2, 3):
        df[f"dir_t{h}"] = (df[f"target_urea_t{h}"] > df["urea"]).astype(int)
    X_all = df[FEATURE_COLS]
    holdout_start = pd.Timestamp(HOLDOUT_START)
    shock_start   = pd.Timestamp(SHOCK_TARGET_START)

    results: dict[int, dict[str, list[float]]] = {
        h: {"rmse": [], "dir_acc": [], "clf_acc": []} for h in (1, 2, 3)
    }

    for seed in range(n_seeds):
        print(f"seed {seed + 1}/{n_seeds}...", end=" ", flush=True)
        for horizon, target_col in enumerate(TARGET_COLS, start=1):
            y_reg = df[target_col]
            y_clf = df[f"dir_t{horizon}"]

            target_month = X_all.index + pd.DateOffset(months=horizon)
            train_mask   = np.asarray(target_month < holdout_start)
            test_mask    = np.asarray(X_all.index >= holdout_start)

            X_tv, y_tv, y_tv_cl = X_all[train_mask], y_reg[train_mask], y_clf[train_mask]
            X_test              = X_all[test_mask]
            y_test, y_test_dir  = y_reg[test_mask], y_clf[test_mask]

            t = (X_tv.index - X_tv.index.min()).days.astype(float)
            w = np.asarray(1.0 + 0.5 * (t / t.max()))

            model = xgb.XGBRegressor(**{**XGB_PARAMS, "random_state": seed})
            model.fit(X_tv, y_tv, sample_weight=w)
            clf = xgb.XGBClassifier(**{**XGB_CLF_PARAMS, "random_state": seed})
            clf.fit(X_tv, y_tv_cl, sample_weight=w)

            test_preds = model.predict(X_test)
            dir_preds  = clf.predict(X_test)
            y_prev     = df.loc[X_test.index, "urea"].values

            pre = np.asarray(X_test.index + pd.DateOffset(months=horizon) < shock_start)
            results[horizon]["rmse"].append(_rmse(y_test.values[pre], test_preds[pre]))
            results[horizon]["dir_acc"].append(
                _directional_accuracy(y_test.values[pre], test_preds[pre], y_prev[pre])
            )
            results[horizon]["clf_acc"].append(
                float(np.mean(dir_preds[pre] == y_test_dir.values[pre]))
            )
        print("done")

    print(f"\nPre-shock holdout metric spread across {n_seeds} seeds")
    print("=" * 72)
    for h in (1, 2, 3):
        r = results[h]
        rmse, dacc, cacc = map(np.asarray, (r["rmse"], r["dir_acc"], r["clf_acc"]))
        print(
            f"t{h} ({h * 30}-day) | "
            f"RMSE ${rmse.mean():.1f} +/- {rmse.std():.1f} "
            f"[{rmse.min():.1f}, {rmse.max():.1f}] | "
            f"dir_acc {dacc.mean() * 100:.1f}% +/- {dacc.std() * 100:.1f} | "
            f"clf_acc {cacc.mean() * 100:.1f}% +/- {cacc.std() * 100:.1f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed-sensitivity sweep of holdout metrics.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds (default 20).")
    args = parser.parse_args()
    sweep(args.seeds)
