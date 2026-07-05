"""
XGBoost fertilizer price forecasting model.
Trains three separate models — one per horizon (t1=30d, t2=60d, t3=90d).
Uses walk-forward (time-series) cross-validation to prevent data leakage.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import xgboost as xgb



from src.features.engineer import FEATURE_COLS, TARGET_COLS

_THIS      = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.normpath(os.path.join(_THIS, "..", "..", ".."))
MODELS_DIR = os.path.join(ROOT, "data", "models")

import wandb
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

# Chronological evaluation boundaries.
# HOLDOUT_START: first feature-row month of the held-out test window (~15% of
# trainable rows by time). Everything from here on is never seen in training.
# SHOCK_TARGET_START: test observations whose TARGET month falls on/after this
# date are the 2026 supply-shock (out-of-distribution) stretch and are reported
# separately — headline metrics come from the pre-shock window only.
HOLDOUT_START      = "2022-01-01"
SHOCK_TARGET_START = "2026-01-01"

XGB_PARAMS = {
    "n_estimators":     100,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "objective":        "reg:squarederror", # <--- Must be squarederror for prices
    "random_state":     42,
    "n_jobs":           -1,
}

def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def _directional_accuracy(y_true, y_pred, y_prev):
    """Fraction of periods where the predicted direction matches the actual direction."""
    actual_dir = np.sign(np.array(y_true) - np.array(y_prev))
    pred_dir   = np.sign(np.array(y_pred) - np.array(y_prev))
    mask = actual_dir != 0
    if mask.sum() == 0:
        return 0.5
    return float(np.mean(actual_dir[mask] == pred_dir[mask]))


def train(feature_store: pd.DataFrame) -> dict:
    """
    Trains 3 XGBoost regressors (price level) + 3 XGBoost classifiers (direction)
    and saves models + metadata.

    Evaluation protocol (chronological, no shuffling):
      Training pool : first trainable month -> HOLDOUT_START - 1, additionally
                      purged per horizon so no training row's forward target
                      (month + horizon) lands inside the holdout window.
      Test holdout  : feature rows from HOLDOUT_START onward — strictly after
                      the training pool, never touched during training or CV.
      Within the holdout, observations are bucketed by TARGET month:
        pre-shock (target <= 2025-12) -> headline test_* metrics
        shock     (target >= SHOCK_TARGET_START) -> *_shock_* metrics, reported
                  separately as an out-of-distribution stretch (small n).
      *_full_* metrics cover the whole holdout for reference.

    Walk-forward CV (TimeSeriesSplit, 5 folds) runs inside the training pool;
    its residual stats are stored as cv_residual_* for reference. The
    residual_mean/std_t{h} keys that feed the Monte Carlo simulation come from
    the PRE-SHOCK holdout residuals of the evaluation model.

    Deployed artifacts: after evaluation, models are refit on the FULL
    trainable history (training pool + holdout) with the same recency
    weighting, and those refit models are what gets pickled. All metadata
    metrics refer to the pre-refit chronological-holdout evaluation.

    Recency weighting: training observations are weighted linearly from 1.0
    (oldest) to 1.5 (newest month in the training pool).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Only rows where ALL features AND ALL targets are defined
    df_train = feature_store.dropna(subset=FEATURE_COLS + TARGET_COLS).copy()

    # Direction labels: did urea rise between now and the target month?
    for horizon in [1, 2, 3]:
        target_col = f"target_urea_t{horizon}"
        df_train.loc[:, f"dir_t{horizon}"] = (
            df_train[target_col] > df_train["urea"]
        ).astype(int)

    holdout_start = pd.Timestamp(HOLDOUT_START)
    shock_start   = pd.Timestamp(SHOCK_TARGET_START)

    metadata = {
        "training_window_start": df_train.index.min().strftime("%Y-%m"),
        # Nominal boundary; per-horizon purge trims up to `horizon` further months.
        "training_window_end":   (holdout_start - pd.DateOffset(months=1)).strftime("%Y-%m"),
        "holdout_start":         holdout_start.strftime("%Y-%m"),
        "holdout_end":           df_train.index.max().strftime("%Y-%m"),
        "shock_target_start":    shock_start.strftime("%Y-%m"),
        "feature_cols":          FEATURE_COLS,
    }

    XGB_CLF_PARAMS = {
        "n_estimators":     100,     # Less trees
        "max_depth":        5,       # STUMPS: Forces the model to only look at massive momentum trends
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "objective":        "binary:logistic",
        "random_state":     42,
        "n_jobs":           -1,
    }
    for horizon, target_col in enumerate(TARGET_COLS, start=1):
        wandb.init(
            project="foregast",
            name=f"t{horizon}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "horizon_days":     horizon * 30,
                "xgb_params":       XGB_PARAMS,
                "n_cv_splits":      5,
                "test_size":        0.15,
                "training_start":   "1997-01",
            },
            reinit="finish_previous",
        )

        y_reg = df_train[target_col]
        y_clf = df_train[f"dir_t{horizon}"]
        X     = df_train[FEATURE_COLS]

        # Chronological split. Training rows are additionally purged so their
        # forward-looking target (month + horizon) lands strictly before the
        # holdout window — otherwise boundary rows train on test-period prices.
        target_month = X.index + pd.DateOffset(months=horizon)
        train_mask   = np.asarray(target_month < holdout_start)
        test_mask    = np.asarray(X.index >= holdout_start)

        X_tv,   y_tv,   y_tv_cl    = X[train_mask], y_reg[train_mask], y_clf[train_mask]
        X_test, y_test, y_test_dir = X[test_mask],  y_reg[test_mask],  y_clf[test_mask]

        t = (X_tv.index - X_tv.index.min()).days.astype(float)
        weights_tv = np.asarray(1.0 + 0.5 * (t / t.max()))

        tscv = TimeSeriesSplit(n_splits=5, gap=1)
        cv_residuals = []
        for tr_idx, vl_idx in tscv.split(X_tv):
            X_tr, X_vl = X_tv.iloc[tr_idx], X_tv.iloc[vl_idx]
            y_tr, y_vl = y_tv.iloc[tr_idx], y_tv.iloc[vl_idx]
            w_tr       = weights_tv[tr_idx]

            m = xgb.XGBRegressor(**XGB_PARAMS)
            m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            preds = m.predict(X_vl)
            cv_residuals.extend((y_vl.values - preds).tolist())

        # Evaluation regressor / classifier — fit on the training pool only,
        # so holdout metrics below are genuinely out-of-sample.
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tv, y_tv, sample_weight=weights_tv)

        clf = xgb.XGBClassifier(**XGB_CLF_PARAMS)
        clf.fit(X_tv, y_tv_cl, sample_weight=weights_tv)

# --- Evaluate on the chronological holdout ---
        test_preds  = model.predict(X_test)
        dir_preds   = clf.predict(X_test)
        y_prev      = df_train.loc[X_test.index, "urea"].values

        # Bucket test observations by TARGET month: a late-2025 feature row
        # predicting into 2026 belongs to the shock stretch.
        test_target_month = X_test.index + pd.DateOffset(months=horizon)
        shock = np.asarray(test_target_month >= shock_start)
        pre   = ~shock

        def _window_metrics(mask: np.ndarray) -> dict:
            return {
                "rmse":    _rmse(y_test.values[mask], test_preds[mask]),
                "mae":     float(mean_absolute_error(y_test.values[mask], test_preds[mask])),
                "dir_acc": _directional_accuracy(y_test.values[mask], test_preds[mask], y_prev[mask]),
                "clf_acc": float(np.mean(dir_preds[mask] == y_test_dir.values[mask])),
            }

        m_pre  = _window_metrics(pre)
        m_full = _window_metrics(np.ones(len(X_test), dtype=bool))
        m_shock = _window_metrics(shock) if shock.any() else None

        # Headline (schema-stable) keys = pre-shock window.
        metadata[f"test_rmse_t{horizon}"]    = m_pre["rmse"]
        metadata[f"test_mae_t{horizon}"]     = m_pre["mae"]
        metadata[f"test_dir_acc_t{horizon}"] = m_pre["dir_acc"]
        metadata[f"test_clf_acc_t{horizon}"] = m_pre["clf_acc"]
        metadata[f"n_test_preshock_t{horizon}"] = int(pre.sum())
        metadata[f"n_test_shock_t{horizon}"]    = int(shock.sum())

        for name, m in (("full", m_full), ("shock", m_shock)):
            if m is None:
                continue
            metadata[f"test_rmse_{name}_t{horizon}"]    = m["rmse"]
            metadata[f"test_mae_{name}_t{horizon}"]     = m["mae"]
            metadata[f"test_dir_acc_{name}_t{horizon}"] = m["dir_acc"]
            metadata[f"test_clf_acc_{name}_t{horizon}"] = m["clf_acc"]

        # Monte Carlo residuals: pre-shock holdout residuals of the final model —
        # recent, unseen, and not distorted by the out-of-distribution shock months.
        mc_residuals = y_test.values[pre] - test_preds[pre]
        metadata[f"residual_mean_t{horizon}"] = float(mc_residuals.mean())
        metadata[f"residual_std_t{horizon}"]  = float(mc_residuals.std())

        # Walk-forward CV residuals, kept for reference/comparison.
        cv_res = np.asarray(cv_residuals)
        metadata[f"cv_residual_mean_t{horizon}"] = float(cv_res.mean())
        metadata[f"cv_residual_std_t{horizon}"]  = float(cv_res.std())

        wandb.log({
            "horizon":                horizon,
            "horizon_days":           horizon * 30,
            "test_rmse":              m_pre["rmse"],
            "test_mae":               m_pre["mae"],
            "directional_acc":        m_pre["dir_acc"],
            "clf_acc":                m_pre["clf_acc"],
            "test_rmse_full":         m_full["rmse"],
            "test_rmse_shock":        m_shock["rmse"] if m_shock else None,
            "n_test_preshock":        int(pre.sum()),
            "n_test_shock":           int(shock.sum()),
            "residual_std":           metadata[f"residual_std_t{horizon}"],
            "residual_mean":          metadata[f"residual_mean_t{horizon}"],
            "cv_residual_std":        metadata[f"cv_residual_std_t{horizon}"],
        })
        wandb.finish()

        # Deployed artifacts: refit on ALL trainable rows (train + holdout) so
        # the served forecast uses the full history. The metrics stored above
        # were measured on the chronological holdout BEFORE this refit and are
        # the honest estimate of this configuration's out-of-sample error.
        t_all = (X.index - X.index.min()).days.astype(float)
        weights_all = np.asarray(1.0 + 0.5 * (t_all / t_all.max()))

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X, y_reg, sample_weight=weights_all)
        clf = xgb.XGBClassifier(**XGB_CLF_PARAMS)
        clf.fit(X, y_clf, sample_weight=weights_all)

        with open(os.path.join(MODELS_DIR, f"xgb_urea_t{horizon}.pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(MODELS_DIR, f"xgb_dir_t{horizon}.pkl"), "wb") as f:
            pickle.dump(clf, f)

        print(
            f"  t{horizon}: pre-shock rmse=${m_pre['rmse']:.1f} "
            f"dir_acc={m_pre['dir_acc']*100:.0f}% clf_acc={m_pre['clf_acc']*100:.0f}% "
            f"(n={int(pre.sum())}) | full rmse=${m_full['rmse']:.1f} | "
            f"shock rmse={'$' + format(m_shock['rmse'], '.1f') if m_shock else 'n/a'} "
            f"(n={int(shock.sum())})"
        )

    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Models saved -> {MODELS_DIR}")
    return metadata


def load_models() -> tuple[dict, dict]:
    """
    Returns (models_dict, metadata).
    models_dict keys: 't1','t2','t3' (regressors) and 'dir_t1','dir_t2','dir_t3' (classifiers).
    """
    models = {}
    for h in [1, 2, 3]:
        with open(os.path.join(MODELS_DIR, f"xgb_urea_t{h}.pkl"), "rb") as f:
            models[f"t{h}"] = pickle.load(f)
        clf_path = os.path.join(MODELS_DIR, f"xgb_dir_t{h}.pkl")
        if os.path.exists(clf_path):
            with open(clf_path, "rb") as f:
                models[f"dir_t{h}"] = pickle.load(f)

    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    return models, metadata


def predict(models: dict, feature_row: pd.DataFrame) -> dict:
    """
    Given a single-row DataFrame with FEATURE_COLS, returns point forecasts
    and direction probabilities from the classifier.
    """
    X = feature_row[FEATURE_COLS]
    result = {
        "t1": float(models["t1"].predict(X)[0]),
        "t2": float(models["t2"].predict(X)[0]),
        "t3": float(models["t3"].predict(X)[0]),
    }
    # Add classifier probability of price going UP (class 1)
    for h in [1, 2, 3]:
        key = f"dir_t{h}"
        if key in models:
            prob_up = float(models[key].predict_proba(X)[0][1])
            result[f"prob_up_t{h}"] = prob_up
    return result
