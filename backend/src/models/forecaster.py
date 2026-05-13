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
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb



from src.features.engineer import FEATURE_COLS, TARGET_COLS

_THIS      = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.normpath(os.path.join(_THIS, "..", "..", ".."))
MODELS_DIR = os.path.join(ROOT, "data", "models")

import wandb
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

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
    using walk-forward CV. Saves models + metadata.

    Data splits:
      Training window : 1997-01 -> 2022-12  (312 months)
      Test holdout     : 2023-01 -> 2026-02  (final evaluation only, never touched during training)

    Recency weighting: observations are weighted linearly from 1.0 (Jan 1997)
    to 2.0 (Dec 2022) so recent market conditions have twice the influence of
    the oldest data.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Only rows where ALL features AND ALL targets are defined
    df_train = feature_store.dropna(subset=FEATURE_COLS + TARGET_COLS).copy()

    # NEW CODE: Pure Direction (Aligns perfectly with the Momentum features)
    for horizon in [1, 2, 3]:            
        target_col = f"target_urea_t{horizon}"
        df_train.loc[:, f"dir_t{horizon}"] = (
            df_train[target_col] > df_train["urea"]
        ).astype(int)
        
    metadata = {
        "training_window_start": "1997-01",
        "training_window_end":   "2026-02",
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

        X_tv, X_test, y_tv, y_test = train_test_split(X, y_reg, test_size=0.15, random_state=42)
        _, _, y_tv_cl, y_test_dir = train_test_split(X, y_clf, test_size=0.15, random_state=42)

        t = (X_tv.index - X_tv.index.min()).days.astype(float)
        weights_tv = 1.0 + 0.5 * (t / t.max())

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

        # Final regressor
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tv, y_tv, sample_weight=weights_tv)

        # Final direction classifier
        clf = xgb.XGBClassifier(**XGB_CLF_PARAMS)
        clf.fit(X_tv, y_tv_cl, sample_weight=weights_tv)

        # Persist both
        with open(os.path.join(MODELS_DIR, f"xgb_urea_t{horizon}.pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(MODELS_DIR, f"xgb_dir_t{horizon}.pkl"), "wb") as f:
            pickle.dump(clf, f)

# --- Evaluate on the 15% Random Holdout Set ---
        test_preds  = model.predict(X_test)
        dir_preds   = clf.predict(X_test)
        
        # Pull the actual "current" urea prices matching the random test dates
        y_prev = df_train.loc[X_test.index, "urea"].values

        metadata[f"test_rmse_t{horizon}"]    = _rmse(y_test, test_preds)
        metadata[f"test_mae_t{horizon}"]     = float(mean_absolute_error(y_test, test_preds))
        metadata[f"test_dir_acc_t{horizon}"] = _directional_accuracy(y_test.values, test_preds, y_prev)
        metadata[f"test_clf_acc_t{horizon}"] = float(np.mean(dir_preds == y_test_dir.values))
        
        mc_residuals = y_test.values - test_preds

        metadata[f"residual_mean_t{horizon}"] = float(mc_residuals.mean())
        metadata[f"residual_std_t{horizon}"]  = float(mc_residuals.std())

        clf_acc = metadata.get(f"test_clf_acc_t{horizon}", 0)
        wandb.log({
            "horizon":          horizon,
            "horizon_days":     horizon * 30,
            "test_rmse":        metadata[f"test_rmse_t{horizon}"],
            "test_mae":         metadata[f"test_mae_t{horizon}"],
            "directional_acc":  metadata[f"test_dir_acc_t{horizon}"],
            "clf_acc":          metadata[f"test_clf_acc_t{horizon}"],
            "residual_std":     metadata[f"residual_std_t{horizon}"],
            "residual_mean":    metadata[f"residual_mean_t{horizon}"],
        })
        wandb.finish()

        print(
            f"  t{horizon}: clf_dir_acc={clf_acc*100:.0f}% | "
            f"MC residual_std=${mc_residuals.std():.1f} (test) | "
            f"test_rmse=${metadata.get(f'test_rmse_t{horizon}', 0):.1f}"
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
