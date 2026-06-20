"""
train_models.py — Run once to train XGBoost models and save artifacts.

Source for the four monthly series (ng_spot, urea, dap, storage_mmcf) is
env-configurable:
    FEATURE_SOURCE=mart      (default) — read dev_marts.mart_commodity_panel_monthly
                                          via DATABASE_URL.
    FEATURE_SOURCE=pipeline  — call run_ingestion() (legacy pandas path).

If FEATURE_SOURCE=mart but DATABASE_URL is unset, the script falls back to
'pipeline' with a logged warning so a deployment without Postgres still works.

Usage (from project root):
    python backend/train_models.py
    python backend/train_models.py --dry-run    # build features, skip save+train
"""

import argparse
import os
import sys

# Make src importable when running from project root or backend/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load .env so FEATURE_SOURCE / DATABASE_URL set there are visible here
# (engineer.py also calls load_dotenv, but _resolve_source runs before that).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_THIS_DIR, "..", ".env"))

from src.features.engineer import build_features, save_feature_store  # noqa: E402
# src.models.forecaster is imported lazily inside main() so that --dry-run
# works without xgboost installed (useful for source-resolution smoke checks
# on minimal envs).


def _resolve_source() -> str:
    """
    Resolve the feature source from env. Returns 'mart' or 'pipeline'.
    Falls back from 'mart' -> 'pipeline' with a warning when DATABASE_URL is unset.
    """
    requested = (os.getenv("FEATURE_SOURCE", "mart").strip().lower() or "mart")
    if requested not in ("mart", "pipeline"):
        raise ValueError(
            f"FEATURE_SOURCE must be 'mart' or 'pipeline', got {requested!r}"
        )
    if requested == "mart" and not os.getenv("DATABASE_URL", "").strip():
        print(
            "     WARNING: FEATURE_SOURCE=mart but DATABASE_URL is unset — "
            "falling back to source='pipeline'."
        )
        return "pipeline"
    return requested


def main(dry_run: bool = False) -> int:
    print("=== Gas Forecast Model Training ===\n")

    source = _resolve_source()
    print(f"1/3 - Building feature store (source={source})...")
    fs = build_features(source=source, start="1997-01")
    print(f"     NG spot:  {fs['ng_spot'].dropna().shape[0]} months")
    print(f"     Urea:     {fs['urea'].dropna().shape[0]} months")
    print(f"     DAP:      {fs['dap'].dropna().shape[0]} months")
    print(f"     Storage:  {fs['storage_mmcf'].dropna().shape[0]} months")
    print(f"     Feature store: {fs.shape[0]} rows x {fs.shape[1]} cols")

    if dry_run:
        print("\n--dry-run set - skipping save and training. Exiting.")
        return 0

    print("\n2/3 - Saving feature store...")
    path = save_feature_store(fs)
    print(f"     -> {path}")

    print("\n3/3 - Training XGBoost models (t1, t2, t3)...")
    from src.models.forecaster import train  # lazy: only needed for real runs
    metadata = train(fs)

    print("\n=== Training complete ===")
    print(f"  Residual std  t1: ${metadata['residual_std_t1']:.1f}/mt")
    print(f"  Residual std  t2: ${metadata['residual_std_t2']:.1f}/mt")
    print(f"  Residual std  t3: ${metadata['residual_std_t3']:.1f}/mt")
    if "test_clf_acc_t3" in metadata:
        print(f"  Test RMSE       t3 (90-Day): ${metadata.get('test_rmse_t3', 0):.1f}/mt")
        print(f"  Classifier Acc  t3 (90-Day): {metadata.get('test_clf_acc_t3', 0)*100:.0f}%")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost models on the feature store.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the feature store but skip save+train (for source smoke checks).",
    )
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run))
