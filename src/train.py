"""
Phase 4 - XGBoost training with time-respecting validation.

Why time-ordered (not random) CV: TransactionDT is sorted over 182 days and
the test/stream data is strictly later. A random split would leak future
information into the fold. So we train on the earlier portion and validate
on the latest ~20% of days, mirroring how the live stream will look.

Imbalance (3.5% fraud, ~28:1) is handled with scale_pos_weight and we tune
against PR-AUC (average precision), the right metric for rare positives.

Outputs to models/:
  * xgb_model.json      - the booster
  * risk_scaler.json    - maps predicted prob -> 0..100 risk score + HIGH cutoff
  * metrics.json        - validation PR-AUC / ROC-AUC / PR@k
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import xgboost as xgb

from data_load import load_split
from features import FeatureBuilder, MODELS_DIR

VALID_FRACTION = 0.20  # last 20% of the time span is the holdout
HIGH_RISK_PCTL = 99.0  # score above this train-percentile of prob -> HIGH RISK


def time_holdout_split(df: pd.DataFrame, frac: float):
    """Split by TransactionDT: earlier -> train, latest `frac` -> valid."""
    dt = df["TransactionDT"]
    cutoff = dt.quantile(1 - frac)
    train_mask = dt <= cutoff
    return train_mask.values


def main() -> None:
    df = load_split("train")

    # fit feature builder on FULL train encoders is standard for this task;
    # causal aggregations are inherently past-only so no per-row leak.
    fb = FeatureBuilder()
    X, y = fb.fit_transform(df)
    fb.save()

    # align y and the time split to X's (DT-sorted) index
    dt_sorted = df.loc[X.index, "TransactionDT"]
    cutoff = dt_sorted.quantile(1 - VALID_FRACTION)
    tr = (dt_sorted <= cutoff).values
    va = ~tr

    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    print(f"train rows: {len(X_tr):,} (fraud {y_tr.mean():.3%}) | "
          f"valid rows: {len(X_va):,} (fraud {y_va.mean():.3%})")

    pos = int(y_tr.sum())
    neg = int((~y_tr.astype(bool)).sum())
    spw = neg / max(pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=3,
        reg_lambda=1.0,
        reg_alpha=0.1,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        tree_method="hist",
        early_stopping_rounds=100,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=100)

    p_va = model.predict_proba(X_va)[:, 1]
    pr_auc = average_precision_score(y_va, p_va)
    roc_auc = roc_auc_score(y_va, p_va)

    # precision@k: of the top 1% riskiest, how many are truly fraud?
    k = max(1, int(0.01 * len(p_va)))
    top_idx = np.argsort(p_va)[-k:]
    precision_at_1pct = y_va.values[top_idx].mean()

    print("\n=== validation (time holdout) ===")
    print(f"PR-AUC (avg precision): {pr_auc:.4f}")
    print(f"ROC-AUC:                {roc_auc:.4f}")
    print(f"Precision@top1%:        {precision_at_1pct:.4f}")
    print(f"best_iteration:         {model.best_iteration}")

    # risk-score scaler: prob -> 0..100 and a HIGH-RISK probability cutoff
    p_tr = model.predict_proba(X_tr)[:, 1]
    high_cut = float(np.percentile(p_tr, HIGH_RISK_PCTL))
    scaler = {
        "mode": "linear_prob_x100",
        "high_risk_prob_cutoff": high_cut,
        "high_risk_pctl": HIGH_RISK_PCTL,
    }

    model.get_booster().save_model(str(MODELS_DIR / "xgb_model.json"))
    (MODELS_DIR / "risk_scaler.json").write_text(json.dumps(scaler, indent=2))
    (MODELS_DIR / "metrics.json").write_text(json.dumps({
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "precision_at_top1pct": round(float(precision_at_1pct), 4),
        "best_iteration": int(model.best_iteration),
        "scale_pos_weight": round(spw, 2),
        "n_features": X.shape[1],
        "valid_rows": int(len(X_va)),
    }, indent=2))
    print(f"\nSaved model + scaler + metrics -> {MODELS_DIR}")


if __name__ == "__main__":
    main()
