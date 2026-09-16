"""
Phase 7 (shared) - StreamScorer: score one live transaction end-to-end.

Loads the fitted FeatureBuilder + XGBoost booster + SHAP explainer once, and
scores incoming raw transaction dicts. To reproduce the causal entity
features (velocity, time-since-last, amount z-score) on a per-row basis, it
keeps a bounded history buffer PER entity (card1_addr1); a new transaction is
transformed together with its own entity's recent past, so its aggregations
match what training saw. Global/row-local features (freq encodings, time,
amount) need no history.

SHAP explanation is only computed for flagged (HIGH RISK) transactions to
keep streaming throughput high.

Used by both the Kafka consumer and the in-process simulator.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from features import FeatureBuilder, MODELS_DIR

HISTORY_PER_ENTITY = 50   # keep last N raw txns per card1_addr1 for aggregations
HIGH_RISK_SCORE = None    # filled from risk_scaler.json at load time


class StreamScorer:
    def __init__(self, with_explainer: bool = True, genai: bool = False) -> None:
        self.fb = FeatureBuilder.load()
        self.booster = xgb.Booster()
        self.booster.load_model(str(MODELS_DIR / "xgb_model.json"))
        self.feature_names = self.booster.feature_names

        scaler = json.loads((MODELS_DIR / "risk_scaler.json").read_text())
        self.high_cut = float(scaler["high_risk_prob_cutoff"])

        self.explainer = None
        if with_explainer:
            from explain import FraudExplainer
            self.explainer = FraudExplainer(MODELS_DIR / "xgb_model.json")

        self.genai = None
        if genai:
            from genai_explain import GenAIExplainer
            self.genai = GenAIExplainer()

        self._hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=HISTORY_PER_ENTITY))

    @staticmethod
    def _entity_key(row: dict) -> str:
        c = row.get("card1")
        a = row.get("addr1")
        c = "na" if pd.isna(c) else c
        a = "na" if pd.isna(a) else a
        return f"{c}_{a}"

    def _status(self, prob: float) -> str:
        if prob >= self.high_cut:
            return "HIGH RISK"
        if prob >= self.high_cut / 3:
            return "MEDIUM"
        return "LOW"

    def score(self, row: dict) -> dict:
        """row: raw transaction dict (one test_transaction record, merged w/ id).

        Returns a record ready for db.insert_scored plus a couple demo fields.
        """
        key = self._entity_key(row)
        hist = self._hist[key]

        frame = pd.DataFrame(list(hist) + [row])
        X, _ = self.fb.transform(frame)
        x_row = X.reindex(columns=self.feature_names).iloc[[-1]]

        margin = self.booster.predict(xgb.DMatrix(x_row), output_margin=True)[0]
        prob = float(1.0 / (1.0 + np.exp(-margin)))
        risk = int(round(prob * 100))
        status = self._status(prob)

        reason = ""
        if status == "HIGH RISK" and self.explainer is not None:
            exp = self.explainer.explain_row(x_row)
            reason = exp["reason"]
            if self.genai is not None and self.genai.available:
                reason = self.genai.enhance(
                    reason, exp["top_features"], float(row["TransactionAmt"]), risk
                )

        # update history AFTER scoring (row must not see itself)
        hist.append(row)

        return {
            "transaction_id": int(row["TransactionID"]),
            "transaction_dt": int(row["TransactionDT"]),
            "amount": float(row["TransactionAmt"]),
            "fraud_prob": round(prob, 4),
            "risk_score": risk,
            "status": status,
            "reason": reason,
            "scored_at": datetime.now(timezone.utc),
        }


if __name__ == "__main__":
    # smoke test: score a few rows from the cached train tail (has labels)
    from data_load import load_split

    sc = StreamScorer()
    df = load_split("train").tail(200)
    recs = df.to_dict(orient="records")
    hits = 0
    for r in recs:
        out = sc.score(r)
        if out["status"] == "HIGH RISK":
            hits += 1
            print(f"  #{out['transaction_id']}  score={out['risk_score']}  {out['status']}")
            print(f"     {out['reason']}")
    print(f"\nScored {len(recs)} rows, {hits} flagged HIGH RISK.")
