"""
Phase 5 - SHAP explainability + natural-language reason.

TreeExplainer on the saved XGBoost booster. For a single scored transaction
we take the top-N SHAP contributions pushing the prediction toward fraud and
render a human sentence like the demo target:

    "Reason: Transaction amount and behavioural pattern significantly differ
     from normal transaction patterns."

The class is built once (explainer construction is the expensive part) and
reused per transaction by the Kafka consumer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from features import MODELS_DIR

# map raw feature names -> readable phrases for the reason sentence
FEATURE_PHRASES = {
    "TransactionAmt": "transaction amount",
    "amt_log": "transaction amount",
    "amt_cents": "unusual amount precision",
    "amt_is_round": "round-number amount",
    "ent_amt_z": "amount vs this card's usual spend",
    "ent_txn_count_sofar": "recent transaction velocity on this card",
    "ent_secs_since_last": "time since this card's last transaction",
    "dt_hour": "time of day",
    "dt_dayofweek": "day of week",
    "card1_freq": "rarity of the card fingerprint",
    "card1_addr1_freq": "rarity of this card+address combination",
    "addr1_freq": "rarity of the billing region",
    "P_emaildomain_freq": "rarity of the purchaser email domain",
    "email_same_pr": "purchaser/recipient email mismatch",
    "grp_id_missing_frac": "missing device/identity information",
    "grp_V_missing_frac": "missing Vesta risk signals",
    "DeviceInfo_freq": "rarity of the device",
}


def _phrase(feat: str) -> str:
    if feat in FEATURE_PHRASES:
        return FEATURE_PHRASES[feat]
    if feat.startswith("V"):
        return "internal risk signal"
    if feat.startswith("C"):
        return "transaction counting pattern"
    if feat.startswith("D"):
        return "account age / timing pattern"
    if feat.startswith("M"):
        return "identity match flags"
    if feat.startswith("id_"):
        return "device/identity attribute"
    return feat.replace("_", " ")


class FraudExplainer:
    def __init__(self, model_path: Path | str = MODELS_DIR / "xgb_model.json") -> None:
        import shap  # imported here so the module loads even before shap is installed

        self.booster = xgb.Booster()
        self.booster.load_model(str(model_path))
        self.explainer = shap.TreeExplainer(self.booster)
        self.feature_names = self.booster.feature_names

    def explain_row(self, x: pd.DataFrame, top_n: int = 3) -> dict:
        """x: single-row DataFrame with the model's feature columns.

        Returns {top_features: [...], reason: "..."}.
        """
        x = x.reindex(columns=self.feature_names)
        sv = self.explainer.shap_values(x)
        sv = np.asarray(sv).reshape(-1)

        order = np.argsort(sv)[::-1]  # most fraud-pushing first
        top = []
        for i in order[:top_n]:
            if sv[i] <= 0:
                break
            top.append((self.feature_names[i], float(sv[i])))

        phrases = []
        for feat, _ in top:
            p = _phrase(feat)
            if p not in phrases:
                phrases.append(p)

        if not phrases:
            reason = "No single factor stood out; risk comes from a mix of weak signals."
        elif len(phrases) == 1:
            reason = (f"Transaction flagged because its {phrases[0]} "
                      f"significantly differs from normal patterns.")
        else:
            head = ", ".join(phrases[:-1])
            reason = (f"Transaction flagged because its {head} and {phrases[-1]} "
                      f"significantly differ from normal transaction patterns.")

        return {"top_features": top, "reason": reason}


if __name__ == "__main__":
    # smoke test on a few validation rows
    from data_load import load_split
    from features import FeatureBuilder

    fb = FeatureBuilder.load()
    df = load_split("train").tail(2000)
    X, y = fb.transform(df)

    ex = FraudExplainer()
    # explain a couple of the highest-risk rows
    probs = 1 / (1 + np.exp(-ex.booster.predict(
        xgb.DMatrix(X.reindex(columns=ex.feature_names)), output_margin=True)))
    for idx in np.argsort(probs)[-3:]:
        row = X.iloc[[idx]]
        out = ex.explain_row(row)
        print(f"prob={probs[idx]:.3f}  {out['reason']}")
        print(f"   top: {[(f, round(v,3)) for f,v in out['top_features']]}")
