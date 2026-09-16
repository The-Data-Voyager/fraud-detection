"""
Phase 3 - Feature engineering for IEEE-CIS Fraud Detection.

Grounded in the Phase 2 numbers:
  * 3.5% fraud, 28:1  -> handled at model level (scale_pos_weight), not here
  * TransactionDT sorted over 182 days -> time features + causal aggregations
  * card1 (13.5k) high cardinality -> frequency encoding, card1_addr1 entity key
  * heavy missingness in id_/dist/D groups -> missingness indicators

Design: a picklable FeatureBuilder with fit()/transform(). The SAME object
is fit on train, saved to models/, then loaded by the Kafka consumer to
transform live transactions identically. Encoders (frequency maps, category
codes) learned on train are frozen and reused downstream.

Causality: velocity / time-since-last / amount z-score are computed with
sorted groupby + shift/expanding so a row only ever sees earlier rows of
the same entity. No future leakage, and the logic is reproducible per-row
in streaming when the consumer maintains per-entity running state.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Categorical columns we frequency-encode (high-card or ID-like).
FREQ_COLS = [
    "card1", "card2", "card3", "card5", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "DeviceInfo", "id_30", "id_31", "id_33",
    "card1_addr1",  # engineered entity key
]
# Low-card categoricals we label-encode to integer codes.
LABEL_COLS = [
    "ProductCD", "card4", "card6", "DeviceType",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]
# Groups whose missingness is itself signal -> add a binary indicator.
MISSING_INDICATOR_PREFIXES = ("id_", "D", "dist", "V")

TARGET = "isFraud"


class FeatureBuilder:
    """Fit encoders on train; transform any split (or one streamed row)."""

    def __init__(self) -> None:
        self.freq_maps: dict[str, dict] = {}
        self.label_maps: dict[str, dict] = {}
        self.feature_names_: list[str] | None = None
        self.fitted_ = False

    # ---- entity key + email helpers -----------------------------------
    @staticmethod
    def _add_entity_keys(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["card1_addr1"] = (
            df["card1"].astype("string").fillna("na")
            + "_"
            + df["addr1"].astype("string").fillna("na")
        )
        return df

    @staticmethod
    def _time_features(df: pd.DataFrame) -> pd.DataFrame:
        # TransactionDT is seconds from an (unknown) reference start.
        dt = df["TransactionDT"].astype("float64")
        df["dt_hour"] = ((dt / 3600) % 24).astype("float32")
        df["dt_dayofweek"] = ((dt / (3600 * 24)) % 7).astype("float32")
        df["dt_day"] = (dt / (3600 * 24)).astype("float32")
        return df

    @staticmethod
    def _amount_features(df: pd.DataFrame) -> pd.DataFrame:
        amt = df["TransactionAmt"].astype("float64")
        df["amt_log"] = np.log1p(amt).astype("float32")
        # cents / decimal part: fraud rings often use round or oddly-precise amounts
        df["amt_cents"] = ((amt - np.floor(amt)) * 100).round().astype("float32")
        df["amt_is_round"] = (df["amt_cents"] == 0).astype("int8")
        return df

    @staticmethod
    def _email_features(df: pd.DataFrame) -> pd.DataFrame:
        p = df["P_emaildomain"].astype("string")
        r = df["R_emaildomain"].astype("string")
        df["email_p_suffix"] = p.str.split(".").str[-1]
        df["email_same_pr"] = (p == r).fillna(False).astype("int8")
        df["email_r_missing"] = r.isna().astype("int8")
        return df

    @staticmethod
    def _causal_entity_aggregations(df: pd.DataFrame) -> pd.DataFrame:
        """Per card1_addr1, using only earlier rows (sorted by DT).

        - txn_count_sofar: how many prior txns this entity had (velocity proxy)
        - secs_since_last: time gap to this entity's previous txn
        - amt_z_entity: z-score of amount vs entity's prior amounts
        """
        df = df.sort_values("TransactionDT", kind="stable")
        g = df.groupby("card1_addr1", sort=False)

        df["ent_txn_count_sofar"] = g.cumcount().astype("float32")

        prev_dt = g["TransactionDT"].shift(1)
        df["ent_secs_since_last"] = (df["TransactionDT"] - prev_dt).astype("float32")

        # expanding mean/std of amount EXCLUDING current row (shift then expanding)
        amt = df["TransactionAmt"].astype("float64")
        prior_mean = g["TransactionAmt"].apply(
            lambda s: s.shift(1).expanding().mean()
        ).reset_index(level=0, drop=True)
        prior_std = g["TransactionAmt"].apply(
            lambda s: s.shift(1).expanding().std()
        ).reset_index(level=0, drop=True)
        df["ent_amt_z"] = ((amt - prior_mean) / prior_std.replace(0, np.nan)).astype("float32")
        return df

    # ---- encoders -----------------------------------------------------
    def _fit_encoders(self, df: pd.DataFrame) -> None:
        for col in FREQ_COLS:
            if col in df.columns:
                vc = df[col].astype("string").value_counts(dropna=True)
                self.freq_maps[col] = (vc / len(df)).to_dict()
        for col in LABEL_COLS:
            if col in df.columns:
                cats = df[col].astype("string").dropna().unique()
                self.label_maps[col] = {v: i + 1 for i, v in enumerate(sorted(cats))}

    def _apply_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, m in self.freq_maps.items():
            if col in df.columns:
                df[col + "_freq"] = df[col].astype("string").map(m).astype("float32").fillna(0.0)
        for col, m in self.label_maps.items():
            if col in df.columns:
                df[col + "_code"] = df[col].astype("string").map(m).astype("float32").fillna(0.0)
        return df

    @staticmethod
    def _missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
        # one indicator per high-missing group (not per column, to stay compact)
        for prefix in MISSING_INDICATOR_PREFIXES:
            cols = [c for c in df.columns if c.startswith(prefix) and not c.endswith("_missing")]
            if cols:
                df[f"grp_{prefix.strip('_')}_missing_frac"] = df[cols].isna().mean(axis=1).astype("float32")
        return df

    # ---- public API ---------------------------------------------------
    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_entity_keys(df)
        df = self._time_features(df)
        df = self._amount_features(df)
        df = self._email_features(df)
        df = self._causal_entity_aggregations(df)
        df = self._missing_indicators(df)
        df = self._apply_encoders(df)
        return df

    def _select_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        # drop raw string/category columns that we've already encoded; XGBoost
        # eats the rest (NaN-aware). Keep numeric originals + engineered.
        drop = set(FREQ_COLS + LABEL_COLS + [
            "email_p_suffix",  # string, encoded implicitly by email_* flags only
            "TransactionDT",   # absolute time: test/stream is strictly later -> extrapolation
            "dt_day",          # absolute day index, same extrapolation risk (kept dt_hour/dow)
        ])
        keep = [
            c for c in df.columns
            if c not in drop
            and c not in (TARGET, "TransactionID")
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        return df[keep]

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        df = self._add_entity_keys(df)
        self._fit_encoders(df)
        self.fitted_ = True
        return self.transform(df, _keys_added=True)

    def transform(self, df: pd.DataFrame, _keys_added: bool = False):
        if not self.fitted_:
            raise RuntimeError("call fit_transform first (or load a fitted builder)")
        if not _keys_added:
            df = self._add_entity_keys(df)
        # engineer everything except entity-key re-add
        df = self._time_features(df)
        df = self._amount_features(df)
        df = self._email_features(df)
        df = self._causal_entity_aggregations(df)
        df = self._missing_indicators(df)
        df = self._apply_encoders(df)

        y = df[TARGET].astype("int8") if TARGET in df.columns else None
        X = self._select_numeric(df)
        if self.feature_names_ is None:
            self.feature_names_ = list(X.columns)
        else:
            # align to training columns (streaming rows must match exactly)
            X = X.reindex(columns=self.feature_names_)
        return X, y

    # ---- persistence --------------------------------------------------
    def save(self, path: Path | str = MODELS_DIR / "feature_builder.pkl") -> Path:
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path: Path | str = MODELS_DIR / "feature_builder.pkl") -> "FeatureBuilder":
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    from data_load import load_split

    train = load_split("train")
    fb = FeatureBuilder()
    X, y = fb.fit_transform(train)
    print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"Positive rate: {y.mean():.4f}")
    eng = [c for c in X.columns if any(
        c.startswith(p) for p in ("dt_", "amt_", "email_", "ent_", "grp_")
    ) or c.endswith(("_freq", "_code"))]
    print(f"Engineered features added: {len(eng)}")
    print("  e.g.:", eng[:20])
    fb.save()
    print(f"Saved builder -> {MODELS_DIR / 'feature_builder.pkl'}")
