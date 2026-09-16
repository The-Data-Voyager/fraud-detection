"""
Phase 2 - EDA & leakage audit for IEEE-CIS Fraud Detection.

Produces the real numbers that ground Phase 3 feature engineering:
  * class imbalance (fraud rate)
  * missingness by column group (C/D/M/V/id/card/addr/dist/email)
  * cardinality of the entity-key candidates (card1..6, addr1/2, emails)
  * TransactionDT span & ordering sanity (time-respecting CV depends on it)
  * simple leakage smell test: columns whose non-null presence alone
    separates fraud from non-fraud almost perfectly.

Writes a markdown report to reports/phase2_eda.md and prints a summary.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_load import load_split, PROJECT_ROOT

REPORT_PATH = PROJECT_ROOT / "reports" / "phase2_eda.md"

GROUP_PREFIXES = {
    "C (counting)": "C",
    "D (timedelta)": "D",
    "M (match flags)": "M",
    "V (Vesta engineered)": "V",
    "id (identity)": "id_",
    "card": "card",
    "addr": "addr",
    "dist": "dist",
}


def _group_missingness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for label, prefix in GROUP_PREFIXES.items():
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            continue
        miss = df[cols].isna().mean()
        rows.append(
            {
                "group": label,
                "n_cols": len(cols),
                "mean_missing_%": round(100 * miss.mean(), 1),
                "min_missing_%": round(100 * miss.min(), 1),
                "max_missing_%": round(100 * miss.max(), 1),
            }
        )
    return pd.DataFrame(rows)


def _cardinality(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "P_emaildomain", "R_emaildomain", "ProductCD",
        "DeviceType", "DeviceInfo",
    ]
    rows = []
    for k in keys:
        if k in df.columns:
            rows.append(
                {
                    "column": k,
                    "nunique": int(df[k].nunique(dropna=True)),
                    "missing_%": round(100 * df[k].isna().mean(), 1),
                }
            )
    return pd.DataFrame(rows)


def _leakage_smell(df: pd.DataFrame, target: str = "isFraud", top: int = 15) -> pd.DataFrame:
    """For each column, does 'is this value missing?' predict fraud?

    A column where presence/absence alone strongly correlates with the
    label is a leakage candidate worth eyeballing (the classic PaySim-style
    trap). We rank by |fraud_rate(present) - fraud_rate(absent)|.
    """
    base = df[target].mean()
    rows = []
    for col in df.columns:
        if col in (target, "TransactionID", "TransactionDT"):
            continue
        na = df[col].isna()
        if na.all() or (~na).all():
            continue
        fr_present = df.loc[~na, target].mean()
        fr_absent = df.loc[na, target].mean()
        rows.append(
            {
                "column": col,
                "fraud_if_present_%": round(100 * fr_present, 2),
                "fraud_if_absent_%": round(100 * fr_absent, 2),
                "gap_pp": round(100 * abs(fr_present - fr_absent), 2),
            }
        )
    out = pd.DataFrame(rows).sort_values("gap_pp", ascending=False).head(top)
    out.attrs["baseline_%"] = round(100 * base, 3)
    return out


def main() -> None:
    df = load_split("train")
    target = "isFraud"

    n = len(df)
    fraud_rate = df[target].mean()

    # time span
    dt = df["TransactionDT"]
    dt_span_days = (dt.max() - dt.min()) / (60 * 60 * 24)
    is_sorted = dt.is_monotonic_increasing

    miss = _group_missingness(df)
    card = _cardinality(df)
    leak = _leakage_smell(df)

    # amount stats
    amt = df["TransactionAmt"]
    amt_by_class = df.groupby(target)["TransactionAmt"].agg(["mean", "median", "max"])

    lines = []
    w = lines.append
    w("# Phase 2 - EDA & Leakage Audit (train)\n")
    w(f"- Rows: **{n:,}**, Columns: **{df.shape[1]}**")
    w(f"- Fraud rate (class imbalance): **{100*fraud_rate:.2f}%** "
      f"({int(df[target].sum()):,} of {n:,})  ->  ~{(1-fraud_rate)/fraud_rate:.0f}:1 negative:positive")
    w(f"- TransactionDT span: **{dt_span_days:.1f} days**, monotonic increasing: **{is_sorted}**")
    w(f"  -> use TransactionDT ordering for time-respecting CV, NOT random splits.\n")

    w("## Transaction amount by class")
    w(amt_by_class.round(2).to_markdown())
    w("")

    w("## Missingness by column group")
    w(miss.to_markdown(index=False))
    w("")

    w("## Cardinality of entity-key / categorical candidates")
    w(card.to_markdown(index=False))
    w("")

    w(f"## Leakage smell test (baseline fraud = {leak.attrs['baseline_%']}%)")
    w("Columns where *presence vs absence of a value* most separates fraud. "
      "Large gaps are candidates to inspect before trusting them as features.\n")
    w(leak.to_markdown(index=False))
    w("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    # console summary
    print("\n" + "=" * 60)
    print(f"Fraud rate: {100*fraud_rate:.2f}%  ({int(df[target].sum()):,}/{n:,})")
    print(f"TransactionDT span: {dt_span_days:.1f} days | sorted: {is_sorted}")
    print(f"Report written -> {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
