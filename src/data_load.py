"""
Phase 1 - Data loading & merge for the IEEE-CIS Fraud Detection project.

Loads train/test transaction + identity CSVs with aggressive dtype
downcasting, left-joins identity onto transaction by TransactionID, and
caches the merged frame to Parquet so the ~1.3 GB of CSVs are only paid
for once.

Usage:
    from src.data_load import load_split
    train = load_split("train")   # merged, downcast, cached
    test  = load_split("test")
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "DATA" / "ieee-fraud-detection"
CACHE_DIR = PROJECT_ROOT / "DATA" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink numeric columns in place to the smallest safe dtype.

    Integers -> smallest signed/unsigned int that holds the range.
    Floats   -> float32 (fraud features don't need float64 precision).
    Low-cardinality object columns -> category.
    """
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            df[col] = s.astype(np.float32)
        elif pd.api.types.is_object_dtype(s):
            # turn repetitive strings (card4, ProductCD, emails...) into categories
            if s.nunique(dropna=True) / max(len(s), 1) < 0.5:
                df[col] = s.astype("category")
    return df


def _read_csv_downcast(path: Path) -> pd.DataFrame:
    """Read a CSV and immediately downcast. TransactionID stays int32."""
    df = pd.read_csv(path)
    df = _downcast(df)
    return df


def load_split(split: str, use_cache: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Load and merge one split ('train' or 'test').

    Left-joins <split>_identity onto <split>_transaction on TransactionID.
    Result is cached to Parquet; subsequent calls read the Parquet.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    cache_path = CACHE_DIR / f"{split}_merged.parquet"
    if use_cache and cache_path.exists():
        if verbose:
            print(f"[load_split] reading cache {cache_path.name}")
        return pd.read_parquet(cache_path)

    t0 = time.time()
    txn_path = DATA_DIR / f"{split}_transaction.csv"
    idy_path = DATA_DIR / f"{split}_identity.csv"

    if verbose:
        print(f"[load_split] reading {txn_path.name} ...")
    txn = _read_csv_downcast(txn_path)

    if verbose:
        print(f"[load_split] reading {idy_path.name} ...")
    idy = _read_csv_downcast(idy_path)

    # test_identity in this dataset uses 'id-01' style names; normalize to 'id_01'
    idy.columns = [c.replace("-", "_") for c in idy.columns]

    merged = txn.merge(idy, how="left", on="TransactionID")
    del txn, idy

    if verbose:
        mem = merged.memory_usage(deep=True).sum() / 1024**2
        print(
            f"[load_split] {split}: {merged.shape[0]:,} rows x {merged.shape[1]} cols, "
            f"{mem:,.0f} MB in RAM, loaded in {time.time()-t0:.1f}s"
        )

    if use_cache:
        merged.to_parquet(cache_path, index=False)
        if verbose:
            print(f"[load_split] cached -> {cache_path.name}")

    return merged


if __name__ == "__main__":
    for sp in ("train", "test"):
        df = load_split(sp)
        print(f"  {sp}: {df.shape}")
