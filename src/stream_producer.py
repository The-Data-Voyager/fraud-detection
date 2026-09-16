"""
Phase 7 - Stream producer.

Reads transactions in TransactionDT (arrival) order and emits them one by one
to simulate a live feed. Two backends:

  * Kafka (plan target): if KAFKA_BOOTSTRAP is set and kafka-python is
    installed, publishes JSON messages to topic FRAUD_TOPIC.
  * Generator: iter_transactions() yields raw dicts, used by the in-process
    simulator (run_stream_sim.py) so the pipeline runs with no broker.

Env:
  KAFKA_BOOTSTRAP   e.g. localhost:9092   (unset -> generator only)
  FRAUD_TOPIC       default 'transactions'
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterator

import numpy as np
import pandas as pd

from data_load import load_split

FRAUD_TOPIC = os.environ.get("FRAUD_TOPIC", "transactions")


def iter_transactions(
    source: str = "test",
    limit: int | None = 500,
    delay: float = 0.0,
) -> Iterator[dict]:
    """Yield raw transaction dicts in TransactionDT order.

    source: 'test' (live-like, unlabeled) or 'train' (labeled, for demos).
    limit:  cap number of rows (None = all).
    delay:  seconds to sleep between rows (simulated arrival rate).
    """
    df = load_split(source)
    df = df.sort_values("TransactionDT", kind="stable")
    if limit is not None:
        df = df.head(limit)
    # JSON-safe: NaN -> None
    df = df.replace({np.nan: None})
    for rec in df.to_dict(orient="records"):
        yield rec
        if delay:
            time.sleep(delay)


def _json_safe(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, float) and (v != v):  # NaN
            out[k] = None
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main() -> None:
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP")
    if not bootstrap:
        raise SystemExit(
            "KAFKA_BOOTSTRAP not set. For a broker-free demo run: python run_stream_sim.py"
        )
    from kafka import KafkaProducer  # type: ignore

    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    n = 0
    for rec in iter_transactions(source="test", limit=None, delay=0.05):
        producer.send(FRAUD_TOPIC, _json_safe(rec))
        n += 1
        if n % 500 == 0:
            print(f"  produced {n} messages")
    producer.flush()
    print(f"done, produced {n} messages to '{FRAUD_TOPIC}'")


if __name__ == "__main__":
    main()
