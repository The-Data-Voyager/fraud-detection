"""
Phase 7 - Stream consumer (Kafka backend).

Reads transaction messages from the Kafka topic, scores each with the shared
StreamScorer (model + SHAP), and writes the scored result to the database.

For a broker-free demo, use run_stream_sim.py instead (same scoring + DB
write, driven by the generator producer in-process).

Env:
  KAFKA_BOOTSTRAP   e.g. localhost:9092
  FRAUD_TOPIC       default 'transactions'
  FRAUD_DB_URL      db.py connection URL (defaults to local SQLite)
"""
from __future__ import annotations

import json
import os

import db
from scorer import StreamScorer
from stream_producer import FRAUD_TOPIC


def process_record(scorer: StreamScorer, rec: dict) -> dict:
    """Score one raw transaction dict and persist it. Returns the scored row."""
    scored = scorer.score(rec)
    db.insert_scored(scored)
    return scored


def main() -> None:
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP")
    if not bootstrap:
        raise SystemExit(
            "KAFKA_BOOTSTRAP not set. For a broker-free demo run: python run_stream_sim.py"
        )
    from kafka import KafkaConsumer  # type: ignore

    db.init_db()
    scorer = StreamScorer(with_explainer=True)
    consumer = KafkaConsumer(
        FRAUD_TOPIC,
        bootstrap_servers=bootstrap,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="fraud-scorer",
    )
    print(f"consuming '{FRAUD_TOPIC}' ...")
    n = 0
    for msg in consumer:
        scored = process_record(scorer, msg.value)
        n += 1
        if scored["status"] == "HIGH RISK":
            print(f"  [ALERT] #{scored['transaction_id']} "
                  f"score={scored['risk_score']} :: {scored['reason']}")
        if n % 200 == 0:
            print(f"  consumed {n}")


if __name__ == "__main__":
    main()
